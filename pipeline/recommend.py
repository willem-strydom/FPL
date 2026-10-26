"""Weekly recommendation: the multi-period planner's transfers for your
squad, plus the planner's wildcard squad.

    python -m pipeline.recommend            # use cached data
    python -m pipeline.recommend --refresh  # pull fresh live + current-season data first
    python -m pipeline.recommend --entry 1234567   # any team, by FPL team id
    python -m pipeline.recommend --entry https://fantasy.premierleague.com/entry/1234567/event/6
    python -m pipeline.recommend --keep "De Cuyper" --keep "Thomas (Coventry City)"   # never sell these

Without --entry, my_team.json at the repo root says whose squad to plan for, either
    {"entry_id": 7592422}
which pulls squad, bank, free transfers and selling prices from the FPL API
(see fpl_entry), or a manual fallback
    {"squad": ["Haaland", "Thomas (Coventry City)", ...], "free_transfers": 1, "bank": 0.5}
with FPL web names ("(Team)" only where a name is ambiguous), priced at
current buy prices.
"""
import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from pipeline.chip_policy import HALF_DEADLINES, deadline_for, decide
from pipeline.fpl_entry import fetch_entry_state
from pipeline.optimizer import PLANNER_HORIZON, load_squad_rules, plan_transfers, select_squad
from pipeline.predict_next_gw import POSITION_NAME, build_enabler_candidates, get_next_gameweek, predict_multi_gw
from pipeline.storage import load_raw

TEAM_FILE = Path(__file__).resolve().parent.parent / "my_team.json"
BASE_COLS = ["code", "web_name", "team_name", "position_name", "price"]

# Chip timing: optimal stopping over the chips still held this half (see
# chip_policy). Each chip is played only if this week's gain beats the value
# of keeping it, which falls as the half's deadline nears.
CHIP_LABELS = {"wildcard": "Wildcard", "freehit": "Free Hit", "bboost": "Bench Boost", "3xc": "Triple Captain"}


def parse_entry_id(value: str) -> int:
    """An FPL team id, or any fantasy.premierleague.com link containing
    /entry/<id>/ (the points page, transfers page, history page...)."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    match = re.search(r"/entry/(\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError(f"expected an FPL team id or a link containing /entry/<id>/, got {value!r}")
    return int(match.group(1))


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def live_player_table() -> pd.DataFrame:
    lp = load_raw("fpl_live_players")
    teams = load_raw("fpl_live_teams").set_index("id")["name"]
    return pd.DataFrame({
        "code": lp["code"], "web_name": lp["web_name"], "team_name": lp["team"].map(teams),
        "position_name": lp["element_type"].map(POSITION_NAME), "price": lp["now_cost"] / 10,
        "status": lp["status"], "news": lp["news"],
    })


def resolve_squad(names: list[str], players: pd.DataFrame) -> set:
    codes = set()
    for raw in names:
        name, _, team = raw.partition("(")
        hits = players[players["web_name"].map(_norm) == _norm(name)]
        if team:
            hits = hits[hits["team_name"].map(_norm) == _norm(team.rstrip(")"))]
        if len(hits) != 1:
            options = players[players["web_name"].map(_norm) == _norm(name)][["web_name", "team_name"]].values.tolist()
            raise ValueError(f"'{raw}' matched {len(hits)} players {options}; add '(Team)' to disambiguate")
        codes.add(int(hits["code"].iloc[0]))
    if len(codes) != 15:
        raise ValueError(f"squad must have 15 players, got {len(codes)}")
    return codes


def resolve_keep(names: list[str], players: pd.DataFrame, squad: set) -> set:
    """--keep names -> codes, matched among the owned players only."""
    if not names:
        return set()
    owned = players[players["code"].isin(squad)]
    codes = set()
    for raw in names:
        name, _, team = raw.partition("(")
        hits = owned[owned["web_name"].map(_norm) == _norm(name)]
        if team:
            hits = hits[hits["team_name"].map(_norm) == _norm(team.rstrip(")"))]
        if len(hits) != 1:
            raise ValueError(f"--keep '{raw}' matched {len(hits)} owned players; owned: {sorted(owned['web_name'])}")
        codes.add(int(hits["code"].iloc[0]))
    return codes


def planning_pool(squad: set, players: pd.DataFrame, horizon: int) -> pd.DataFrame:
    horizons = tuple(range(1, horizon + 1))
    gw_cols = [f"predicted_points_gw{h}" for h in horizons]
    scored = predict_multi_gw(horizons=horizons)
    enablers = build_enabler_candidates(horizons=horizons)
    pool = pd.concat(
        [scored[BASE_COLS + gw_cols], enablers[~enablers["code"].isin(scored["code"])][BASE_COLS + gw_cols]],
        ignore_index=True,
    )
    # Owned players outside the scored pool (injured, unavailable, cold-start)
    # still occupy a squad slot: keep them in at zero projected points.
    missing = players[players["code"].isin(squad - set(pool["code"]))]
    if not missing.empty:
        print("owned but not projected (treated as 0 pts):",
              ", ".join(f"{r.web_name} [{r.status}] {r.news}".strip() for r in missing.itertuples()))
        pool = pd.concat([pool, missing[BASE_COLS]], ignore_index=True)
    pool[gw_cols] = pool[gw_cols].fillna(0.0)
    return pool


def print_plan(res: dict, title: str, first_gw: int) -> None:
    print(f"\n=== {title} ===")
    info = res["transfer_info"]
    if info["transferred_out"]:
        print(f"GW{first_gw} transfers: OUT {', '.join(info['transferred_out'])}  ->  IN {', '.join(info['transferred_in'])}"
              f"  ({info['hard_cost']:.0f} pt hit)" if info["hard_cost"] else
              f"GW{first_gw} transfers: OUT {', '.join(info['transferred_out'])}  ->  IN {', '.join(info['transferred_in'])}")
    else:
        print(f"GW{first_gw} transfers: none (roll the free transfer)")
    xi = res["starters"]
    cap, vice = res["captain"]["web_name"], res["vice_captain"]["web_name"]
    print(f"Starting XI (GW{first_gw} predicted {res['starting_predicted_points']:.1f} incl. captain):")
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        names = [f"{r.web_name}{' (C)' if r.web_name == cap else ' (V)' if r.web_name == vice else ''} {r.predicted_points:.1f}"
                 for r in xi[xi["position_name"] == pos].itertuples()]
        print(f"  {pos}: {', '.join(names)}")
    print("  Bench:", ", ".join(f"{r.web_name} {r.predicted_points:.1f}" for r in res["bench"].itertuples()))
    print(f"  Squad cost £{res['total_cost']:.1f}m")
    print("Plan:")
    for wk in res["plan"]:
        moves = (f"OUT {', '.join(wk['transfers_out'])} -> IN {', '.join(wk['transfers_in'])}"
                 if wk["transfers_in"] else "hold")
        hit = f", {wk['hits']} hit(s)" if wk["hits"] else ""
        print(f"  GW{first_gw + wk['week'] - 1}: {wk['free_transfers']} FT, {moves}{hit}, XI {wk['xi_points']:.1f}")


def chip_report(pool: pd.DataFrame, normal: dict, wild: dict, rules: dict, chips_used: dict, gw: int) -> None:
    """Each held chip's gain this week vs the value of keeping it (chip_policy);
    recommend at most one, since FPL allows one chip per gameweek."""
    free_hit = select_squad(pool.assign(predicted_points=pool["predicted_points_gw1"]), rules=rules)
    gains = {
        "wildcard": wild["objective"] - normal["objective"],
        "freehit": free_hit["starting_predicted_points"] - normal["plan"][0]["xi_points"],
        "bboost": normal["plan"][0]["bench_points"],
        "3xc": normal["plan"][0]["captain_points"],
    }
    forecasts = {"bboost": [wk["bench_points"] for wk in normal["plan"]],
                 "3xc": [wk["captain_points"] for wk in normal["plan"]]}
    deadline = deadline_for(gw)
    half_start = 1 if deadline == HALF_DEADLINES[0] else HALF_DEADLINES[0] + 1
    used = {name for event, name in chips_used.items() if half_start <= event <= deadline}
    result = decide(gains, gw, set(CHIP_LABELS) - used, forecasts)

    print(f"\n=== Chips (GW{gw}, {result['weeks_left']} week(s) left before the GW{deadline} deadline) ===")
    for chip, label in CHIP_LABELS.items():
        if chip in used:
            print(f"  {label:15s} used this half")
            continue
        r = result["chips"][chip]
        print(f"  {label:15s} gain {r['gain']:5.1f} vs value of keeping it {r['threshold']:5.1f} (margin {r['margin']:+.1f})")
    if result["play"]:
        print(f"  -> play {CHIP_LABELS[result['play']]} this week")
    else:
        print("  -> no chip beats the value of keeping it this week")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-ingest live FPL and current-season data first")
    parser.add_argument("--team", type=Path, default=TEAM_FILE)
    parser.add_argument("--entry", type=parse_entry_id,
                        help="plan for any FPL team: team id or a link to the team (overrides --team)")
    parser.add_argument("--keep", action="append", default=[], metavar="NAME",
                        help="owned player the plan must not sell (FPL web name, '(Team)' if ambiguous); repeatable")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.refresh:
        from pipeline import fpl_current_season, fpl_live
        fpl_live.ingest()
        fpl_current_season.ingest()

    team = {"entry_id": args.entry} if args.entry is not None else json.loads(args.team.read_text())
    players = live_player_table()
    if "entry_id" in team:
        state = fetch_entry_state(int(team["entry_id"]))
        squad, free_transfers, bank = state["squad"], state["free_transfers"], state["bank"]
        selling, chips_used = state["selling_prices"], state["chips_used"]
        print(f"squad from FPL entry {team['entry_id']} as of GW{state['as_of_gw']} deadline")
    else:
        squad = resolve_squad(team["squad"], players)
        free_transfers, bank = int(team["free_transfers"]), float(team.get("bank", 0.0))
        selling = players.set_index("code").loc[list(squad), "price"].to_dict()
        chips_used = {}

    rules = load_squad_rules()
    rules["budget"] = sum(selling.values()) + bank
    pool = planning_pool(squad, players, PLANNER_HORIZON)
    # Owned players are worth their selling price, not today's buy price.
    owned = pool["code"].isin(squad)
    pool.loc[owned, "price"] = pool.loc[owned, "code"].map(selling)
    gw = get_next_gameweek()
    print(f"GW{gw} | {free_transfers} free transfer(s) | bank £{bank:.1f}m | budget £{rules['budget']:.1f}m")

    keep = resolve_keep(args.keep, players, squad)
    normal = plan_transfers(pool, squad, free_transfers, horizon=PLANNER_HORIZON, rules=rules, keep=keep)
    wild = plan_transfers(pool, squad, free_transfers, horizon=PLANNER_HORIZON, rules=rules, wildcard=True, keep=keep)
    if keep:
        free = plan_transfers(pool, squad, free_transfers, horizon=PLANNER_HORIZON, rules=rules)
        names = ", ".join(players.set_index("code").loc[list(keep), "web_name"])
        print(f"keeping {names}: plan objective {normal['objective']:.1f} vs {free['objective']:.1f} unrestricted "
              f"(costs {free['objective'] - normal['objective']:.1f} decayed points over {PLANNER_HORIZON} weeks)")
    print_plan(normal, "Recommended transfers", gw)
    print_plan(wild, "Wildcard squad", gw)
    chip_report(pool, normal, wild, rules, chips_used, gw)


if __name__ == "__main__":
    main()
