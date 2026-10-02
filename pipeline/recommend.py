"""Weekly recommendation: the multi-period planner's transfers for your
squad, plus the planner's wildcard squad.

    python -m pipeline.recommend            # use cached data
    python -m pipeline.recommend --refresh  # pull fresh live + current-season data first

Your squad lives in my_team.json at the repo root:
    {"squad": ["Haaland", "Thomas (Coventry City)", ...], "free_transfers": 1, "bank": 0.5}
Names are FPL web names; add "(Team)" only where a web name is ambiguous.

Budget is your squad's current prices plus bank. FPL's real selling price
halves any price rise, so this slightly overstates what you can spend if
players have risen since you bought them.
"""
import argparse
import json
import logging
import unicodedata
from pathlib import Path

import pandas as pd

from pipeline.optimizer import PLANNER_HORIZON, load_squad_rules, plan_transfers
from pipeline.predict_next_gw import POSITION_NAME, build_enabler_candidates, get_next_gameweek, predict_multi_gw
from pipeline.storage import load_raw

TEAM_FILE = Path(__file__).resolve().parent.parent / "my_team.json"
BASE_COLS = ["code", "web_name", "team_name", "position_name", "price"]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="re-ingest live FPL and current-season data first")
    parser.add_argument("--team", type=Path, default=TEAM_FILE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.refresh:
        from pipeline import fpl_current_season, fpl_live
        fpl_live.ingest()
        fpl_current_season.ingest()

    team = json.loads(args.team.read_text())
    players = live_player_table()
    squad = resolve_squad(team["squad"], players)
    free_transfers, bank = int(team["free_transfers"]), float(team.get("bank", 0.0))

    rules = load_squad_rules()
    rules["budget"] = players.loc[players["code"].isin(squad), "price"].sum() + bank
    pool = planning_pool(squad, players, PLANNER_HORIZON)
    gw = get_next_gameweek()
    print(f"GW{gw} | {free_transfers} free transfer(s) | budget £{rules['budget']:.1f}m")

    print_plan(plan_transfers(pool, squad, free_transfers, horizon=PLANNER_HORIZON, rules=rules),
               "Recommended transfers", gw)
    print_plan(plan_transfers(pool, squad, free_transfers, horizon=PLANNER_HORIZON, rules=rules, wildcard=True),
               "Wildcard squad", gw)


if __name__ == "__main__":
    main()
