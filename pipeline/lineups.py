"""Predicted/confirmed starting-lineup ingestion from RotoWire.

Why this source: the design doc explicitly deferred both "betting odds as a
start-probability proxy" and "scraped predicted lineups (RotoWire, FF
Scout)" as nice-to-haves. The trailing-minutes proxy this pipeline shipped
with turned out to fail exactly the case that mattered most in practice —
summer transfers (a player who was a clear starter at their old club reads
as a near-certain starter at their new one, even when they're now third
choice) — because it has no visibility into the player's *current* squad
role. A predicted-lineup source is the direct fix: it's the market/analyst
view of who's actually in the XI right now, not an extrapolation from old
box scores.

FBref and Oddschecker are both walled off behind an active Cloudflare/bot
challenge (see fbref.py); RotoWire is not the same situation — the lineups
page returns a normal 200 with server-rendered HTML (no JS execution
needed), `robots.txt` doesn't disallow it, and RotoWire's own published
`llms.txt` scopes exactly this kind of content ("Daily Lineups", "Depth
Charts") as intended for LLM/agent use. This is a plain content fetch, not
a bypass.

Coverage caveat: predicted lineups exist for whichever gameweek is next up
— useful right now for GW1, but there's no lineup data for gameweeks
further out. Horizons 2+ still need the trailing-minutes fallback.
"""
import logging
import re
from datetime import date

import pandas as pd
import requests
from bs4 import BeautifulSoup

from pipeline.config import REQUEST_TIMEOUT, USER_AGENT
from pipeline.matching import normalize_name, normalize_team
from pipeline.storage import save_raw

logger = logging.getLogger(__name__)

ROTOWIRE_URL = "https://www.rotowire.com/soccer/lineups.php"
HEADERS = {"User-Agent": USER_AGENT}

STARTER_PROB = {"confirmed": 0.97, "predicted": 0.90}
BENCH_PROB = 0.12  # team has a published lineup and this player isn't in it
NOT_LISTED_AT_ALL = None  # team has no lineup published yet -> caller should fall back

# A predicted starter who's also carrying an injury tag gets scaled down
# rather than trusted at face value — the site lists a projected XI and
# flags fitness doubts on top of it, not instead of it.
INJURY_MULTIPLIER = {"OUT": 0.05, "SUSP": 0.05, "QUES": 0.6, "GTD": 0.75, "PROB": 0.9}


def fetch_page() -> str:
    resp = requests.get(ROTOWIRE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _parse_status(box) -> str:
    status_el = box.find(class_="lineup__status")
    text = status_el.get_text(strip=True).lower() if status_el else ""
    return "confirmed" if "confirmed" in text else "predicted"


def _parse_side(lst, side_label: str) -> list[dict]:
    """First 11 <li> entries in a lineup__list are the starting XI; any
    entries after that are the team's injury-report callouts (which repeat
    a subset of names with a simplified position code), not more starters.

    Before RotoWire publishes a side's XI, its list holds only those
    callouts (e.g. 3 names, some tagged OUT). Fewer than 11 entries means no
    lineup yet: return nothing, so the side neither supplies "starters" nor
    marks the rest of its squad as benched.
    """
    items = lst.find_all(class_="lineup__player")
    if len(items) < 11:
        return []
    starters = items[:11]
    rows = []
    for li in starters:
        name_el = li.find("a")
        name = name_el.get_text(strip=True) if name_el else li.get_text(strip=True)
        inj_el = li.find(class_="lineup__inj")
        injury = inj_el.get_text(strip=True) if inj_el else None
        rows.append({"player_name": name, "side": side_label, "injury": injury})
    return rows


def parse_lineups(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    boxes = soup.find_all("div", class_="lineup__box")

    rows = []
    for box in boxes:
        mteams = box.find_all(class_="lineup__mteam")
        if len(mteams) != 2:
            continue
        home_team = mteams[0].get_text(strip=True)
        away_team = mteams[1].get_text(strip=True)
        status = _parse_status(box)

        lists = box.find_all(class_="lineup__list")
        if len(lists) != 2:
            # No lineup published yet for this fixture — skip rather than guess.
            continue
        home_players = _parse_side(lists[0], "home")
        away_players = _parse_side(lists[1], "away")

        for p in home_players:
            rows.append({**p, "team": home_team, "opponent": away_team, "was_home": True, "status": status})
        for p in away_players:
            rows.append({**p, "team": away_team, "opponent": home_team, "was_home": False, "status": status})

    df = pd.DataFrame(rows)
    logger.info(
        "parsed RotoWire lineups: %d starters across %d fixtures with published lineups",
        len(df), df[["team", "opponent"]].drop_duplicates().shape[0] // 1 if len(df) else 0,
    )
    return df


def match_to_fpl(lineups_df: pd.DataFrame, live_players: pd.DataFrame, live_teams: pd.DataFrame) -> pd.DataFrame:
    """Name+team match, same token-containment approach as understat.py
    (handles RotoWire's abbreviated-first-name style like "R. Calafiori").
    """
    if lineups_df.empty:
        return lineups_df.assign(code=pd.Series(dtype="int64"))

    team_name_by_id = live_teams.set_index("id")["name"]
    lp = live_players.copy()
    lp["fpl_team_name"] = lp["team"].map(team_name_by_id)
    lp["tokens_full"] = (lp["first_name"] + " " + lp["second_name"]).apply(
        lambda s: set(normalize_name(s).split())
    )
    lp["tokens_web"] = lp["web_name"].apply(lambda s: set(normalize_name(s).split()))
    lp["norm_team"] = lp["fpl_team_name"].apply(normalize_team)

    def tokens_compatible(a: set, b: set) -> bool:
        # RotoWire often abbreviates the first name to a single initial
        # ("V. van Dijk"); a bare single-letter token matches if some token
        # on the other side starts with that letter, not just full equality.
        if not a or not b:
            return False
        if a <= b or b <= a:
            return True
        a_rest = {t for t in a if len(t) > 1}
        a_initials = {t for t in a if len(t) == 1}
        b_rest = {t for t in b if len(t) > 1}
        b_initials = {t for t in b if len(t) == 1}
        if a_rest and not (a_rest <= b_rest or b_rest <= a_rest):
            return False
        if not a_rest and not b_rest:
            return False
        initials, full_side = (a_initials, b_rest) if a_initials else (b_initials, a_rest)
        return all(any(w.startswith(i) for w in full_side) for i in initials) if initials else True

    def overlap_score(target: set, full: set, web: set) -> int:
        # Squad-mates who share one name token (players who share a first
        # name, or a surname that's a strict prefix of a hyphenated one —
        # "Gabriel" x3 on one Arsenal squad, "James" vs "James Justin" on
        # one Leeds squad) all pass tokens_compatible individually, since
        # a short, real identity being a *subset* of the target is exactly
        # what the initial-abbreviation handling above is designed to
        # allow. Total token overlap across both the full name and
        # web_name breaks that tie correctly: the actual target player
        # matches on both counts, a same-first-name-only squadmate
        # matches on just the one shared token.
        return len(target & full) + len(target & web)

    matched_rows = []
    for row in lineups_df.itertuples():
        team_pool = lp[lp["norm_team"] == normalize_team(row.team)]
        target_tokens = set(normalize_name(row.player_name).split())
        hits = team_pool[
            team_pool["tokens_full"].apply(lambda t: tokens_compatible(t, target_tokens))
            | team_pool["tokens_web"].apply(lambda t: tokens_compatible(t, target_tokens))
        ]
        if len(hits) == 1:
            matched_rows.append({**row._asdict(), "code": hits.iloc[0]["code"]})
        elif len(hits) > 1:
            scores = hits.apply(
                lambda r: overlap_score(target_tokens, r["tokens_full"], r["tokens_web"]), axis=1
            )
            top = scores[scores == scores.max()]
            if len(top) == 1:
                matched_rows.append({**row._asdict(), "code": hits.loc[top.index[0], "code"]})
                logger.debug(
                    "resolved ambiguous RotoWire match: %s (%s) -> %s via token-overlap tiebreak",
                    row.player_name, row.team, hits.loc[top.index[0], "web_name"],
                )
            else:
                logger.debug(
                    "ambiguous RotoWire match: %s (%s), %d candidates, unresolved",
                    row.player_name, row.team, len(hits),
                )

    result = pd.DataFrame(matched_rows).drop(columns=["Index"], errors="ignore")
    logger.info("matched %d/%d RotoWire lineup entries to FPL players", len(result), len(lineups_df))
    return result


def build_start_probability(matched_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (code, team) with a start-probability estimate derived
    purely from lineup presence/status/injury tags — no trailing stats.
    """
    if matched_df.empty:
        return matched_df

    def prob_for(row):
        base = STARTER_PROB[row["status"]]
        if pd.notna(row["injury"]):
            tag = row["injury"].upper()
            base *= INJURY_MULTIPLIER.get(tag, 0.8)
        return base

    matched_df = matched_df.copy()
    matched_df["p_plays_60_lineup"] = matched_df.apply(prob_for, axis=1)
    return matched_df[["code", "team", "opponent", "was_home", "status", "injury", "p_plays_60_lineup"]]


def ingest() -> pd.DataFrame:
    html = fetch_page()
    lineups_df = parse_lineups(html)
    save_raw(lineups_df, "rotowire_lineups_raw")
    # `rotowire_lineups_raw` above gets overwritten on every fetch (save_raw
    # has no versioning), so nothing about a past gameweek's lineup call
    # survives to be checked against what actually happened. This dated
    # snapshot is purely additive (never read by the live pipeline) so that
    # a real RotoWire-vs-trailing-fallback accuracy comparison becomes
    # possible once enough gameweeks have accumulated. Date-grained, not
    # per-fetch: a lineup can flip from "predicted" to "confirmed" status
    # intraday, but the daily snapshot closest to kickoff is what a real
    # backtest would have had to work with anyway.
    save_raw(lineups_df, f"rotowire_lineups_history_{date.today().isoformat()}")
    return lineups_df


def get_start_probabilities(
    live_players: pd.DataFrame, live_teams: pd.DataFrame
) -> tuple[pd.DataFrame, set]:
    """Convenience entrypoint for callers that just want (per-player start
    probability, set of team names with a published lineup) — fetches,
    parses, matches, and scores in one call. Returns an empty frame and
    empty set (not an exception) if the fetch fails, so a caller can fall
    back to the trailing-minutes proxy without special-casing network
    errors.
    """
    try:
        lineups_df = ingest()
    except requests.RequestException as exc:
        logger.warning("RotoWire lineup fetch failed (%s); no lineup override this run", exc)
        return pd.DataFrame(), set()

    if lineups_df.empty:
        return pd.DataFrame(), set()

    teams_with_lineup = {normalize_team(t) for t in lineups_df["team"].unique()}
    matched = match_to_fpl(lineups_df, live_players, live_teams)
    probs = build_start_probability(matched)
    return probs, teams_with_lineup


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = ingest()
    print(df.head(30).to_string())
