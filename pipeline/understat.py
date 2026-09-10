"""Understat ingestion: shot-based xG/xA, used as the cross-check source
for FPL's own (Opta-sourced) expected_goals/expected_assists.

FBref is the other cross-check source named in DesignDoc.md, but it sits
behind Cloudflare's JS bot-challenge (verified: a plain request gets a 403
"Just a moment..." interstitial). We do not attempt to defeat that
challenge. See pipeline/fbref.py for details and status.

Understat's data isn't behind a public REST API — its own frontend fetches
JSON from same-origin endpoints (`/getLeagueData/...`, `/getPlayerData/...`)
using a session cookie set on first page load plus a Referer header. That's
exactly what we replicate here: normal same-site AJAX behavior, no
challenge-solving or identity spoofing involved.
"""
import logging
import time

import pandas as pd
import requests

from pipeline.config import (
    REQUEST_TIMEOUT,
    UNDERSTAT_BASE,
    UNDERSTAT_LEAGUE,
    UNDERSTAT_SEASON,
    USER_AGENT,
)
from pipeline.matching import normalize_name, normalize_team
from pipeline.storage import save_raw

logger = logging.getLogger(__name__)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    # Load the league page first so the site's session cookie gets set,
    # same as a browser would before firing its own AJAX calls.
    referer = f"{UNDERSTAT_BASE}/league/{UNDERSTAT_LEAGUE}/{UNDERSTAT_SEASON}"
    s.get(referer, timeout=REQUEST_TIMEOUT)
    s.headers.update({"Referer": referer, "X-Requested-With": "XMLHttpRequest"})
    return s


def fetch_season_players(session: requests.Session | None = None) -> pd.DataFrame:
    """Season-aggregate xG/xA/shots per player for the configured league+season."""
    s = session or _session()
    url = f"{UNDERSTAT_BASE}/getLeagueData/{UNDERSTAT_LEAGUE}/{UNDERSTAT_SEASON}"
    resp = s.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["players"])
    numeric_cols = [
        "games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
        "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.info("fetched Understat season player table: %d players", len(df))
    return df


def fetch_player_matches(
    understat_player_id: str, session: requests.Session, season: int = UNDERSTAT_SEASON
) -> pd.DataFrame:
    """Per-match xG/xA log for one Understat player, filtered to one season."""
    url = f"{UNDERSTAT_BASE}/getPlayerData/{understat_player_id}"
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    matches = pd.DataFrame(payload["matches"])
    if matches.empty:
        return matches
    matches = matches[matches["season"].astype(str) == str(season)].copy()
    numeric_cols = ["goals", "shots", "xG", "time", "xA", "assists", "key_passes", "npg", "npxG"]
    for col in numeric_cols:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches["understat_player_id"] = understat_player_id
    return matches


def fetch_all_player_matches(
    season_players: pd.DataFrame, session: requests.Session, sleep_seconds: float = 0.3
) -> pd.DataFrame:
    """Per-match logs for every player in the season table.

    Sequential with a small delay between requests — this is one request per
    player (~500), and there's no bulk endpoint, so we keep the request rate
    modest rather than hammering the site.
    """
    frames = []
    for i, row in enumerate(season_players.itertuples(), start=1):
        try:
            m = fetch_player_matches(row.id, session)
        except requests.RequestException as exc:
            logger.warning("failed to fetch Understat matches for player %s: %s", row.id, exc)
            continue
        if not m.empty:
            frames.append(m)
        if i % 50 == 0:
            logger.info("fetched Understat match logs for %d/%d players", i, len(season_players))
        time.sleep(sleep_seconds)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_player_id_map(
    understat_players: pd.DataFrame, fpl_players_raw: pd.DataFrame, fpl_teams: pd.DataFrame
) -> pd.DataFrame:
    """Map Understat player ids onto FPL element ids via normalized name+team."""
    team_name_by_id = fpl_teams.set_index("id")["name"]

    fpl = fpl_players_raw.copy()
    fpl["fpl_team_name"] = fpl["team"].map(team_name_by_id)
    fpl["tokens_full"] = (fpl["first_name"] + " " + fpl["second_name"]).apply(
        lambda s: set(normalize_name(s).split())
    )
    fpl["tokens_web"] = fpl["web_name"].apply(lambda s: set(normalize_name(s).split()))
    fpl["norm_team"] = fpl["fpl_team_name"].apply(normalize_team)

    us = understat_players.copy()
    us["tokens"] = us["player_name"].apply(lambda s: set(normalize_name(s).split()))
    # A player who transferred mid-season has a comma-joined team_title.
    us["norm_teams"] = us["team_title"].apply(
        lambda t: [normalize_team(x) for x in str(t).split(",")]
    )

    def tokens_compatible(a: set, b: set) -> bool:
        # Understat commonly uses a shortened "common name" (drops middle
        # names/patronymics FPL includes), so containment in either
        # direction counts, not exact equality.
        return bool(a) and bool(b) and (a <= b or b <= a)

    matches = []
    for u in us.itertuples():
        team_pool = fpl[fpl["norm_team"].isin(u.norm_teams)]
        name_hits = team_pool[
            team_pool["tokens_full"].apply(lambda t: tokens_compatible(t, u.tokens))
            | team_pool["tokens_web"].apply(lambda t: tokens_compatible(t, u.tokens))
        ]
        if len(name_hits) == 1:
            row = name_hits.iloc[0]
            matches.append(
                {
                    "understat_id": u.id,
                    "understat_name": u.player_name,
                    "fpl_id": row["id"],
                    "fpl_code": row["code"],
                    "match_confidence": "team+name",
                }
            )
        elif len(name_hits) > 1:
            logger.debug(
                "ambiguous match for Understat player %s (%s): %d candidates",
                u.player_name, u.team_title, len(name_hits),
            )

    result = pd.DataFrame(matches)
    logger.info(
        "matched %d/%d Understat players to FPL ids (%d unmatched)",
        len(result), len(us), len(us) - len(result),
    )
    return result


def ingest(fetch_match_logs: bool = True) -> dict[str, pd.DataFrame]:
    session = _session()
    season_players = fetch_season_players(session)
    save_raw(season_players, "understat_season_players")

    from pipeline.storage import load_raw

    fpl_players_raw = load_raw("fpl_hist_players_raw")
    fpl_teams = load_raw("fpl_hist_teams")
    id_map = build_player_id_map(season_players, fpl_players_raw, fpl_teams)
    save_raw(id_map, "understat_fpl_id_map")

    result = {"season_players": season_players, "id_map": id_map}

    if fetch_match_logs:
        matches = fetch_all_player_matches(season_players, session)
        save_raw(matches, "understat_player_matches")
        result["player_matches"] = matches

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
