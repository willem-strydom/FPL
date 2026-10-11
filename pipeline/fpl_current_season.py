"""Ingestion of the *current*, in-progress season's gameweek-by-gameweek
player data, straight from the live FPL API.

Unlike pipeline/fpl_history.py (which pulls a completed season from the
community archive because the live API drops that granularity once a
season ends), the live API's own `element-summary/{id}/` endpoint exposes
real gameweek-by-gameweek history for whichever gameweeks of the *current*
season have actually been played — exactly the signal needed to blend this
season's early evidence into the trailing features (see
pipeline/features.py's `season_weight` handling) instead of waiting for a
full rolling-5 window of real current-season games to exist.

One request per live player, no bulk endpoint — same sequential-with-delay
pattern as pipeline/understat.py's per-player match-log fetch.
"""
import logging
import time

import numpy as np
import pandas as pd
import requests

from pipeline.config import FPL_ELEMENT_SUMMARY_URL, REQUEST_TIMEOUT, USER_AGENT
from pipeline.storage import load_raw, save_raw

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": USER_AGENT}


FETCH_ATTEMPTS = 3


def fetch_player_history(element_id: int) -> pd.DataFrame:
    """This season's gameweek-by-gameweek rows for one player (possibly
    empty if the season hasn't kicked off, or they haven't featured yet).
    """
    url = FPL_ELEMENT_SUMMARY_URL.format(element_id=element_id)
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    history = resp.json().get("history", [])
    return pd.DataFrame(history)


def fetch_all_current_season(
    live_players: pd.DataFrame | None = None, sleep_seconds: float = 0.2
) -> pd.DataFrame:
    live_players = live_players if live_players is not None else load_raw("fpl_live_players")

    frames, failed = [], []
    for i, element_id in enumerate(live_players["id"], start=1):
        for attempt in range(FETCH_ATTEMPTS):
            try:
                h = fetch_player_history(element_id)
                break
            except requests.RequestException as exc:
                if attempt == FETCH_ATTEMPTS - 1:
                    logger.warning("failed to fetch current-season history for element %s: %s", element_id, exc)
                    failed.append(element_id)
                    h = None
                else:
                    time.sleep(2 ** attempt)
        if h is not None and not h.empty:
            frames.append(h)
        if i % 100 == 0:
            logger.info("fetched current-season history for %d/%d players", i, len(live_players))
        time.sleep(sleep_seconds)

    # A partial pull would silently drop players' current-season form from
    # every prediction downstream — refuse rather than save it.
    if failed:
        raise RuntimeError(f"current-season history fetch failed for {len(failed)} players (e.g. {failed[:5]}); not saving")

    if not frames:
        logger.warning("no current-season gameweek history available yet (season hasn't started?)")
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(
        "fetched current-season gameweek history: %d rows, %d players, gameweeks %s",
        len(result), result["element"].nunique(),
        sorted(result["round"].unique().tolist()),
    )
    return result


def ingest() -> pd.DataFrame:
    df = fetch_all_current_season()
    save_raw(df, "fpl_current_gw")
    return df


# Mirrors predict_next_gw.POSITION_NAME / the "GKP"->"GK" rename it does —
# duplicated rather than imported to avoid a circular import (predict_next_gw
# imports from this module for the season-blended live pool).
ELEMENT_TYPE_TO_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_current_season_table() -> pd.DataFrame:
    """Process the raw current-season gameweek log into the same shape
    features.py already expects from the historical archive — just enough
    columns to compute origin-side trailing features from (no target-side
    columns: this table is never used to build training targets, only to
    extend the live inference pool's trailing window with real current-
    season evidence — see pipeline/features.py's `season_weight` handling
    and predict_next_gw.py's use of this table).

    Understat xG/xA are deliberately left NaN here (not re-ingesting
    Understat for the current season yet) — the existing
    understat_form_missing indicator already handles that gracefully
    wherever these rows get pooled.
    """
    gw = load_raw("fpl_current_gw")
    if gw.empty:
        logger.warning("no current-season data to process yet")
        return gw

    live_players = load_raw("fpl_live_players")
    live_fixtures = load_raw("fpl_live_fixtures")

    players = live_players[["id", "code", "element_type"]].rename(columns={"id": "element"})
    players["position"] = players["element_type"].map(ELEMENT_TYPE_TO_POSITION)
    gw = gw.merge(players[["element", "code", "position"]], on="element", how="left")

    fx = live_fixtures[["id", "team_h_difficulty", "team_a_difficulty", "finished_provisional"]].rename(
        columns={"id": "fixture"}
    )
    gw = gw.merge(fx, on="fixture", how="left")
    # Once a gameweek opens, the API lists every fixture in it for every
    # player at 0 minutes, played or not. Mid-gameweek those placeholder rows
    # read as a missed game (start rate, trailing minutes), so keep only
    # fixtures that have actually been played.
    unplayed = ~gw["finished_provisional"].fillna(False).astype(bool)
    if unplayed.any():
        logger.info("dropping %d rows for fixtures not yet played", int(unplayed.sum()))
    gw = gw[~unplayed].drop(columns=["finished_provisional"])
    gw["was_home"] = gw["was_home"].astype(bool)
    gw["own_team_difficulty"] = gw["team_h_difficulty"].where(gw["was_home"], gw["team_a_difficulty"])
    gw = gw.drop(columns=["team_h_difficulty", "team_a_difficulty"])

    # np.nan (float), not pd.NA — pandas' NAType poisons a column's dtype to
    # `object` when concatenated with a float column elsewhere, which breaks
    # .rolling() in _add_trailing_features downstream.
    gw["understat_xG"] = np.nan
    gw["understat_xA"] = np.nan
    gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"])

    numeric_cols = [
        "minutes", "total_points", "goals_scored", "assists", "bps", "ict_index",
        "expected_goals", "expected_assists", "expected_goals_conceded", "defensive_contribution",
        "clearances_blocks_interceptions", "tackles", "recoveries", "starts",
    ]
    for col in numeric_cols:
        gw[col] = pd.to_numeric(gw[col], errors="coerce")

    n_players = gw["element"].nunique()
    logger.info(
        "current-season processed table: %d rows, %d players, gameweeks %s",
        len(gw), n_players, sorted(gw["round"].unique().tolist()),
    )
    return gw


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
