"""Ingestion of last season's (2025-26) per-gameweek player data.

The official live FPL API only exposes granular gameweek-by-gameweek
history for the *current* season; once a new season starts, a player's
`element-summary.history` for the prior season goes empty and only season
totals survive (`history_past`). Since DesignDoc.md requires training on
last season specifically, we pull it from the vaastav/Fantasy-Premier-League
GitHub mirror, which snapshots each season's gameweek-level data (a
long-standing, widely used community archive of the same official-API
responses, not a third-party scrape of a different dataset).
"""
import logging

import pandas as pd
import requests

from pipeline.config import HISTORY_SEASON, REQUEST_TIMEOUT, USER_AGENT, VAASTAV_BASE
from pipeline.storage import save_raw

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": USER_AGENT}


def _get_csv(relative_path: str) -> pd.DataFrame:
    url = f"{VAASTAV_BASE}/{HISTORY_SEASON}/{relative_path}"
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(resp.text))


def fetch_merged_gw() -> pd.DataFrame:
    df = _get_csv("gws/merged_gw.csv")
    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        logger.warning(
            "dropped %d exact-duplicate rows from merged_gw (upstream source artifact)",
            n_before - len(df),
        )
    logger.info(
        "fetched %s merged_gw: %d rows, %d players, gameweeks %d-%d",
        HISTORY_SEASON, len(df), df["element"].nunique(),
        df["round"].min(), df["round"].max(),
    )
    return df


def fetch_players_raw() -> pd.DataFrame:
    df = _get_csv("players_raw.csv")
    logger.info("fetched %s players_raw: %d players", HISTORY_SEASON, len(df))
    return df


def fetch_teams() -> pd.DataFrame:
    df = _get_csv("teams.csv")
    logger.info("fetched %s teams: %d teams", HISTORY_SEASON, len(df))
    return df


def fetch_fixtures() -> pd.DataFrame:
    df = _get_csv("fixtures.csv")
    logger.info("fetched %s fixtures: %d fixtures", HISTORY_SEASON, len(df))
    return df


def ingest() -> dict[str, pd.DataFrame]:
    merged_gw = fetch_merged_gw()
    players_raw = fetch_players_raw()
    teams = fetch_teams()
    fixtures = fetch_fixtures()

    save_raw(merged_gw, "fpl_hist_merged_gw")
    save_raw(players_raw, "fpl_hist_players_raw")
    save_raw(teams, "fpl_hist_teams")
    save_raw(fixtures, "fpl_hist_fixtures")

    return {
        "merged_gw": merged_gw,
        "players_raw": players_raw,
        "teams": teams,
        "fixtures": fixtures,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
