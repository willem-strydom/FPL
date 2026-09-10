"""Ingestion from the official, live FPL API.

This reflects the *current* season (whatever's live right now) — squad
membership, prices, ownership, injury/rotation flags, and DefCon stats.
It's the source of truth for "who is in the player pool right now and are
they fit," which the cold-start filter and the optimizer both need. It is
NOT used for historical per-gameweek training rows: once a new season
starts, the live API drops the previous season's gameweek granularity (see
pipeline/fpl_history.py for that).
"""
import json
import logging

import pandas as pd
import requests

from pipeline.config import FPL_BOOTSTRAP_URL, FPL_FIXTURES_URL, REQUEST_TIMEOUT, USER_AGENT
from pipeline.storage import save_raw

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": USER_AGENT}


def _get_json(url: str) -> dict | list:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap() -> dict[str, pd.DataFrame]:
    """Fetch bootstrap-static and split it into flat per-entity tables."""
    data = _get_json(FPL_BOOTSTRAP_URL)

    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])
    positions = pd.DataFrame(data["element_types"])
    events = pd.DataFrame(data["events"])
    # squad-building rules (squad size, budget, starting-XI size, per-club cap)
    # come straight from the API rather than being hardcoded — the optimizer
    # reads this instead of assuming "£100m, max 3 per club" will always hold.
    game_settings = pd.DataFrame([data["game_settings"]])
    # A few nested fields (per-gameweek rule overrides, chip usage) are
    # irrelevant to model training and parquet can't write empty/variable
    # structs, so flatten them to JSON strings instead of dropping them.
    for col in ("overrides", "chip_plays", "top_element_info"):
        if col in events.columns:
            events[col] = events[col].apply(json.dumps)

    logger.info(
        "fetched bootstrap-static: %d players, %d teams, %d gameweeks",
        len(players), len(teams), len(events),
    )
    return {
        "players": players, "teams": teams, "positions": positions,
        "events": events, "game_settings": game_settings,
    }


def fetch_fixtures() -> pd.DataFrame:
    data = _get_json(FPL_FIXTURES_URL)
    df = pd.DataFrame(data)
    logger.info("fetched %d fixtures", len(df))
    return df


def current_season_label(events: pd.DataFrame) -> str:
    """Infer the live season string (e.g. '2026-27') from gameweek 1's deadline."""
    gw1 = events.loc[events["id"] == 1].iloc[0]
    start_year = pd.to_datetime(gw1["deadline_time"]).year
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def ingest() -> dict[str, pd.DataFrame]:
    bootstrap = fetch_bootstrap()
    fixtures = fetch_fixtures()

    save_raw(bootstrap["players"], "fpl_live_players")
    save_raw(bootstrap["teams"], "fpl_live_teams")
    save_raw(bootstrap["positions"], "fpl_live_positions")
    save_raw(bootstrap["events"], "fpl_live_events")
    save_raw(bootstrap["game_settings"], "fpl_live_game_settings")
    save_raw(fixtures, "fpl_live_fixtures")

    season = current_season_label(bootstrap["events"])
    logger.info("live API is currently serving season %s", season)

    return {**bootstrap, "fixtures": fixtures}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
