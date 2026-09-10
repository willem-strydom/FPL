"""Orchestrates ingestion of every source and assembles the processed,
training-ready player-gameweek table for the 2025-26 season.

Run as a script: `python -m pipeline.build_dataset`
"""
import logging

import pandas as pd

from pipeline import cold_start, fbref, fpl_history, fpl_live, understat
from pipeline.storage import load_processed, load_raw, save_processed

logger = logging.getLogger(__name__)


def ingest_all() -> None:
    logger.info("=== ingesting live FPL API (current squad/prices/status) ===")
    fpl_live.ingest()

    logger.info("=== ingesting 2025-26 historical gameweek data (vaastav mirror) ===")
    fpl_history.ingest()

    logger.info("=== ingesting Understat xG/xA (season + per-match) ===")
    understat.ingest(fetch_match_logs=True)

    logger.info("=== FBref ===")
    fbref.ingest()

    logger.info("=== deriving cold-start candidate-pool exclusions ===")
    cold_start.build()


def add_rolling_minutes(gw: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Last-N-game average minutes, computed strictly from *prior* gameweeks
    (shifted before rolling) so it can't leak the target gameweek's own
    minutes — this is the "will they start" feature from DesignDoc.md.

    Sorted by actual kickoff time, not just `round`: FPL gameweeks can be
    "double gameweeks" (a team plays twice in one round, usually rearranged
    postponements), so two rows can share the same round number. Sorting on
    round alone risks processing those two matches out of chronological
    order.
    """
    gw = gw.sort_values(["element", "kickoff_time"]).copy()
    gw[f"rolling_{window}gw_minutes"] = (
        gw.groupby("element")["minutes"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return gw


def add_fixture_difficulty(gw: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attach the player's own-team FDR for each fixture (Option A from
    DesignDoc.md: fixture difficulty as a direct model input feature).
    """
    fx = fixtures[["id", "team_h_difficulty", "team_a_difficulty"]].rename(
        columns={"id": "fixture"}
    )
    gw = gw.merge(fx, on="fixture", how="left")
    gw["own_team_difficulty"] = gw["team_h_difficulty"].where(
        gw["was_home"], gw["team_a_difficulty"]
    )
    return gw.drop(columns=["team_h_difficulty", "team_a_difficulty"])


def add_understat_xg_xa(gw: pd.DataFrame) -> pd.DataFrame:
    """Join Understat's per-match xG/xA as a cross-check on FPL's own
    (Opta-sourced) expected_goals/expected_assists. Joined on FPL element id
    (via the name/team match) + calendar date, since Understat has no
    concept of FPL gameweek numbers.
    """
    id_map = load_raw("understat_fpl_id_map")
    us_matches = load_raw("understat_player_matches")
    if us_matches.empty:
        logger.warning("no Understat match rows available; skipping xG/xA join")
        gw["understat_xG"] = pd.NA
        gw["understat_xA"] = pd.NA
        return gw

    us = us_matches.merge(
        id_map[["understat_id", "fpl_id"]],
        left_on="understat_player_id",
        right_on="understat_id",
        how="inner",
    )
    us["match_date"] = pd.to_datetime(us["date"]).dt.date
    us = us.rename(columns={"xG": "understat_xG", "xA": "understat_xA"})
    us_small = us[["fpl_id", "match_date", "understat_xG", "understat_xA"]].drop_duplicates(
        subset=["fpl_id", "match_date"]
    )

    gw = gw.copy()
    gw["match_date"] = pd.to_datetime(gw["kickoff_time"]).dt.date
    gw = gw.merge(
        us_small, left_on=["element", "match_date"], right_on=["fpl_id", "match_date"], how="left"
    )
    coverage = gw["understat_xG"].notna().mean()
    logger.info("Understat xG/xA joined onto %.1f%% of gameweek rows", coverage * 100)
    return gw.drop(columns=["fpl_id"])


def add_cold_start_flag(gw: pd.DataFrame) -> pd.DataFrame:
    """Carry the FPL `code` and current cold-start status through, so a
    downstream candidate-pool step can filter without a second lookup.
    Historical rows themselves are training data either way (a player who
    played in 25-26 has real minutes by definition); this flag describes
    whether the player is still eligible as a *forward-looking* candidate.
    """
    players_raw = load_raw("fpl_hist_players_raw")[["id", "code"]].rename(
        columns={"id": "element"}
    )
    exclusions = load_processed("cold_start_exclusions") if _processed_exists("cold_start_exclusions") \
        else None
    gw = gw.merge(players_raw, on="element", how="left")
    if exclusions is not None:
        gw = gw.merge(
            exclusions[["code", "include", "reason"]].rename(
                columns={"include": "candidate_pool_eligible", "reason": "cold_start_reason"}
            ),
            on="code",
            how="left",
        )
    return gw


def _processed_exists(name: str) -> bool:
    from pipeline.config import PROCESSED_DIR

    return (PROCESSED_DIR / f"{name}.parquet").exists()


def build_processed_table() -> pd.DataFrame:
    gw = load_raw("fpl_hist_merged_gw")
    fixtures = load_raw("fpl_hist_fixtures")

    gw = add_rolling_minutes(gw)
    gw = add_fixture_difficulty(gw, fixtures)
    gw = add_understat_xg_xa(gw)
    gw = add_cold_start_flag(gw)

    save_processed(gw, "player_gameweek_2025_26")
    return gw


def run() -> pd.DataFrame:
    ingest_all()
    return build_processed_table()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
