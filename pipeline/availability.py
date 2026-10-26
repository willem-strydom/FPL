"""FPL's own availability flags per gameweek, so the start model can tell a
game missed through a known injury or suspension from a game a player was
left out of.

FPL's API only shows the current status. FPL-Core-Insights
(github.com/olbauday/FPL-Core-Insights, credited as the source) saves a
snapshot of it after every gameweek. A snapshot labelled gameweek g is taken
after that week's games: suspended players play 47% of the time in week g
and 4% in week g+1. So the snapshot for g - 1 is what a manager knew before
gameweek g's deadline.

Two flags per player-gameweek, added to a player log by `add_flags`:
- excused: the player didn't start, and the previous snapshot had him
  injured, suspended, or at most EXCUSED_MAX_CHANCE% to play;
- injury_shortened: he started, played under 60 minutes, and the snapshot
  taken after that week had him injured or doubtful (he went off hurt).
"""
import logging
from io import StringIO

import pandas as pd
import requests

from pipeline.config import RAW_DIR, REQUEST_TIMEOUT, USER_AGENT
from pipeline.storage import load_raw, save_raw

logger = logging.getLogger(__name__)

CORE_INSIGHTS_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
SEASON_FOLDERS = {"2025-26": "2025-2026", "2026-27": "2026-2027"}
EXCUSED_STATUSES = {"i", "s"}
EXCUSED_MAX_CHANCE = 25
INJURED_AFTER_STATUSES = {"i", "d"}


def _raw_name(season: str) -> str:
    return f"availability_{season.replace('-', '_')}"


def fetch(season: str, last_gw: int = 38) -> pd.DataFrame:
    """Snapshots for gameweeks 1..last_gw (missing weeks are skipped)."""
    folder = SEASON_FOLDERS[season]
    frames = []
    for gw in range(1, last_gw + 1):
        resp = requests.get(f"{CORE_INSIGHTS_BASE}/{folder}/By%20Gameweek/GW{gw}/playerstats.csv",
                            headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        snap = pd.read_csv(StringIO(resp.text), usecols=["id", "status", "chance_of_playing_next_round"])
        frames.append(snap.assign(snapshot_gw=gw))
    out = pd.concat(frames, ignore_index=True).rename(columns={"id": "element"})
    save_raw(out, _raw_name(season))
    logger.info("availability %s: %d snapshots", season, out["snapshot_gw"].nunique())
    return out


def load(season: str, refresh: bool = False, last_gw: int = 38) -> pd.DataFrame:
    if refresh or not (RAW_DIR / f"{_raw_name(season)}.parquet").exists():
        return fetch(season, last_gw)
    return load_raw(_raw_name(season))


def add_flags(df: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add `excused` and `injury_shortened` to a per-fixture player log keyed
    by FPL `element` and `round` (one season's ids)."""
    df = df.copy()
    snap = snapshots.copy()
    snap["out_before"] = snap["status"].isin(EXCUSED_STATUSES) | (
        snap["chance_of_playing_next_round"].fillna(100) <= EXCUSED_MAX_CHANCE
    )
    snap["hurt_after"] = snap["status"].isin(INJURED_AFTER_STATUSES)
    before = snap[["element", "snapshot_gw", "out_before"]].assign(round=lambda s: s["snapshot_gw"] + 1)
    after = snap[["element", "snapshot_gw", "hurt_after"]].rename(columns={"snapshot_gw": "round"})
    df = df.merge(before[["element", "round", "out_before"]], on=["element", "round"], how="left")
    df = df.merge(after, on=["element", "round"], how="left")
    df["excused"] = (df["starts"] == 0) & df["out_before"].fillna(False).astype(bool)
    df["injury_shortened"] = (df["starts"] == 1) & (df["minutes"] < 60) & df["hurt_after"].fillna(False).astype(bool)
    logger.info("availability flags: %d excused absences, %d injury-shortened starts",
                int(df["excused"].sum()), int(df["injury_shortened"].sum()))
    return df.drop(columns=["out_before", "hurt_after"])


def avail_now(status: pd.Series, chance: pd.Series) -> pd.Series:
    """Availability at the decision deadline as a 0-1 start-model input:
    injured, suspended, unavailable or not in the squad -> 0, otherwise FPL's
    chance of playing next round (blank means 100)."""
    out = chance.fillna(100).astype(float) / 100
    return out.where(~status.isin(["i", "s", "u", "n"]), 0.0)


def live_eligible(status: pd.Series, chance: pd.Series) -> pd.Series:
    """Same bar as the live scoring pool: available, or doubtful at 75%+."""
    return (status == "a") | ((status == "d") & (chance.fillna(0) >= 75))


def origin_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    """(element, round, avail_now, live_eligible): what was known after
    `round`, i.e. at the deadline for round + 1."""
    return pd.DataFrame({
        "element": snapshots["element"].values,
        "round": snapshots["snapshot_gw"].values,
        "avail_now": avail_now(snapshots["status"], snapshots["chance_of_playing_next_round"]).values,
        "live_eligible": live_eligible(snapshots["status"], snapshots["chance_of_playing_next_round"]).values,
    })
