"""Earlier seasons as extra GLM training data (2022-23 to 2024-25).

The 2025-26 archive (fpl_history / build_dataset) stays the model's core
season. This adds earlier seasons from the same vaastav mirror, shaped like
player_gameweek_2025_26 so features.build_horizon_table runs on them
unchanged:

- `element` is made season-unique (season start year * 100000 + id), so
  rolling and season-to-date features never cross a season boundary, the
  same way live features use the current season only.
- 2024-25 is rescored to 2025-26 rules: DefCon (+2 for 10 CBIT as a DEF, 12
  CBIRT as a MID/FWD) didn't exist then. Per-match clearances, blocks,
  interceptions, tackles and recoveries come from FotMob, as republished by
  FPL-Core-Insights (github.com/olbauday/FPL-Core-Insights, credited as the
  source). Checked against FPL's official 2025-26 DefCon: threshold hit
  agreement 99.2% (DEF) / 99.3% (MID) / 99.9% (FWD), using total tackles.
- 2022-23 and 2023-24 have no per-match defensive stats anywhere we can use
  (FotMob's terms forbid automated access, FBref lost its Opta feed), so
  defensive_contribution is 0 and `defcon_known` is False. Callers use only
  MID/FWD rows from those seasons, where DefCon is worth ~0.1-0.2 points a
  game. FPL's own xG/xA start in 2022-23, so nothing earlier is used.
"""
import logging
from io import StringIO

import numpy as np
import pandas as pd
import requests

from pipeline.config import REQUEST_TIMEOUT, USER_AGENT, VAASTAV_BASE
from pipeline.features import build_horizon_table
from pipeline.storage import PROCESSED_DIR, load_processed, save_processed

logger = logging.getLogger(__name__)

SEASONS = ["2022-23", "2023-24", "2024-25"]
DEFCON_SEASONS = {"2024-25": "2024-2025"}  # season -> FPL-Core-Insights folder
CORE_INSIGHTS_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights/main/data"
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
DEFCON_POINTS = 2
PROCESSED_NAME = "player_gameweek_extra_seasons"
HEADERS = {"User-Agent": USER_AGENT}


def _csv(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def _season_id(season: str) -> int:
    return int(season[:4])


def fetch_season(season: str) -> pd.DataFrame:
    gw = _csv(f"{VAASTAV_BASE}/{season}/gws/merged_gw.csv").drop_duplicates()
    gw = gw[gw["position"].isin(["GK", "GKP", "DEF", "MID", "FWD"])].copy()  # drops 2024-25 assistant managers
    gw["position"] = gw["position"].replace({"GKP": "GK"})
    players = _csv(f"{VAASTAV_BASE}/{season}/players_raw.csv")[["id", "code"]]
    fixtures = _csv(f"{VAASTAV_BASE}/{season}/fixtures.csv")[["id", "team_h_difficulty", "team_a_difficulty"]]
    gw = gw.merge(players.rename(columns={"id": "element"}), on="element", how="left")
    gw = gw.merge(fixtures.rename(columns={"id": "fixture"}), on="fixture", how="left")
    gw["was_home"] = gw["was_home"].astype(bool)
    gw["own_team_difficulty"] = gw["team_h_difficulty"].where(gw["was_home"], gw["team_a_difficulty"])
    gw = gw.drop(columns=["team_h_difficulty", "team_a_difficulty"])
    logger.info("%s: %d rows, %d players", season, len(gw), gw["element"].nunique())
    return gw


def fetch_defensive_stats(folder: str) -> pd.DataFrame:
    """Per-match FotMob defensive stats for one season, keyed by FPL id and match date."""
    frames = []
    for n in range(1, 39):
        stats = _csv(f"{CORE_INSIGHTS_BASE}/{folder}/playermatchstats/GW{n}/playermatchstats.csv")
        matches = _csv(f"{CORE_INSIGHTS_BASE}/{folder}/matches/GW{n}/matches.csv")[["match_id", "kickoff_time"]]
        frames.append(stats.merge(matches, on="match_id", how="left"))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["minutes_played"] > 0].drop_duplicates(["player_id", "match_id"])
    out["match_date"] = pd.to_datetime(out["kickoff_time"], format="mixed", utc=True).dt.date
    return out[["player_id", "match_date", "minutes_played", "tackles", "interceptions", "recoveries", "blocks", "clearances"]]


def add_defcon(gw: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """FPL's 2025-26 definition: DEF count CBIT, MID/FWD count CBIRT, GK 0;
    +2 points at the positional threshold, added to total_points."""
    gw = gw.copy()
    gw["match_date"] = pd.to_datetime(gw["kickoff_time"], format="mixed", utc=True).dt.date
    gw = gw.merge(stats.rename(columns={"player_id": "element"}), on=["element", "match_date"], how="left")
    played = gw["minutes"] > 0
    matched = gw["minutes_played"].notna()
    logger.info("DefCon stats matched for %.1f%% of appearances", 100 * (matched & played).sum() / played.sum())
    for c in ["tackles", "interceptions", "recoveries", "blocks", "clearances"]:
        gw[c] = gw[c].fillna(0)
    gw["clearances_blocks_interceptions"] = gw["clearances"] + gw["blocks"] + gw["interceptions"]
    cbit = gw["clearances_blocks_interceptions"] + gw["tackles"]
    gw["defensive_contribution"] = np.select(
        [gw["position"] == "DEF", gw["position"].isin(["MID", "FWD"])], [cbit, cbit + gw["recoveries"]], 0
    )
    hit = gw["defensive_contribution"] >= gw["position"].map(DEFCON_THRESHOLD).fillna(np.inf)
    gw["defcon_points_added"] = DEFCON_POINTS * hit.astype(int)
    gw["total_points"] = gw["total_points"] + gw["defcon_points_added"]
    gw["defcon_known"] = True
    return gw.drop(columns=["minutes_played", "blocks", "clearances", "interceptions"])


def build() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        gw = fetch_season(season)
        if season in DEFCON_SEASONS:
            gw = add_defcon(gw, fetch_defensive_stats(DEFCON_SEASONS[season]))
        else:
            gw["match_date"] = pd.to_datetime(gw["kickoff_time"], format="mixed", utc=True).dt.date
            gw[["clearances_blocks_interceptions", "tackles", "recoveries", "defensive_contribution"]] = 0
            gw["defcon_points_added"] = 0
            gw["defcon_known"] = False
        gw["season"] = season
        gw["season_element"] = gw["element"]
        gw["element"] = _season_id(season) * 100000 + gw["element"]
        gw["understat_xG"] = np.nan  # Understat isn't ingested for these seasons
        gw["understat_xA"] = np.nan
        gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"], format="mixed", utc=True)
        frames.append(gw)
    out = pd.concat(frames, ignore_index=True)
    save_processed(out, PROCESSED_NAME)
    logger.info("saved %s: %d rows across %s", PROCESSED_NAME, len(out), SEASONS)
    return out


def load() -> pd.DataFrame:
    if not (PROCESSED_DIR / f"{PROCESSED_NAME}.parquet").exists():
        logger.info("%s not built yet, building it now", PROCESSED_NAME)
        return build()
    return load_processed(PROCESSED_NAME)


# Extra GLM training rows: MID/FWD priced >= £7.0m at the origin gameweek in
# 2022-23 and 2023-24 (60+ minute targets, like the 25-26 training table),
# each season with its own intercept and price slope, so these rows inform
# how form, xG and fixtures move premium predictions without moving the
# premium level, which stays learned from 2025-26 (the live baseline: all
# season columns are 0 for 25-26 and live rows). Purged walk-forward on
# 25-26: top-3 per position per week +0.10 ± 0.04 realized points per pick,
# RMSE and bias unchanged, cheap tiers unchanged, £10m+ residual -0.42 ->
# -0.28. Season-ahead (no 25-26 in training): top-3 +0.23 ± 0.07, £7.5m+ MAE
# -0.05 ± 0.02. The full 2024-25 season was tested too and rejected: it
# over-predicted 25-26 premiums badly (Haaland 9.2 vs 5.9 actual) and
# everyone by ~0.3, even with its own intercept and price slope.
PREMIUM_SEASONS = ["2022-23", "2023-24"]
PREMIUM_MIN_VALUE = 70
PREMIUM_POSITIONS = ["MID", "FWD"]


def season_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Per-season intercept and price slope for the premium seasons; all 0
    for any row outside them (2025-26 training rows, live scoring rows)."""
    season = table["season"] if "season" in table else pd.Series("", index=table.index)
    cols = {}
    for s in PREMIUM_SEASONS:
        on = (season == s).astype(float)
        cols[f"season_{s[:4]}"] = on
        cols[f"season_{s[:4]}_x_value"] = on * (table["value"] - 55)
    return pd.DataFrame(cols, index=table.index)


def premium_training_table() -> pd.DataFrame:
    extra = load()
    frames = []
    for s in PREMIUM_SEASONS:
        h = build_horizon_table(extra[extra["season"] == s]).assign(season=s)
        frames.append(h[h["position"].isin(PREMIUM_POSITIONS) & (h["value"] >= PREMIUM_MIN_VALUE)])
    out = pd.concat(frames, ignore_index=True)
    logger.info("premium training rows from %s: %d rows, %d players", PREMIUM_SEASONS, len(out), out["code"].nunique())
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
