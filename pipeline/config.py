"""Shared constants and paths for the FPL data ingestion pipeline."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# The season we train/validate on, per DesignDoc.md ("only use data from the
# past season"). The live FPL API always reflects the *current* season, and
# once a new season starts it stops exposing the previous season's
# gameweek-by-gameweek granularity (only season totals remain via
# history_past). So historical per-gameweek data comes from the
# community-maintained archive instead.
HISTORY_SEASON = "2025-26"          # vaastav/Fantasy-Premier-League folder naming
UNDERSTAT_SEASON = 2025             # Understat labels a season by its start year
UNDERSTAT_LEAGUE = "EPL"

FPL_API_BASE = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_API_BASE}/bootstrap-static/"
FPL_FIXTURES_URL = f"{FPL_API_BASE}/fixtures/"
FPL_ELEMENT_SUMMARY_URL = f"{FPL_API_BASE}/element-summary/{{element_id}}/"

# --- Current-season (in-progress) gameweek-by-gameweek blending ---
# Once the new season kicks off, the live API's own element-summary.history
# carries real gameweek-by-gameweek data for it (see pipeline/fpl_current_season.py)
# — the trailing features can blend this in immediately rather than waiting
# for a full rolling-5 window of real current-season games to accumulate.
# Per-game weights inside that blended 5-game window (applied only at live-
# inference time, never to historical training data). Explicit request as
# of GW4 (3 real current-season games now exist, filling most of the
# window): 1/2 per historical game, 4/3 per current-season game — for a
# player with the now-typical 2 historical + 3 current-season split, that's
# 2*(1/2) + 3*(4/3) = 1 + 4 = 5, i.e. still a 5-game-equivalent window in
# total weight, just reallocated toward this season's real evidence rather
# than uniformly. Previously a single CURRENT_SEASON_GAME_WEIGHT=1.5 with
# historical implicitly at 1.0 (requested "roughly 3:2") — superseded here,
# not additive with it.
HISTORICAL_SEASON_GAME_WEIGHT = 1 / 2
CURRENT_SEASON_GAME_WEIGHT = 4 / 3

VAASTAV_BASE = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)

UNDERSTAT_BASE = "https://understat.com"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

REQUEST_TIMEOUT = 30

# --- Cold-start player filter thresholds (see DesignDoc.md "Scope cut") ---
# A player must clear both bars on the *historical* (25-26) season to be
# treated as having a real top-flight track record. Anyone missing from the
# 25-26 archive entirely (new-to-the-league transfer, academy graduate,
# promoted-team player) is excluded outright.
COLD_START_MIN_STARTS = 5
COLD_START_MIN_MINUTES = 450  # ~5 full matches worth

# --- DefCon (defensive contribution) bonus-point thresholds, 2025-26 rules ---
# FPL awards 2 points for clearing this many defensive_contribution actions
# (clearances+blocks+interceptions+tackles for defenders; +recoveries for
# mid/forward) in a single match. Confirmed empirically against the 25-26
# archive rather than taken on faith: at these thresholds, hit rates for
# players with 60+ minutes come out to ~27% (DEF), ~18% (MID), ~1% (FWD) —
# consistent with defenders being built for this stat and forwards almost
# never clearing it. Goalkeepers have no DefCon component at all
# (defensive_contribution is always 0 for GK in the data) and are
# deliberately absent from this dict — a threshold lookup miss is the
# correct "never hits" behavior for them, not a bug to special-case.
DEFCON_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}

# Explicit overrides for club names that don't normalize to the same string
# across FPL and Understat (short names, abbreviations, ampersands, etc.).
# Keys are normalize_team_name() output.
TEAM_NAME_OVERRIDES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "man united": "manchester united",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "nott m forest": "nottingham forest",
    "nottm forest": "nottingham forest",
    "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united",
    "leeds united": "leeds",
    "west brom": "west bromwich albion",
    "sheffield utd": "sheffield united",
    # RotoWire-specific full club names
    "afc bournemouth": "bournemouth",
    "brighton hove albion": "brighton",
    "brighton and hove albion": "brighton",
}
