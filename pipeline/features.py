"""Horizon-as-feature training table for the "quality model" (form model)
described in DesignDoc.md.

Two separate ideas get computed here, matching the doc's own split:

  - "will they start" signal: trailing 5-gameweek average minutes,
    INCLUDING zero-minute (benched/injured) gameweeks — this is a playing
    time trend, not a quality signal.
  - "form"/quality signal: per-90 rates (points, goals, assists, bps, ict,
    xG, xA) rolled over the player's last 5 *appearances* only (minutes>0),
    so a stretch on the bench doesn't dilute the quality read — this
    mirrors the doc's instruction that the quality model itself is "trained
    only on gameweeks where the player actually played."

Rows are (player, origin_gameweek, horizon) triples, horizon in {1..4} by
default. All input features are computed strictly from data up to and
including the origin gameweek; the target is total_points at
origin+horizon, restricted to target gameweeks where the player played
60+ minutes (the dominant band in DesignDoc.md's point-decomposition
formula, and the piece explicitly slated for ML treatment).

Double gameweeks (a round with two fixtures) are summed for the target
(one gameweek-total prediction) and the origin snapshot is taken after the
round's last fixture, so trailing features already reflect both matches.
"""
import logging

import numpy as np
import pandas as pd

from pipeline.config import DEFCON_THRESHOLD

logger = logging.getLogger(__name__)

MIN_ORIGIN_ROUND = 4  # need a few prior fixtures before trailing stats mean anything
TARGET_MIN_MINUTES = 60  # the "plays 60+ min" band from the point-decomposition formula
DEFAULT_HORIZONS = (1, 2, 3, 4)
MIN_RELIABLE_TRAILING_MINUTES = 30  # below this pooled minutes, a per-90 rate is noise

_PER90_SOURCE_COLS = {
    "total_points": "form_points_per90",
    "goals_scored": "form_goals_per90",
    "assists": "form_assists_per90",
    "bps": "form_bps_per90",
    "ict_index": "form_ict_per90",
    "expected_goals": "form_xG_per90",
    "expected_assists": "form_xA_per90",
    # Conceded-side signal, strongest for defenders: expected goals against
    # while this player was on the pitch, and their raw DefCon (defensive
    # contribution) rate — same minutes-weighted pooling treatment as every
    # attacking per-90 rate above. form_defcon_per90 is deliberately excluded
    # from FEATURE_COLUMNS below (still computed here for diagnostics) — the
    # GLM uses DEFCON_HIT_RATE_COL instead, see its docstring for why.
    "expected_goals_conceded": "form_xGA_per90",
    "defensive_contribution": "form_defcon_per90",
    "understat_xG": "form_understat_xG_per90",
    "understat_xA": "form_understat_xA_per90",
}

# Not a per-90 rate — a rolling, position-thresholded hit rate: what fraction
# of this player's last-5 appearances cleared their position's DefCon bonus
# threshold (see pipeline.config.DEFCON_THRESHOLD, confirmed against the
# 25-26 archive). Computed alongside the per-90 pool below (same "played"
# subset, same reliability floor, same season-weighting) since it's the same
# kind of appearance-pooled quality signal, just a rate-of-hitting-a-
# threshold instead of a rate-per-90. Used in FEATURE_COLUMNS INSTEAD of
# form_defcon_per90: the DefCon bonus is a hard scoring threshold (+2 points
# for clearing it, nothing for falling short), so expected-bonus-points is
# linear in P(hit) — which is exactly what this rolling hit rate estimates —
# not linear in the raw action count a linear model would otherwise be
# forced to treat as if it were. A same-vs-same walk-forward swap test
# (per90 alone vs hit_rate alone) came back statistically tied overall, but
# this is the theoretically correct one to prefer given the tie, and it's
# the one carried into the position-architecture comparison below.
DEFCON_HIT_RATE_COL = "form_defcon_hit_rate"

# Excluded from FEATURE_COLUMNS despite being computed above (VIF-diagnosed
# multicollinearity — see conversation): form_goals_per90, form_bps_per90,
# and form_ict_per90 are mechanically near-duplicates of form_points_per90
# (FPL points are computed FROM goals/assists/bonus in the first place), and
# form_understat_xG_per90/form_understat_xA_per90 duplicate the FPL-native
# xG/xA the model already has. Tried dropping just the understat pair first
# — didn't fix anything, form_goals_per90 stayed significantly *negative*.
# Dropping the whole attacking-output cluster (this list) did: every
# coefficient in the resulting fit gets the sign you'd expect (xG, xA both
# flip to positive and significant; assists goes from a nonsensical
# significant negative to a non-significant near-zero), walk-forward
# accuracy is statistically unchanged (best or tied-best at every position
# except a negligible loss at DEF/MID, all in the 4th decimal of MAE), and
# it has the best BIC of every candidate tested. form_understat_xG_per90 is
# still computed and still drives the understat_form_missing flag — dropping
# it from FEATURE_COLUMNS doesn't remove that indicator's own value.
_PRUNED_FOR_COLLINEARITY = {
    "form_defcon_per90", "form_goals_per90", "form_bps_per90", "form_ict_per90",
    "form_understat_xG_per90", "form_understat_xA_per90",
}

FEATURE_COLUMNS = [
    "horizon",
    "trail_minutes5",
    "trail_games5",
    "value",
    "target_difficulty",
    "target_was_home",
] + [c for c in _PER90_SOURCE_COLS.values() if c not in _PRUNED_FOR_COLLINEARITY] + [DEFCON_HIT_RATE_COL]

POSITION_COLUMN = "position"


def _add_trailing_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["element", "kickoff_time"]).reset_index(drop=True)

    # Per-game weight inside the trailing window. Defaults to uniform (1.0)
    # — reproducing the old unweighted behavior exactly — for a plain
    # single-season table, which is what model TRAINING always uses.
    # predict_next_gw.py's live inference pool is the one place that passes
    # a real season_weight column (see pipeline.config.CURRENT_SEASON_GAME_WEIGHT):
    # concatenating last season's tail with this season's games-so-far and
    # weighting the current-season ones higher lets a handful of real
    # current-season games actually move the trailing read immediately,
    # rather than being diluted by however many stale historical games still
    # fill out the rest of a fixed 5-game window. Never applied to training.
    if "season_weight" not in df.columns:
        df["season_weight"] = 1.0

    df["_w_minutes"] = df["minutes"] * df["season_weight"]
    df["_w_points"] = df["total_points"] * df["season_weight"]
    df["_w_starts"] = df["starts"] * df["season_weight"]

    grp = df.groupby("element")
    roll_w = grp["season_weight"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    df["trail_minutes5"] = grp["_w_minutes"].transform(lambda s: s.rolling(window, min_periods=1).sum()) / roll_w
    df["trail_games5"] = grp["minutes"].transform(
        lambda s: s.rolling(window, min_periods=1).apply(lambda x: (x > 0).sum(), raw=True)
    )
    # Rolling (season-weighted) average of the actual target variable
    # (points), zero-weeks included — the "naive forecast" baseline: what
    # would you predict if you just went with what they've been averaging
    # lately, no model at all. Inclusive of the current row (origin already
    # happened, so there's no leakage in using its own points).
    df["trail_points5"] = grp["_w_points"].transform(lambda s: s.rolling(window, min_periods=1).sum()) / roll_w
    # Season-weighted fraction of the last 5 games actually started (not
    # just "appeared in") — not a model feature, used only by
    # predict_next_gw.compute_p_plays_60 as the fallback/future-week start-
    # probability estimate, in place of trail_minutes5/90. Distinguishes a
    # reliable 90-minute player from one who starts but is frequently
    # subbed early (both can show similar trail_minutes5, but their `starts`
    # record differs once you'd expect it to).
    df["trail_starts5"] = grp["_w_starts"].transform(lambda s: s.rolling(window, min_periods=1).sum()) / roll_w
    df = df.drop(columns=["_w_minutes", "_w_points", "_w_starts"])

    # Expanding (not rolling, not season-weighted) cumulative-to-date
    # counters: same leakage-safe convention as the rolling features above
    # (sorted by kickoff_time, computed strictly from each player's own
    # past + current row), but an unbounded window instead of a trailing-5
    # one. This is a fold-safe stand-in for cold_start.py's full-season
    # "established squad player" bar — that filter uses the *complete*
    # historical season, which is fine for gating live prediction candidates
    # but would leak future rounds if reused inside a walk-forward fold.
    # These columns answer the same question ("has this player actually
    # proven a squad role, as of right now") using only information
    # available at each row's own origin. Left unweighted deliberately —
    # tenure is a count of proof, not a recency-weighted quality read.
    df["cum_starts_to_date"] = grp["minutes"].transform(lambda s: (s >= 60).expanding().sum())
    df["cum_minutes_to_date"] = grp["minutes"].transform(lambda s: s.expanding().sum())

    # Minutes-weighted (pooled), not a mean-of-per-match-rates: summing the raw
    # stat and the minutes separately over the window before dividing avoids a
    # short cameo appearance dominating the rate. A single goal in a 2-minute
    # substitute cameo is a 45-points-per-90 extrapolation if treated as its
    # own data point and averaged in — pooling across the full window's
    # minutes dilutes it the way a real per-90 rate should. season_weight is
    # folded in as a second multiplicative weight alongside minutes, so the
    # same current-season-counts-for-more treatment applies here too.
    played = df[df["minutes"] > 0].copy()
    played = played.sort_values(["element", "kickoff_time"])
    src_cols = list(_PER90_SOURCE_COLS.keys())

    played["_w_minutes"] = played["minutes"] * played["season_weight"]
    for src in src_cols:
        played[f"_w_{src}"] = played[src] * played["season_weight"]
    played["_defcon_hit"] = (
        played["defensive_contribution"] >= played[POSITION_COLUMN].map(DEFCON_THRESHOLD)
    ).astype(float) * played["season_weight"]

    grp = played.groupby("element")
    rolled_w_stats = grp[[f"_w_{src}" for src in src_cols]].transform(
        lambda s: s.rolling(window, min_periods=1).sum()
    )
    rolled_w_minutes = grp["_w_minutes"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    rolled_w_defcon_hit = grp["_defcon_hit"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    rolled_weight = grp["season_weight"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    # Plain (unweighted) pooled minutes too — the reliability floor below is
    # about how much *real match time* backs a rate, not how much weighted
    # credit it earned, so a couple of heavily-upweighted current-season
    # cameos still can't fake their way past MIN_RELIABLE_TRAILING_MINUTES.
    rolled_minutes_real = grp["minutes"].transform(lambda s: s.rolling(window, min_periods=1).sum())

    rolled = pd.DataFrame(
        {name: rolled_w_stats[f"_w_{src}"] / rolled_w_minutes * 90 for src, name in _PER90_SOURCE_COLS.items()},
        index=played.index,
    )
    rolled[DEFCON_HIT_RATE_COL] = rolled_w_defcon_hit / rolled_weight
    # Even pooled, a rate built from well under a match's worth of trailing
    # minutes (e.g. two separate 1-minute cameos) is still noise dominated by
    # whatever fluke did or didn't happen in that handful of minutes. Below
    # the floor, don't trust this window's rate — fall through to ffill and
    # carry the last reliable estimate instead.
    unreliable = rolled_minutes_real < MIN_RELIABLE_TRAILING_MINUTES
    rolled.loc[unreliable] = np.nan
    form_snapshot = pd.concat([played[["element", "round", "kickoff_time"]], rolled], axis=1)

    df = df.merge(form_snapshot, on=["element", "round", "kickoff_time"], how="left")
    form_cols = list(_PER90_SOURCE_COLS.values()) + [DEFCON_HIT_RATE_COL]
    df = df.sort_values(["element", "kickoff_time"])
    df[form_cols] = df.groupby("element")[form_cols].ffill()
    return df


def _origin_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (element, round): state after that round's last fixture."""
    return (
        df.sort_values("kickoff_time")
        .groupby(["element", "round"], as_index=False)
        .last()
    )


def _target_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (element, round): gameweek-total target, DGW-summed."""
    return df.groupby(["element", "round"], as_index=False).agg(
        target_points=("total_points", "sum"),
        target_minutes=("minutes", "sum"),
        target_difficulty=("own_team_difficulty", "mean"),
        target_was_home=("was_home", "first"),
    )


def build_horizon_table(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_origin_round: int = MIN_ORIGIN_ROUND,
    target_min_minutes: int = TARGET_MIN_MINUTES,
    target_max_minutes: int | None = None,
    rolling_window: int = 5,
) -> pd.DataFrame:
    df = _add_trailing_features(df, window=rolling_window)

    origin = _origin_snapshot(df)
    origin = origin[origin["round"] >= min_origin_round]
    origin = origin[origin["trail_games5"] > 0]  # drop players with no appearances yet

    targets = _target_table(df)

    frames = []
    for h in horizons:
        block = origin.copy()
        block["horizon"] = h
        block["target_round"] = block["round"] + h
        block = block.merge(
            targets, left_on=["element", "target_round"], right_on=["element", "round"],
            how="inner", suffixes=("", "_target"),
        )
        block = block.rename(columns={"round": "origin_round"})
        frames.append(block)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["target_minutes"] >= target_min_minutes]
    if target_max_minutes is not None:
        combined = combined[combined["target_minutes"] <= target_max_minutes]
    combined = combined.copy()

    # Two independent reasons a form column can be NaN here, flagged before
    # either gets filled in (order matters — the FPL-native columns go NaN
    # together under the trailing-minutes floor; the Understat columns go NaN
    # together under match/no-match coverage; filling one pass would erase
    # the signal the other pass still needs to read):
    #
    # 1. A player can clear the "at least one appearance" bar above yet still
    #    fall under MIN_RELIABLE_TRAILING_MINUTES (e.g. two separate 1-minute
    #    cameos), leaving the core form columns NaN.
    fpl_native_form_cols = [c for c in _PER90_SOURCE_COLS.values() if "understat" not in c] + [DEFCON_HIT_RATE_COL]
    combined["form_unreliable"] = combined["form_points_per90"].isna().astype(int)
    combined[fpl_native_form_cols] = combined[fpl_native_form_cols].fillna(0.0)

    # 2. The player never matched to an Understat record, or matched but has
    #    no trailing match-log coverage yet. xG and xA missingness always
    #    coincide (same match/no-match), so one indicator covers both instead
    #    of adding two perfectly collinear columns.
    combined["understat_form_missing"] = combined["form_understat_xG_per90"].isna().astype(int)
    for col in ("form_understat_xG_per90", "form_understat_xA_per90"):
        combined[col] = combined[col].fillna(0.0)

    combined["group_key"] = (
        combined["element"].astype(str) + "_" + combined["origin_round"].astype(str)
    )

    n_groups = combined["group_key"].nunique()
    logger.info(
        "built horizon table: %d rows, %d (player, origin_gw) groups, horizons=%s",
        len(combined), n_groups, horizons,
    )
    return combined


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot the position column, add the clean-sheet interaction terms,
    and return (X, feature_names). Same encoding used for both the GLM and
    the GBM for a fair comparison.
    """
    cols = FEATURE_COLUMNS + ["understat_form_missing", "form_unreliable"]
    X = df[cols].copy()
    X["target_was_home"] = X["target_was_home"].astype(int)
    pos_dummies = pd.get_dummies(df[POSITION_COLUMN], prefix="pos", dtype=int)
    X = pd.concat([X, pos_dummies], axis=1)

    # Clean-sheet-relevant slopes allowed to differ for DEF/GK vs MID/FWD,
    # everything else shared: walk-forward validated against a plain pooled
    # model (this, minus these two columns) and a full 2-way position-split
    # architecture — this won outright (best overall, best for DEF and GK,
    # ~tied for MID/FWD), the split lost even to the plain pooled baseline.
    # Main effects for target_difficulty/form_xGA_per90 stay in above so the
    # interaction captures the *difference* in slope, not the whole effect.
    is_defensive = df[POSITION_COLUMN].isin(["DEF", "GK"]).astype(int)
    X["xGA_x_defensive"] = X["form_xGA_per90"] * is_defensive
    X["difficulty_x_defensive"] = X["target_difficulty"] * is_defensive

    return X, list(X.columns)
