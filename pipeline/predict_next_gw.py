"""Deploy the NB2 GLM (the model favored for the template/owned player tier —
see the ownership-filtered robustness check) to score every eligible player
for the coming season's first gameweek.

"Eligible" means: passes the DesignDoc.md cold-start filter (real 25-26 PL
minutes, not a promoted-team player) and is currently fit (status == 'a', or
a doubtful/'d' player whose chance_of_playing_next_round is still >=75).

Trailing form features come from each player's last available 2025-26
gameweek (there's nothing more recent — the new season hasn't kicked off).
The target fixture is each team's actual GW1 2026-27 match: real fixture
difficulty and home/away, horizon fixed at 1. This is a genuine forecast,
not a backtest — there's no ground truth to score it against yet.
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from sklearn.ensemble import HistGradientBoostingClassifier

from pipeline.features import (
    MAX_HORIZON,
    MIN_ORIGIN_ROUND,
    POSITION_COLUMN,
    START_MODEL_COLS,
    START_RATE_COL,
    START_RATE_NEW_GAME_WEIGHT,
    _add_trailing_features,
    _origin_snapshot,
    build_horizon_table,
    fill_missing_form,
)
from pipeline.lineups import BENCH_PROB, get_start_probabilities
from pipeline.matching import normalize_team
from pipeline.model_compare import GLM_DROP_COL, GLM_OFFSET
from pipeline.storage import load_processed, load_raw

logger = logging.getLogger(__name__)

POSITION_NAME = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FLOOR_POINTS = 0.5  # old naive constant — kept only as a reference value in logs, no longer used in the formula


def get_next_gameweek() -> int:
    """The next literal FPL gameweek number to predict for — NOT hardcoded
    to 1. Bug fixed here: earlier callers passed `horizon` (1-4, "weeks from
    now") directly as the literal gameweek to fetch fixtures for, which only
    happened to be right in week 1 of the season and would silently target
    already-finished gameweeks (and feed the GLM a wrong `horizon` value —
    see build_gw_feature_rows) every week after. Read from live data instead
    of computed once and cached: this changes every gameweek.
    """
    events = load_raw("fpl_live_events")
    next_events = events.loc[events["is_next"], "id"]
    if not next_events.empty:
        return int(next_events.iloc[0])
    # Fallback if is_next is ever stale/unset: first not-yet-finished gameweek.
    unfinished = events.loc[~events["finished"], "id"]
    if unfinished.empty:
        raise RuntimeError("no upcoming gameweek found in fpl_live_events — season may be over")
    return int(unfinished.min())


_PREMIUM_CACHE: tuple | None = None


def with_premium_seasons(table: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GLM training set: these 25-26 rows plus the 2022-24 premium MID/FWD
    rows, with the premium seasons' own intercept and price-slope columns (0
    for 25-26 rows). Returns (frame with target_points, X)."""
    global _PREMIUM_CACHE
    from pipeline.extra_seasons import premium_training_table, season_columns
    from pipeline.features import feature_matrix

    if _PREMIUM_CACHE is None:
        premium = premium_training_table()
        X_premium, _ = feature_matrix(premium)
        _PREMIUM_CACHE = (premium[["target_points"]], pd.concat([X_premium, season_columns(premium)], axis=1))
    y_premium, X_premium = _PREMIUM_CACHE
    X_train = pd.concat([pd.concat([X, season_columns(table)], axis=1), X_premium], ignore_index=True).fillna(0.0)
    y_train = pd.concat([table[["target_points"]], y_premium], ignore_index=True)
    return y_train, X_train


def fit_production_glm(horizon_df: pd.DataFrame, X: pd.DataFrame):
    """NB2 GLM trained on the full 25-26 horizon table — no held-out fold,
    since this is used to forecast forward into unseen fixtures, not to
    validate against known outcomes."""
    Xc = sm.add_constant(X.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    y = horizon_df["target_points"].values + GLM_OFFSET
    res = dm.NegativeBinomial(y, Xc, loglike_method="nb2").fit(disp=0, maxiter=200)
    return res, Xc.columns


def fit_floor_mae_optimal(full_df: pd.DataFrame, X_full: pd.DataFrame, glm60_res) -> float:
    """MAE-optimal replacement for the old hardcoded FLOOR_POINTS=0.5,
    validated in two_stage_pipeline.py's walk-forward comparison (MAE 1.919
    vs 1.981 for the hardcoded floor, holding up across every fold,
    position, ownership tier, and season-third checked, including the
    deployment-realistic "plausible candidates only" validation slice).

    Fit the same way as there: a single constant solving
    min_floor sum |actual - p60*q60 - (1-p60)*floor| via a weighted median
    over ALL rows (not just 60+ ones — full_df must be the unfiltered
    target population), weighted by (1-p60) itself.

    Deliberately fit on the FULL, unfiltered population — restricting this
    specific fit to "established"/plausible-candidate rows was tried and
    made things worse (see two_stage_pipeline.py and the conversation):
    established players mostly carry high p60, hence small (1-p60) weight,
    so restricting to them doesn't just shrink the row count, it shrinks
    the total weight mass backing the fit by ~8x, leaving an unstable
    estimate dominated by a handful of noisy rotation-risk cases. That
    finding is specific to this single-constant fit, though — it does NOT
    generalize to the GLM's own training data (see fit_production_glm /
    the GW-scoring pool above), which already separates populations via
    its own per-row features and only got noisier when its training rows
    were restricted the same way.
    """
    from pipeline.two_stage_pipeline import _predict_with_fitted_glm, weighted_median

    p60_train = (full_df["trail_minutes5"] / 90).clip(0, 1).values
    glm60_pred_train = _predict_with_fitted_glm(glm60_res, X_full, GLM_OFFSET)
    weight_train = 1 - p60_train
    resid_train = full_df["target_points"].values - p60_train * glm60_pred_train
    valid = weight_train > 1e-6
    if not valid.any():
        logger.warning("floor fit: no valid rows, falling back to hardcoded FLOOR_POINTS=%.1f", FLOOR_POINTS)
        return FLOOR_POINTS
    floor = weighted_median(resid_train[valid] / weight_train[valid], weight_train[valid])
    logger.info("production MAE-optimal floor: %.4f (old hardcoded reference: %.1f)", floor, FLOOR_POINTS)
    return floor


def _load_blended_player_log() -> pd.DataFrame:
    """This season's games-so-far, one per-player timeline for
    _add_trailing_features (see pipeline/features.py) to compute trailing
    features from. Never used for GLM training (see _fit_production_models,
    which stays on the plain unweighted 25-26 archive) — only for scoring
    the live candidate pool.

    As of GW5 (4 real current-season games exist — enough to stop leaning
    on last season's tail at all, per request): last season's archive is
    excluded entirely rather than blended in at a reduced weight. Earlier
    in the season this function did carry a HISTORICAL_SEASON_GAME_WEIGHT/
    CURRENT_SEASON_GAME_WEIGHT blend (see pipeline/config.py, still defined
    there in case an earlier point in a future season wants it again) — cut
    now on the reasoning that once real current-season history fills the
    5-game window, the old crutch adds nothing but noise. season_weight is
    still set (uniformly) rather than dropped as a column: every downstream
    weighted-average feature (trail_minutes5, trail_points5, trail_starts5,
    the per-90 form rates) is scale-invariant to a uniform weight — verified
    directly, not assumed — so the exact value here is inert as long as
    it's the same for every row; only the historical/current *split* was
    ever doing anything, and that split no longer exists.

    `element` (FPL's numeric player id) still gets aliased to `code` even
    though only one season is in play now — harmless, and keeps this
    function's output shape consistent with what build_gw_feature_rows
    already expects.
    """
    from pipeline.fpl_current_season import build_current_season_table

    current = build_current_season_table()
    if current.empty:
        raise RuntimeError(
            "no current-season games played yet, and last season's archive is no longer "
            "used as a fallback — re-enable the historical blend (see git history) if "
            "scoring before real current-season data exists is ever needed again."
        )

    current = current.copy()
    current["season_weight"] = 1.0
    current["element"] = current["code"]

    logger.info("live inference trailing features: %d current-season rows, no historical blend", len(current))
    return current


def graduated_new_arrivals(cold_start: pd.DataFrame) -> pd.Series:
    """Cold-start-excluded players (no/limited 25-26 PL record) who've since
    played enough real Premier League minutes THIS season to trust a
    trailing read from — the general replacement for manual_additions.py's
    hand-picked list. Once a player clears this bar, nothing special
    happens for them: they enter build_gw_feature_rows' normal candidate
    pool and flow through _load_blended_player_log/_add_trailing_features
    exactly like anyone else. Since they have zero historical (25-26) rows,
    their trailing window is built entirely from real current-season games
    (season_weight=CURRENT_SEASON_GAME_WEIGHT each, same pooling, same
    reliability floor as everyone) — no separate discount formula, no
    fabricated/imputed games. Same MIN_RELIABLE_TRAILING_MINUTES bar
    _add_trailing_features already uses to decide a per-90 rate is trustworthy
    (a couple of real 90-minute appearances clears it easily; a single
    substitute cameo does not) — reused here rather than inventing a second
    threshold, and it means anyone included is guaranteed to already be
    past the point where their own per-90 features stop being NaN/flagged
    unreliable.

    Returns a Series of `code` values — merge/isin against it, don't assume
    it lines up with any other table's index.
    """
    from pipeline.features import MIN_RELIABLE_TRAILING_MINUTES
    from pipeline.fpl_current_season import build_current_season_table

    excluded_codes = set(cold_start.loc[~cold_start["include"], "code"])
    current = build_current_season_table()
    if current.empty:
        return pd.Series([], dtype=cold_start["code"].dtype)

    minutes_by_code = current.groupby("code")["minutes"].sum()
    qualifying = minutes_by_code[
        (minutes_by_code >= MIN_RELIABLE_TRAILING_MINUTES) & minutes_by_code.index.isin(excluded_codes)
    ]
    logger.info(
        "graduated %d new arrival(s) into the main model on real current-season minutes: %s",
        len(qualifying), qualifying.index.tolist(),
    )
    return pd.Series(qualifying.index)


def build_gw_feature_rows(target_event: int, horizon: int = 1) -> pd.DataFrame:
    """Candidate pool + features for one target gameweek. Trailing/origin
    features come from a season-weighted blend of last season's archive and
    this season's games-so-far (see _load_blended_player_log) — only the
    target fixture (difficulty, home/away, opponent) and `horizon` change
    between calls, exactly mirroring how training rows for different
    horizons off the same origin gameweek only differ in those same two
    things.

    `target_event` is the literal FPL gameweek to pull fixtures for (see
    get_next_gameweek() for computing it — never hardcode this); `horizon`
    is the separate 1-4 "weeks from now" value fed to the GLM, matching how
    it was trained (target_round - origin_round). These are NOT the same
    number once the season is underway — conflating them was a real bug
    (predicted_points silently used the wrong horizon and the wrong
    fixtures for every gameweek after the season's first).
    """
    hist = _load_blended_player_log()
    hist = _add_trailing_features(hist)
    snap = _origin_snapshot(hist)
    # one row per player: their state as of the last gameweek they featured in.
    # `code` (the season-stable id) already rode along from build_dataset.py's
    # cold-start join, so no extra lookup is needed to get from element -> code.
    last_snap = snap.sort_values("round").groupby("element", as_index=False).last()

    live_players = load_raw("fpl_live_players")
    live_teams = load_raw("fpl_live_teams")
    live_fixtures = load_raw("fpl_live_fixtures")
    cold_start = load_processed("cold_start_exclusions")

    # Prefixed fx_* to avoid colliding with the same-named columns already
    # carried over from each player's historical last-gameweek-of-25-26 row.
    target_fixtures = live_fixtures[live_fixtures["event"] == target_event]
    fixture_rows = []
    for _, f in target_fixtures.iterrows():
        fixture_rows.append({"team": f["team_h"], "fx_difficulty": f["team_h_difficulty"], "fx_was_home": True, "fx_opponent": f["team_a"]})
        fixture_rows.append({"team": f["team_a"], "fx_difficulty": f["team_a_difficulty"], "fx_was_home": False, "fx_opponent": f["team_h"]})
    fixtures_by_team = pd.DataFrame(fixture_rows).set_index("team")

    eligible_codes = pd.concat([
        cold_start.loc[cold_start["include"], "code"],
        graduated_new_arrivals(cold_start),
    ]).drop_duplicates()
    pool = live_players.merge(eligible_codes.to_frame("code"), on="code", how="inner")

    available = (pool["status"] == "a") | (
        (pool["status"] == "d") & (pool["chance_of_playing_next_round"].fillna(0) >= 75)
    )
    pool = pool[available].copy()

    pool = pool.merge(last_snap, on="code", how="inner", suffixes=("", "_hist"))
    pool = pool.merge(fixtures_by_team, left_on="team", right_index=True, how="inner")

    team_name = live_teams.set_index("id")["name"]
    pool["team_name"] = pool["team"].map(team_name)
    pool["opponent_name"] = pool["fx_opponent"].map(team_name)
    pool["position_name"] = pool["element_type"].map(POSITION_NAME)
    pool["full_name"] = pool["first_name"] + " " + pool["second_name"]
    pool["price"] = pool["now_cost"] / 10

    pool["horizon"] = horizon
    pool["value"] = pool["now_cost"]
    pool["target_difficulty"] = pool["fx_difficulty"]
    pool["target_was_home"] = pool["fx_was_home"]
    pool[POSITION_COLUMN] = pool["position_name"].map({"GKP": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"})

    pool = fill_missing_form(pool)

    # `was_home` here would collide with the stale historical was_home carried
    # over from each player's last 25-26 gameweek row (still present from the
    # last_snap merge) — drop that one first so the rename can't produce two
    # same-named columns holding two different fixtures' values.
    pool = pool.drop(columns=["was_home"]).rename(columns={"fx_was_home": "was_home"})

    logger.info(
        "GW%d candidate pool: %d eligible, available players with a real fixture",
        target_event, len(pool),
    )
    return pool.reset_index(drop=True)


def build_gw1_feature_rows() -> pd.DataFrame:
    """Backwards-compatible alias — GW1 specifically."""
    return build_gw_feature_rows(1)


def fetch_lineup_data() -> tuple[pd.DataFrame, set]:
    """One RotoWire fetch, reusable across every horizon's compute_p_plays_60
    call — see predict_multi_gw, which is the caller that actually needs
    this (predict_gw's default single-call path fetches for itself)."""
    live_players = load_raw("fpl_live_players")
    live_teams = load_raw("fpl_live_teams")
    return get_start_probabilities(live_players, live_teams)


def compute_p_plays_60(
    pool: pd.DataFrame, use_lineups: bool = True, lineup_data: tuple | None = None, trust_lineup: bool = True,
    horizon: int = 1, recalibrators: dict | None = None, next_week_start: pd.Series | None = None,
) -> pd.DataFrame:
    """Start-probability blend, in priority order:

    1. RotoWire predicted/confirmed lineup, if the player's team has one
       published AND `trust_lineup` is True — forward-looking, current-
       squad-aware information (summer transfers, new signings, this
       week's fitness) that no trailing stat can carry. Overrides the
       proxy outright rather than averaging with it: a confirmed lineup
       and a stale trend aren't two votes of equal standing.
    2. Team has a published lineup but this player isn't in the matched
       11 -> BENCH_PROB. This is what catches the Darlow case: he won't
       be in Man Utd's list at all once someone else is confirmed in
       goal, regardless of how many minutes he played last season
       elsewhere.
    3. No lineup published yet for that team, `use_lineups=False`, or
       `trust_lineup=False` -> fall back to the per-horizon start model
       (fit_p60_recalibration) on START_MODEL_COLS: the exponentially
       weighted start rate (recency-weighted, so a player who just won a
       starting place isn't held back by benchings earlier in the season),
       average minutes when starting (a full-90 regular vs one usually
       subbed around the hour) and season start share (one recent gap
       weighs less for an established starter). For weeks 2+, next week's
       lineup (when published) is folded into the start rate as one more
       game first.

    RotoWire only ever publishes a lineup for the *next* upcoming
    gameweek — there's no real GW2-4 lineup to fetch. `trust_lineup`
    exists specifically so callers predicting horizon > 1 can say so:
    passing trust_lineup=False for those calls means this function
    doesn't apply the current week's lineup read to a week it was never
    actually about — a real gap fixed after being diagnosed in
    conversation (every confirmed starter got the same flat probability
    for weeks 2-4 regardless of their own reliability pattern; a player
    who's frequently subbed early got the same forward-carried number as
    one who never leaves the pitch). Callers that still want the old
    carry-forward behavior can pass trust_lineup=True for every horizon.
    """
    inputs = pool[START_MODEL_COLS].astype(float).copy()
    inputs[START_RATE_COL] = inputs[START_RATE_COL].clip(0, 1)
    model_horizon = pd.Series(horizon, index=pool.index)
    source = pd.Series("trailing_starts", index=pool.index)
    if next_week_start is not None and horizon > 1:
        # Next week's lineup is still evidence about the weeks after it:
        # fold it into the start rate as one more game (`next_week_start`,
        # by code: 1 = named to start, 0 = left out of a published lineup;
        # see predict_multi_gw), then read it as a rate one week closer to
        # the target. Fixes a one-game injury absence dragging a returning
        # starter down for every later week. Archive walk-forward, actual
        # next start flipped 10% of the time (RotoWire's starters actually
        # started ~90% in GW5): Brier 0.2042 vs 0.2099 over horizons 2-8,
        # gains concentrated in 2-5.
        x = pool["code"].map(next_week_start)
        known = x.notna()
        rate = inputs[START_RATE_COL]
        updated = START_RATE_NEW_GAME_WEIGHT * x + (1 - START_RATE_NEW_GAME_WEIGHT) * rate
        inputs[START_RATE_COL] = rate.where(~known, updated)
        model_horizon = model_horizon.where(~known, horizon - 1)
        source = source.where(~known, "trailing_starts+lineup")
    trailing_p60 = inputs[START_RATE_COL].copy()
    if recalibrators:
        # Start-probability model (fit_p60_recalibration): P(60+) per
        # horizon from the start rate, minutes when starting and season start
        # share. Never applied to a RotoWire-sourced probability below —
        # that's real lineup evidence, not something this model estimates.
        for h in model_horizon.unique():
            rows = model_horizon == h
            if h in recalibrators:
                trailing_p60[rows] = recalibrators[h].predict_proba(inputs.loc[rows, START_MODEL_COLS])[:, 1]
    p60 = trailing_p60.copy()

    if not use_lineups or not trust_lineup:
        pool = pool.copy()
        pool["p_plays_60"] = p60
        pool["p_plays_60_source"] = source
        return pool

    probs, _ = lineup_data if lineup_data is not None else fetch_lineup_data()

    if not probs.empty:
        # RotoWire can carry more than one fixture per team at once — e.g. a
        # postponed/rearranged earlier-gameweek match still to be played
        # alongside the normal next-gameweek preview (hit directly: GW4 had
        # an outstanding Leeds-Newcastle fixture live at the same time as
        # GW5's real previews, giving Newcastle two lineup rows with two
        # different opponents). `code` alone isn't a safe lookup key here.
        # Disambiguate on the actual opponent this pool has the player
        # facing for the gameweek being scored, not on RotoWire's own
        # confirmed/predicted status label — "confirmed" means "more
        # imminent," not "the gameweek we're scoring."
        probs = probs.copy()
        probs["opponent_norm"] = probs["opponent"].apply(normalize_team)
        pool_opponent_norm = pool["opponent_name"].apply(normalize_team)
        pool_key = pool["code"].astype(str) + "|" + pool_opponent_norm
        probs["_key"] = probs["code"].astype(str) + "|" + probs["opponent_norm"]
        lineup_map = probs.drop_duplicates(subset="_key").set_index("_key")["p_plays_60_lineup"]

        has_lineup_entry = pool_key.isin(lineup_map.index)
        p60 = p60.where(~has_lineup_entry, pool_key.map(lineup_map))
        source = source.where(~has_lineup_entry, "lineup_starter")

        # A team only "has a lineup" for this gameweek if the lineup is for
        # this fixture: mid-gameweek, RotoWire still shows the current
        # round's previews for teams that haven't played yet, and those
        # must not mark every player of theirs benched for the next round.
        fixture_keys = set(probs["team"].apply(normalize_team) + "|" + probs["opponent_norm"])
        team_has_lineup = (pool["team_name"].apply(normalize_team) + "|" + pool_opponent_norm).isin(fixture_keys)
        is_bench = team_has_lineup & ~has_lineup_entry
        p60 = p60.where(~is_bench, BENCH_PROB)
        source = source.where(~is_bench, "lineup_bench")

        logger.info(
            "p(plays 60+) source: %s",
            dict(source.value_counts()),
        )
    else:
        logger.warning("no RotoWire lineup data available; using trailing-minutes proxy for everyone")

    pool = pool.copy()
    pool["p_plays_60"] = p60
    pool["p_plays_60_source"] = source
    return pool


def fit_p60_recalibration(max_horizon: int = MAX_HORIZON) -> dict[int, HistGradientBoostingClassifier]:
    """Per-horizon start-probability model: P(plays 60+ at origin + h) from
    START_MODEL_COLS (recent start rate, average minutes over the last few
    starts, season start share), fit on real outcomes over the complete
    25-26 history.

    History: a one-input logistic recalibration of the start rate alone was
    used until 2026-10-11. It couldn't tell a full-90 regular back from one
    injury-missed game (actual 60+ rate next week 0.84, it said 0.69) from a
    starter usually subbed around the hour (actual 0.68, it said 0.85).
    Shallow monotone GBM on the three inputs: walk-forward Brier 0.1952 vs
    0.2045, better in 44/44 fold x horizon cells; a logistic model with
    interaction terms was close behind (0.1955). Monotone-increasing in all
    three inputs, so more starting can never lower the estimate.
    """
    hist_df = load_processed("player_gameweek_2025_26")
    return fit_start_models(hist_df, max_horizon=max_horizon)


START_MODEL_MIN_ROWS = 2000


def fit_start_models(
    hist_df: pd.DataFrame, max_horizon: int = MAX_HORIZON, max_target_round: int | None = None,
) -> dict[int, HistGradientBoostingClassifier]:
    """The per-horizon start models of fit_p60_recalibration, fit on one
    season's player log. `max_target_round` keeps only outcomes known by that
    round (the season backtest refits weekly with it); a horizon with fewer
    than START_MODEL_MIN_ROWS rows is skipped, and callers fall back to the
    raw start rate for it."""
    df2 = _add_trailing_features(hist_df)
    origin = _origin_snapshot(df2)
    origin = origin[origin["round"] >= MIN_ORIGIN_ROUND]
    origin = origin[origin["trail_games5"] > 0]
    origin[START_RATE_COL] = origin[START_RATE_COL].clip(0, 1)
    targets = hist_df.groupby(["element", "round"], as_index=False).agg(target_minutes=("minutes", "sum"))

    recalibrators = {}
    for h in range(1, max_horizon + 1):
        block = origin.copy()
        block["target_round"] = block["round"] + h
        block = block.merge(
            targets, left_on=["element", "target_round"], right_on=["element", "round"],
            how="inner", suffixes=("", "_target"),
        )
        if max_target_round is not None:
            block = block[block["target_round"] <= max_target_round]
        if len(block) < START_MODEL_MIN_ROWS:
            continue
        y = (block["target_minutes"] >= 60).astype(int).values
        model = HistGradientBoostingClassifier(
            max_depth=3, min_samples_leaf=200, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            monotonic_cst=[1] * len(START_MODEL_COLS), early_stopping=False, random_state=0,
        )
        model.fit(block[START_MODEL_COLS].astype(float), y)
        recalibrators[h] = model
    return recalibrators


def _fit_production_models():
    """Fit all three production pieces once on the full 25-26 history (the
    GLM also gets 2022-24 premium attackers, see extra_seasons): the
    60+ NB2 GLM, the MAE-optimal floor for everything below it (see
    fit_floor_mae_optimal), and the per-horizon p_plays_60 recalibration
    (see fit_p60_recalibration). Reused across every horizon in
    predict_multi_gw rather than refit per gameweek — there's nothing
    horizon-specific about the GLM or floor fit, both are read off each
    player's last 25-26 appearance; the recalibration IS horizon-specific
    but is itself fit once, up front, covering every horizon at once.
    """
    from pipeline.features import feature_matrix

    hist_df = load_processed("player_gameweek_2025_26")
    full_df = build_horizon_table(hist_df, target_min_minutes=0)
    X_full, _ = feature_matrix(full_df)

    # GLM: 25-26 60+ rows plus 2022-24 premium MID/FWD rows with their own
    # season intercepts and price slopes (see extra_seasons). The floor and
    # the start model stay on 25-26 only: premium-only seasons would skew a
    # population-level constant and a start model.
    train_60 = (full_df["target_minutes"] >= 60).values
    y_train, X_train = with_premium_seasons(full_df[train_60], X_full[train_60])
    res, train_cols = fit_production_glm(y_train, X_train)
    floor = fit_floor_mae_optimal(full_df, X_full, res)
    recalibrators = fit_p60_recalibration()
    return res, train_cols, floor, recalibrators


PLAYER_ADJUST_K = 80


def fit_player_adjustments(res, train_cols) -> pd.Series:
    """Per-player correction for persistent model error this season:
    log(sum(actual + offset) / sum(predicted + offset)) over the player's
    one-week-ahead 60+ minute games so far, shrunk toward 0 by
    n / (n + PLAYER_ADJUST_K). Returned by `code`, applied multiplicatively
    to quality_pred in predict_gw.

    The GLM is population-only: two players with the same features get the
    same prediction, so it can't learn that one player keeps beating it.
    Purged walk-forward on 25-26 (adjustments from games before each
    validation block): top 3 per position per week scored +0.09 ± 0.05
    points more (+0.11 ± 0.04 at k=40), MAE neutral (-0.0012 ± 0.0021);
    k=80 chosen on folds 1-3. It does not fix the premium over-prediction:
    Haaland's block-to-block errors swing sign, so there's no stable trait to
    learn. Only this season's games are used (that is what was validated),
    so the effect is near zero early on and grows as games accumulate.
    """
    from pipeline.features import feature_matrix
    from pipeline.fpl_current_season import build_current_season_table

    current = build_current_season_table()
    if current.empty:
        return pd.Series(dtype=float)
    return player_adjustments_from(build_horizon_table(current, horizons=(1,)), res, train_cols)


def player_adjustments_from(h1: pd.DataFrame, res, train_cols) -> pd.Series:
    """fit_player_adjustments' calculation on any table of one-week-ahead,
    60+ minute rows (the backtest passes the games known by each week)."""
    from pipeline.features import feature_matrix

    if h1.empty:
        return pd.Series(dtype=float)
    X, _ = feature_matrix(h1)
    X = sm.add_constant(X.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    X = X.reindex(columns=train_cols, fill_value=0.0)
    games = pd.DataFrame({
        "code": h1["code"].values,
        "actual": h1["target_points"].values + GLM_OFFSET,
        "predicted": np.asarray(res.predict(X)),
    }).groupby("code").agg(n=("actual", "size"), actual=("actual", "sum"), predicted=("predicted", "sum"))
    weight = games["n"] / (games["n"] + PLAYER_ADJUST_K)
    adjust = weight * np.log(games["actual"] / games["predicted"])
    logger.info(
        "player adjustments: %d players, up to %d games each, largest |multiplier - 1| %.3f",
        len(adjust), int(games["n"].max()), float(np.expm1(adjust.abs()).max()),
    )
    return adjust


def predict_gw(
    target_event: int, horizon: int = 1, res=None, train_cols=None, floor: float | None = None,
    use_lineups: bool = True, lineup_data: tuple | None = None, recalibrators: dict | None = None,
    next_week_start: pd.Series | None = None, player_adjust: pd.Series | None = None,
) -> pd.DataFrame:
    """Score the entire eligible candidate pool for one target gameweek
    (not just a top-N slice) — this is what the squad optimizer draws
    from, since a budget/formation/club-limit-constrained selection needs
    the full pool, not a pre-cut top 50 that might exclude an affordable
    enabler a squad needs.

    `target_event` is the literal gameweek to fetch fixtures for; `horizon`
    is the separate 1-4 value fed to the model — see build_gw_feature_rows.
    Callers predicting the very next gameweek should pass
    target_event=get_next_gameweek(), horizon=1, not a hardcoded 1.

    Pass a pre-fit (res, train_cols, floor) and pre-fetched lineup_data when
    scoring multiple gameweeks in one call (see predict_multi_gw) so the
    GLM/floor are fit once and RotoWire is fetched once, then carried
    forward to every horizon rather than re-fetched (there's nothing new to
    fetch — see compute_p_plays_60 for why the same snapshot still applies).
    """
    from pipeline.features import feature_matrix

    if res is None or train_cols is None or floor is None:
        res, train_cols, floor, recalibrators = _fit_production_models()

    pool = build_gw_feature_rows(target_event, horizon)
    Xp, _ = feature_matrix(pool)
    Xp = sm.add_constant(Xp.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xp = Xp.reindex(columns=train_cols, fill_value=0.0)

    pred = np.asarray(res.predict(Xp)) - GLM_OFFSET
    pool = pool.copy()
    pool["quality_pred"] = pred  # E[points | plays 60+ min] — the piece this GLM actually models
    if player_adjust is not None:
        adjust = pool["code"].map(player_adjust).fillna(0.0).values
        pool["quality_pred"] = (pool["quality_pred"] + GLM_OFFSET) * np.exp(adjust) - GLM_OFFSET

    # DesignDoc.md's formula is E[points] = P(60+min)*E[points|60+min] + ...,
    # with start probability explicitly "sourced from data directly ...
    # NOT modeled from scratch." See compute_p_plays_60 for the source
    # priority (lineup data first, trailing starts fraction as fallback). `floor` is
    # the MAE-optimal fitted constant for "doesn't get 60+ minutes" (see
    # fit_floor_mae_optimal) — replaces the old hardcoded FLOOR_POINTS=0.5.
    # RotoWire's fetched lineup only really speaks to the literal next
    # gameweek — trust it as an override only for horizon==1; for horizon
    # 2-4, compute_p_plays_60 falls back to the recalibrated start rate instead of
    # blindly carrying the horizon-1 lineup read forward (see its
    # docstring for the failure mode this fixes).
    pool = compute_p_plays_60(
        pool, use_lineups=use_lineups, lineup_data=lineup_data, trust_lineup=(horizon == 1),
        horizon=horizon, recalibrators=recalibrators, next_week_start=next_week_start,
    )
    pool["predicted_points"] = (
        pool["p_plays_60"] * pool["quality_pred"] + (1 - pool["p_plays_60"]) * floor
    )
    return pool.reset_index(drop=True)


def predict_all() -> pd.DataFrame:
    """Backwards-compatible alias — the next upcoming gameweek."""
    return predict_gw(get_next_gameweek(), 1)


def predict_multi_gw(horizons: tuple[int, ...] = (1, 2, 3, 4), discount: float = 0.95) -> pd.DataFrame:
    """Score every candidate across several upcoming gameweeks and combine
    into one discounted multi-gameweek predicted_points column — the
    objective coefficient a squad-selection optimization should use when
    building a squad to hold across several weeks, not just the next one.

    weight(h) = discount ** (h - 1): gameweek 1 is undiscounted, each week
    further out counts a little less. `discount` close to 1.0 barely
    discounts at all; something like 0.9-0.97 is a reasonable "small"
    default — it's a genuine judgment call, not fit to data, reflecting
    that a fixed squad decided today has more (if imperfect) certainty
    about next week than about four weeks out, and in real play you'd get
    to react with a transfer before later gameweeks arrive anyway, which
    this static, no-swaps-within-the-window optimization can't capture.

    Blank gameweeks score 0 for that week (outer merge) and double
    gameweeks sum both fixtures, so the per-week columns are safe to feed a
    multi-period planner over long horizons.

    The RotoWire lineup snapshot (next-gameweek predicted/confirmed
    starters) is fetched once and applied to every horizon, not just the
    first — see compute_p_plays_60's docstring for why that carry-forward
    is preferred over reverting to the trailing-minutes proxy for GW2-4.
    """
    res, train_cols, floor, recalibrators = _fit_production_models()
    player_adjust = fit_player_adjustments(res, train_cols)
    lineup_data = fetch_lineup_data()
    next_gw = get_next_gameweek()
    logger.info("next gameweek is GW%d — targeting GW%d-%d", next_gw, next_gw, next_gw + max(horizons) - 1)

    per_gw = {}
    combined = None
    next_week_start = None
    for h in sorted(horizons):
        target_event = next_gw + h - 1
        gw_pool = predict_gw(
            target_event, h, res=res, train_cols=train_cols, floor=floor, lineup_data=lineup_data,
            recalibrators=recalibrators, next_week_start=next_week_start, player_adjust=player_adjust,
        )
        if h == 1:
            # Next week's lineup read (named starter 1, benched 0) carried
            # into the start rate for every later week — see compute_p_plays_60.
            lineup_rows = gw_pool["p_plays_60_source"].isin(["lineup_starter", "lineup_bench"])
            next_week_start = (
                (gw_pool.loc[lineup_rows, "p_plays_60_source"] == "lineup_starter").astype(float)
                .groupby(gw_pool.loc[lineup_rows, "code"]).max()
            )
        weight = discount ** (h - 1)
        gw_pool = gw_pool.rename(columns={
            "predicted_points": f"predicted_points_gw{h}",
            "quality_pred": f"quality_pred_gw{h}",
            "p_plays_60": f"p_plays_60_gw{h}",
            "p_plays_60_source": f"p_plays_60_source_gw{h}",
            "target_difficulty": f"target_difficulty_gw{h}",
            "opponent_name": f"opponent_gw{h}",
            "was_home": f"was_home_gw{h}",
        })
        per_gw[h] = (gw_pool, weight)
        keep_cols = [f"quality_pred_gw{h}", f"p_plays_60_gw{h}", f"p_plays_60_source_gw{h}",
                     f"target_difficulty_gw{h}", f"opponent_gw{h}", f"was_home_gw{h}"]
        slim = gw_pool.groupby("code").agg(
            {f"predicted_points_gw{h}": "sum", **{c: "first" for c in keep_cols}}
        ).reset_index()
        combined = slim if combined is None else combined.merge(slim, on="code", how="outer")

    base_cols = ["code", "web_name", "full_name", "team_name", "position_name", "price", "selected_by_percent"]
    base = pd.concat([pool[base_cols] for pool, _ in per_gw.values()]).drop_duplicates("code")
    combined = base.merge(combined, on="code", how="inner")
    pp_cols = [f"predicted_points_gw{h}" for h in horizons]
    combined[pp_cols] = combined[pp_cols].fillna(0.0)

    combined["predicted_points"] = sum(
        weight * combined[f"predicted_points_gw{h}"] for h, (_, weight) in per_gw.items()
    )
    combined["horizons_used"] = str(list(horizons))
    combined["discount"] = discount
    logger.info(
        "multi-GW predictions: %d players, horizons=%s, discount=%.3f, %d dropped (missing a fixture in the window)",
        len(combined), list(horizons), discount, len(base) - len(combined),
    )
    return combined


ENABLER_POINTS_PER_START = 2.0  # nominal flat return for a cheap player if they play — not a quality estimate


def build_enabler_candidates(
    n_per_position: int = 2,
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    discount: float = 0.95,
    lineup_data: tuple | None = None,
) -> pd.DataFrame:
    """The cold-start filter is doing its job correctly — it excludes
    players (mostly at newly-promoted clubs) with no reliable 25-26 PL
    track record, because there's no trustworthy way to model their
    *scoring rate*. But that leaves the optimizer with no true rock-bottom
    budget option: the cheapest ELIGIBLE forward, for instance, is £5.5m,
    because every genuinely cheap forward got filtered out along with the
    quality-model risk they'd have carried.

    This doesn't reintroduce them as scored candidates for the quality
    model — it adds a small, separate pool of cheap cold-start-excluded
    players per position, split into two confidence tiers:

    1. RotoWire lists them as a predicted/confirmed starter — real
       evidence they'll be on the pitch. Scored at
       ENABLER_POINTS_PER_START x p(start).
    2. Not found in any lineup at all — genuinely unknown, but still a
       legal, cheap, *available* (not injured/departed) squad-filler
       option. A pure bench slot barely affects the objective (only the
       small bench tie-break in select_squad rewards it at all), so
       requiring lineup confirmation here was leaving real, cheaper
       fodder on the table for no benefit — the true price floor by
       position (checked directly against cold-start-excluded players
       regardless of lineup status) sits a full price tier below what
       tier-1-only was finding for forwards specifically. Scored at
       BENCH_PROB instead of a real probability, i.e. treated the same
       as "confirmed NOT starting" — deliberately pessimistic, since
       there's no positive evidence either way.

    Confirmed starters still rank first within a price tier (their score
    is real evidence, tier 2's is a floor), so this only changes anything
    when tier 2 is strictly cheaper.
    """
    cold_start = load_processed("cold_start_exclusions")
    excluded = cold_start[~cold_start["include"]][["code", "reason"]]
    # Players who've graduated into the main scored pool (see
    # graduated_new_arrivals) get a real, model-based score there — drop
    # them here so they don't also show up as an unscored flat-value enabler.
    excluded = excluded[~excluded["code"].isin(graduated_new_arrivals(cold_start))]

    live_players = load_raw("fpl_live_players")
    live_teams = load_raw("fpl_live_teams")
    probs, _ = lineup_data if lineup_data is not None else fetch_lineup_data()

    lp = live_players.copy()
    lp["price"] = lp["now_cost"] / 10
    lp["position_name"] = lp["element_type"].map(POSITION_NAME)
    lp["team_name"] = lp["team"].map(live_teams.set_index("id")["name"])
    lp["full_name"] = lp["first_name"] + " " + lp["second_name"]

    # same availability bar as the main pool: genuinely injured/departed
    # players are excluded outright, not just scored low.
    available = (lp["status"] == "a") | (
        (lp["status"] == "d") & (lp["chance_of_playing_next_round"].fillna(0) >= 75)
    )
    lp = lp[available]

    # `probs` only ever contains matched *starters* (see lineups.parse_lineups
    # — bench isn't parsed at all); left-joining (rather than the tier-1-only
    # inner join this used to be) keeps not-found players in with NaN, which
    # the fillna below turns into the BENCH_PROB floor.
    candidates = excluded.merge(lp, on="code", how="inner")
    if not probs.empty:
        candidates = candidates.merge(probs[["code", "p_plays_60_lineup"]], on="code", how="left")
    else:
        candidates["p_plays_60_lineup"] = pd.NA
    candidates["confirmed_starter"] = candidates["p_plays_60_lineup"].notna()
    candidates["p_plays_60_lineup"] = candidates["p_plays_60_lineup"].fillna(BENCH_PROB)

    weight_sum = sum(discount ** (h - 1) for h in horizons)
    candidates["predicted_points"] = (
        ENABLER_POINTS_PER_START * candidates["p_plays_60_lineup"] * weight_sum
    )
    for h in horizons:  # per-week columns for the multi-period planner
        candidates[f"predicted_points_gw{h}"] = ENABLER_POINTS_PER_START * candidates["p_plays_60_lineup"]
    candidates["is_enabler_candidate"] = True
    candidates["p_plays_60"] = candidates["p_plays_60_lineup"]
    candidates["p_plays_60_source"] = candidates["confirmed_starter"].map(
        {True: "lineup_starter_cold_start_enabler", False: "unconfirmed_cold_start_enabler"}
    )

    picked = (
        candidates.sort_values(["position_name", "price", "predicted_points"], ascending=[True, True, False])
        .groupby("position_name", group_keys=False)
        .head(n_per_position)
    )

    keep_cols = [
        "code", "web_name", "full_name", "team_name", "position_name", "price",
        "predicted_points", "p_plays_60", "p_plays_60_source", "selected_by_percent",
        "reason", "is_enabler_candidate",
    ] + [f"predicted_points_gw{h}" for h in horizons]
    logger.info(
        "enabler candidates: %d added (%s)",
        len(picked), dict(picked.groupby("position_name")["web_name"].apply(list)),
    )
    return picked[keep_cols].reset_index(drop=True)


def predict_top_n(n: int = 50) -> pd.DataFrame:
    pool = predict_all()
    top = pool.sort_values("predicted_points", ascending=False).head(n)
    out = top[
        [
            "full_name", "web_name", "team_name", "position_name", "price",
            "predicted_points", "quality_pred", "p_plays_60", "p_plays_60_source",
            "opponent_name", "was_home", "target_difficulty",
            "selected_by_percent",
        ]
    ].reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    table = predict_top_n(50)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    table.to_csv("data/gw1_top50_glm_predictions.csv", index=False)
