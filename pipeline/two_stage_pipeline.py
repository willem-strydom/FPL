"""Closes a gap named explicitly in conversation: every validation number
reported so far was for the 60+-minute quality model in isolation, scored
only on rows where the player actually reached 60 minutes. Nothing
downstream of it — the p(plays 60+) blend, the flat FLOOR_POINTS fallback
for everything else — had ever been checked against real outcomes.

This module fits a second GLM for the 1-59-minute band (the "sub"
appearance case DesignDoc.md's decomposition formula names but this
pipeline never modeled), combines it with the existing 60+ GLM into the
full three-term formula, and walk-forward validates the *combined* output
against ALL rows (unfiltered — 0 minutes, sub minutes, 60+ minutes
together), which is the number anyone actually consuming a prediction
would see. Four pipelines are compared on identical folds and identical
(unfiltered) validation rows:

  - NEW: P(60+)*E[pts|60+] + P(sub)*E[pts|sub] + P(dnp)*E[pts|dnp]
  - OLD (hardcoded floor): P(60+)*E[pts|60+] + (1-P(60+))*0.5 — the
    original pipeline, FLOOR_POINTS never fit to anything.
  - OLD (MAE-optimal floor): same formula, floor replaced by the
    per-fold weighted-median-optimal constant (see
    `weighted_median_floor`) — this is NOT the naive population mean of
    the <60-minute band (that was tried and made things worse: the
    (1-p60) weight concentrates on low-p60 players, whose true
    conditional outcome skews lower than the unconditional mean, and
    with points heavily zero-inflated/right-skewed, the MAE-minimizing
    constant is nowhere near the mean anyway — it's a weighted median).
  - Rolling-avg baseline: trail_points5, no model at all

P(60+) here is the historical trail_minutes5/90 proxy, not RotoWire —
there's no archived predicted-lineups feed to backtest against, only the
live current snapshot (see pipeline/lineups.py). P(sub | not 60+) and the
MAE-optimal floor are both computed fresh per fold from that fold's
training data only (no leakage).
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm

from pipeline.config import COLD_START_MIN_MINUTES, COLD_START_MIN_STARTS
from pipeline.features import build_horizon_table, feature_matrix
from pipeline.model_compare import GLM_DROP_COL, GLM_OFFSET, evaluate, fit_predict_glm, make_folds
from pipeline.storage import load_processed

logger = logging.getLogger(__name__)

FLOOR_POINTS = 0.5  # the existing (old pipeline) flat constant for "didn't reach 60 min"

# Roughly half of everyone who ever appears in the archive is a fringe squad
# player who's basically never going to feature (reserve, youth graduate,
# season-ending injury) — the FPL API tracks every registered player, not
# just realistic selections. build_horizon_table's `trail_games5 > 0` origin
# filter already prunes most of them out, and the 60+ GLM self-filters
# almost completely since a true fringe player can't generate a "reached 60
# minutes" outcome row. But the DNP band is exactly where they concentrate
# (measured: ~33% of DNP training rows vs ~1% of 60+ rows), which drags on
# any single constant fit over that population and then applied only to
# cold-start-eligible (i.e. non-fringe, by construction) prediction
# candidates. Reusing the cold-start bar here, but computed as an
# expanding/cumulative-to-origin stat (see features.py) instead of the full
# season, keeps it walk-forward-safe.
ESTABLISHED_MIN_STARTS = COLD_START_MIN_STARTS
ESTABLISHED_MIN_MINUTES = COLD_START_MIN_MINUTES

# On top of career tenure (established, above), also require recent playing
# time: a player can clear the tenure bar early in the season and then fall
# out of the team entirely, at which point they're just as irrelevant to a
# live decision as someone who never cleared it. Deliberately NOT a filter
# on trail_points5 / target_points — that would drop legitimate "plays a lot,
# doesn't score much" rows (real, common, ~25% of established 60+ outcomes —
# see conversation) and bias the model toward over-predicting everyone.
# trail_minutes5 is a pure recency-of-selection signal, known at origin, and
# doesn't carry that risk.
PLAUSIBLE_MIN_TRAIL_MINUTES = 45


def fit_predict_glm_poisson(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame):
    """Same shape as model_compare.fit_predict_glm, but plain Poisson with a
    dynamic offset — the sub-60 band came back underdispersed (Pearson
    dispersion ~0.34, NB2's alpha collapses to ~0 and its AIC is worse than
    Poisson's), the opposite finding from the 60+ band, so NB2 isn't
    justified here. Offset is computed from the training fold's own
    minimum rather than reusing GLM_OFFSET, since a different subpopulation
    isn't guaranteed to share the same floor.
    """
    Xtr = sm.add_constant(X_train.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xva = sm.add_constant(X_val.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)

    offset = float(-y_train.min() + 1)
    y_shifted = y_train + offset
    res = sm.GLM(y_shifted, Xtr, family=sm.families.Poisson()).fit()
    pred = np.asarray(res.predict(Xva)) - offset
    return pred, res


def _predict_with_fitted_glm(res, X: pd.DataFrame, offset: float) -> np.ndarray:
    """Reuse an already-fit GLM's result object to predict on a different X
    (e.g. the full training set, not just the held-out fold) — avoids
    re-fitting the same model twice just to get a second set of predictions.
    """
    Xc = sm.add_constant(X.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xc = Xc.reindex(columns=res.model.exog_names, fill_value=0.0)
    return np.asarray(res.predict(Xc)) - offset


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The MAE-optimal single constant for a weighted absolute-error
    problem — not a weighted mean, which would be RMSE-optimal instead
    (see module docstring for why the two differ substantially here).
    """
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cutoff = w.sum() / 2.0
    idx = np.searchsorted(np.cumsum(w), cutoff)
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def run_full_pipeline_walkforward(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 2, 3, 4)):
    """Returns (results_df, band_results_df). results_df is the headline
    NEW vs OLD vs rolling-avg comparison on all (unfiltered) validation
    rows per fold. band_results_df breaks the same three predictions out
    by which minutes band each validation row actually fell into, plus the
    sub-60 GLM's own standalone accuracy on its native band — useful for
    seeing *where* any improvement is coming from, not just the totals.
    """
    full_df = build_horizon_table(df, horizons=horizons, target_min_minutes=0)
    X, _ = feature_matrix(full_df)
    folds = make_folds(full_df["origin_round"].unique())
    logger.info("running %d walk-forward folds over the FULL (unfiltered) target population", len(folds))

    results = []
    band_results = []
    pred_rows = []
    for i, (train_range, val_range) in enumerate(folds, start=1):
        train_mask = full_df["origin_round"].between(*train_range)
        val_mask = full_df["origin_round"].between(*val_range)
        if val_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        train_df, val_df = full_df[train_mask], full_df[val_mask]
        Xtr_all, Xva_all = X[train_mask], X[val_mask]

        # --- "plausible candidate" mask: is this player, as of this exact
        # origin snapshot, someone a live decision would ever be weighed
        # against — not a reserve, not someone who's since fallen out of the
        # team? Two origin-time-only signals (leakage-safe, never looks at
        # this row's own target_minutes/target_points):
        #   - tenure: cumulative starts/minutes to date clears the same bar
        #     cold_start.py uses for live prediction eligibility
        #   - recency: trail_minutes5 shows they're currently getting run
        #     out, not just proven once and since discarded
        # Deliberately NOT a filter on trail_points5/target_points — that
        # would drop legitimate "plays a lot, doesn't score much" rows and
        # bias the model toward over-predicting everyone (see conversation:
        # ~25% of established players who reach 60+ minutes trail under 2
        # points/gw — real players, not noise).
        plausible = (
            (train_df["cum_starts_to_date"] >= ESTABLISHED_MIN_STARTS)
            & (train_df["cum_minutes_to_date"] >= ESTABLISHED_MIN_MINUTES)
            & (train_df["trail_minutes5"] >= PLAUSIBLE_MIN_TRAIL_MINUTES)
        )

        # --- fit both quality models on their own band, using ALL training
        # rows (not restricted to `plausible`). Tried restricting this too —
        # measured directly, made things worse even scored against the
        # matching plausible-only validation slice (MAE 2.441->2.456 for the
        # calibrated-floor pipeline; every other variant moved the same
        # direction). Reason: unlike the floor (a single constant with no
        # covariates, which genuinely needed a matched population), the GLM
        # already has per-row features — trail_minutes5, form_points_per90,
        # etc. — that let it tell a fringe player from a regular within the
        # model itself; a fringe row already predicts low because its
        # trailing stats are low, not because it's contaminating a shared
        # number. Restricting training here doesn't sharpen anything the
        # model couldn't already do — it just deletes real data (the sub-60
        # GLM's fold-1 training set dropped 83%, 3,666 -> 624 rows) and adds
        # noise. `plausible` is still used below, for the rate constants and
        # (via the caller) for validation reporting — both of those lack the
        # GLM's per-row escape hatch and demonstrably needed the filter.
        train_60 = train_df["target_minutes"] >= 60
        train_sub = train_df["target_minutes"].between(1, 59)
        train_dnp = train_df["target_minutes"] == 0

        glm60_pred, glm60_res = fit_predict_glm(
            Xtr_all[train_60.values], train_df.loc[train_60, "target_points"].values, Xva_all
        )
        glmsub_pred, _ = fit_predict_glm_poisson(
            Xtr_all[train_sub.values], train_df.loc[train_sub, "target_points"].values, Xva_all
        )

        # --- probabilities: P(60+) from the historical minutes proxy, P(sub | not 60+)
        # as a single rate empirically measured on this fold's training data only,
        # restricted to plausible-candidate rows — see above. ---
        p60 = (val_df["trail_minutes5"] / 90).clip(0, 1).values
        train_sub_est = train_sub & plausible
        train_dnp_est = train_dnp & plausible
        if (train_sub_est.sum() + train_dnp_est.sum()) == 0:
            logger.warning("fold %d: no plausible rows in sub/dnp bands, falling back to unfiltered", i)
            train_sub_est, train_dnp_est = train_sub, train_dnp
        p_sub_given_not60 = train_sub_est.sum() / (train_sub_est.sum() + train_dnp_est.sum())
        dnp_baseline = train_df.loc[train_dnp_est, "target_points"].mean() if train_dnp_est.sum() else 0.0

        # --- MAE-optimal floor for the OLD (2-term) formula, fit on training
        # data only. NOT the naive mean of the <60-minute band (tried first;
        # made MAE worse in every fold — see module docstring) — this solves
        # min_floor sum |actual - p60*q60 - (1-p60)*floor| via a weighted
        # median over ALL training rows, weighted by (1-p60) itself, so the
        # fit is correct where the floor's influence on the formula is
        # actually large.
        #
        # Tried restricting this fit to established rows too (matching
        # p_sub_given_not60 above) and measured it directly: it makes things
        # worse, not better. Established players mostly carry high p60,
        # hence small (1-p60) weight, so the established-only subset isn't
        # just smaller (4,136 vs 13,808 rows in fold 1) but far more thinly
        # weighted (total weight 963 vs 7,399) — the fit ends up dominated by
        # a handful of established-but-currently-rotation-risk players whose
        # residual/weight ratio is inherently noisy (dividing by a small
        # weight amplifies whatever residual they happen to have). Result:
        # the floor swung from a stable ~-0.10 to an unstable -0.95 -> -0.39
        # across folds and made every downstream metric worse (see
        # conversation). Left unfiltered — the large, low-variance mass of
        # fringe rows this constant is disproportionately fit on is a real
        # and separately-documented problem (see module docstring / ongoing
        # discussion), but re-filtering the same point-estimate isn't the
        # fix; it just trades one bias for a worse, noisier one.
        p60_train = (train_df["trail_minutes5"] / 90).clip(0, 1).values
        glm60_pred_train = _predict_with_fitted_glm(glm60_res, Xtr_all, GLM_OFFSET)
        weight_train = 1 - p60_train
        resid_train = train_df["target_points"].values - p60_train * glm60_pred_train
        valid = weight_train > 1e-6
        floor_mae_optimal = weighted_median(resid_train[valid] / weight_train[valid], weight_train[valid])

        p_sub = (1 - p60) * p_sub_given_not60
        p_dnp = (1 - p60) * (1 - p_sub_given_not60)

        new_pred = p60 * glm60_pred + p_sub * glmsub_pred + p_dnp * dnp_baseline
        old_pred = p60 * glm60_pred + (1 - p60) * FLOOR_POINTS
        old_mae_opt_pred = p60 * glm60_pred + (1 - p60) * floor_mae_optimal
        rolling_avg_pred = val_df["trail_points5"].fillna(dnp_baseline).values

        y_val = val_df["target_points"].values
        logger.info(
            "fold %d: train origin %s (n=%d, %d/%d/%d in 60+/sub/dnp, %d plausible of which "
            "%d/%d sub/dnp) -> val origin %s (n=%d), P(sub|not60+)=%.3f, dnp_baseline=%.3f, "
            "MAE-optimal floor=%.4f (hardcoded=%.1f)",
            i, train_range, len(train_df), train_60.sum(), train_sub.sum(), train_dnp.sum(),
            plausible.sum(), train_sub_est.sum(), train_dnp_est.sum(),
            val_range, len(val_df), p_sub_given_not60, dnp_baseline, floor_mae_optimal, FLOOR_POINTS,
        )

        for model_name, pred in [
            ("NEW: 3-term pipeline", new_pred),
            ("OLD: 2-term pipeline (floor=0.5 hardcoded)", old_pred),
            ("OLD: 2-term pipeline (MAE-optimal floor)", old_mae_opt_pred),
            ("Rolling-avg baseline", rolling_avg_pred),
        ]:
            results.append({
                "fold": i, "train_range": f"{train_range[0]}-{train_range[1]}",
                "val_range": f"{val_range[0]}-{val_range[1]}", "n_val": len(y_val),
                "model": model_name, **evaluate(y_val, pred),
            })

        val_band = pd.cut(
            val_df["target_minutes"], bins=[-0.1, 0, 59, 200], labels=["dnp", "sub60", "60plus"]
        )
        for band in ["dnp", "sub60", "60plus"]:
            band_mask = (val_band == band).values
            if band_mask.sum() == 0:
                continue
            for model_name, pred in [
                ("NEW: 3-term pipeline", new_pred),
                ("OLD: 2-term pipeline (floor=0.5 hardcoded)", old_pred),
                ("OLD: 2-term pipeline (MAE-optimal floor)", old_mae_opt_pred),
                ("Rolling-avg baseline", rolling_avg_pred),
                ("sub-60 GLM (own band only)", glmsub_pred),
            ]:
                band_results.append({
                    "fold": i, "band": band, "n_val": int(band_mask.sum()), "model": model_name,
                    **evaluate(y_val[band_mask], pred[band_mask]),
                })

        pred_rows.append(pd.DataFrame({
            "fold": i,
            "element": val_df["element"].values,
            "code": val_df["code"].values if "code" in val_df.columns else np.nan,
            "origin_round": val_df["origin_round"].values,
            "horizon": val_df["horizon"].values,
            "position": val_df["position"].values,
            "band": val_band.values,
            "target_points": y_val,
            "new_pred": new_pred,
            "old_pred": old_pred,
            "old_mae_opt_pred": old_mae_opt_pred,
            "rolling_avg_pred": rolling_avg_pred,
        }))

    predictions_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    return pd.DataFrame(results), pd.DataFrame(band_results), predictions_df


def _weighted_agg(results_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    def wavg(g):
        w = g["n_val"]
        return pd.Series({
            "mae": (g["mae"] * w).sum() / w.sum(),
            "rmse": (g["rmse"] * w).sum() / w.sum(),
            "bias": (g["bias"] * w).sum() / w.sum(),
            "n_val": w.sum(),
        })
    return results_df.groupby(group_cols).apply(wavg, include_groups=False).reset_index()


MODEL_COLS = {
    "NEW: 3-term pipeline": "new_pred",
    "OLD: 2-term pipeline (floor=0.5 hardcoded)": "old_pred",
    "OLD: 2-term pipeline (MAE-optimal floor)": "old_mae_opt_pred",
    "Rolling-avg baseline": "rolling_avg_pred",
}


def _evaluate_by(predictions_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Generic partition-and-score: for every distinct value of group_col,
    compute MAE/RMSE/bias for all three predictions against target_points.
    """
    rows = []
    for group_val, g in predictions_df.groupby(group_col):
        for model_name, col in MODEL_COLS.items():
            rows.append({
                group_col: group_val, "n": len(g), "model": model_name,
                **evaluate(g["target_points"].values, g[col].values),
            })
    return pd.DataFrame(rows)


def metrics_by_ownership(
    predictions_df: pd.DataFrame, ownership: pd.Series, thresholds: tuple[float, ...] = (0, 2, 5, 10)
) -> pd.DataFrame:
    df = predictions_df.copy()
    df["ownership"] = df["element"].map(ownership)
    df = df.dropna(subset=["ownership"])
    rows = []
    for thresh in thresholds:
        sub = df[df["ownership"] >= thresh]
        for model_name, col in MODEL_COLS.items():
            rows.append({
                "min_ownership": thresh, "n": len(sub), "n_players": sub["element"].nunique(),
                "model": model_name, **evaluate(sub["target_points"].values, sub[col].values),
            })
    return pd.DataFrame(rows)


def metrics_by_time_of_season(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Coarse early/mid/late tertiles over the validated origin-gameweek
    range, on top of the fine-grained per-fold table already produced by
    run_full_pipeline_walkforward — same underlying data, easier to read
    as a single seasonal trend.
    """
    df = predictions_df.copy()
    lo, hi = df["origin_round"].min(), df["origin_round"].max()
    edges = [lo - 1, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]
    labels = [f"early (GW{lo}-{int(edges[1])})", f"mid (GW{int(edges[1])+1}-{int(edges[2])})", f"late (GW{int(edges[2])+1}-{hi})"]
    df["season_third"] = pd.cut(df["origin_round"], bins=edges, labels=labels, include_lowest=True)
    return _evaluate_by(df, "season_third")


def metrics_by_position(predictions_df: pd.DataFrame) -> pd.DataFrame:
    return _evaluate_by(predictions_df, "position")


if __name__ == "__main__":
    import logging as _logging

    from pipeline.storage import load_raw

    _logging.basicConfig(level=_logging.INFO)
    df = load_processed("player_gameweek_2025_26")
    results_df, band_results_df, predictions_df = run_full_pipeline_walkforward(df)

    results_df.to_csv("data/two_stage_walkforward_results.csv", index=False)
    band_results_df.to_csv("data/two_stage_band_results.csv", index=False)
    predictions_df.to_csv("data/two_stage_predictions.csv", index=False)

    print("=== Overall (all rows, all folds, weighted) ===")
    print(_weighted_agg(results_df, ["model"]).to_string(index=False))
    print()
    print("=== Per fold (time of season, fine-grained) ===")
    print(results_df.sort_values(["fold", "model"]).to_string(index=False))
    print()
    print("=== By season third (time of season, coarse) ===")
    season_df = metrics_by_time_of_season(predictions_df)
    season_df.to_csv("data/two_stage_by_season_third.csv", index=False)
    print(season_df.to_string(index=False))
    print()
    print("=== By minutes band (weighted across folds) ===")
    print(_weighted_agg(band_results_df, ["band", "model"]).to_string(index=False))
    print()
    print("=== By position ===")
    position_df = metrics_by_position(predictions_df)
    position_df.to_csv("data/two_stage_by_position.csv", index=False)
    print(position_df.to_string(index=False))
    print()
    print("=== By ownership floor ===")
    players_raw = load_raw("fpl_hist_players_raw")
    ownership = players_raw.set_index("id")["selected_by_percent"]
    ownership_df = metrics_by_ownership(predictions_df, ownership)
    ownership_df.to_csv("data/two_stage_by_ownership.csv", index=False)
    print(ownership_df.to_string(index=False))
