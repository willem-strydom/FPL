"""Sliding-window (walk-forward) comparison of the two model families named
in DesignDoc.md's "Model choice" section for the quality/form model:

  1. Poisson (or Negative Binomial, if overdispersed) GLM
  2. A shallow, heavily regularized GBM

Both are fit on the horizon-as-feature table from pipeline/features.py and
evaluated with an expanding-window walk-forward split over ORIGIN gameweek,
satisfying both CV requirements from DesignDoc.md's Validation Methodology:

  1. All horizon-rows from one (player, origin gameweek) decision point
     share the same origin_round, so splitting on origin_round ranges never
     separates a group across train/val.
  2. Training windows only ever precede validation windows in time.
"""
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as dm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)

GLM_OFFSET = 2   # total_points is very rarely negative (min observed: -2); shift so the
                 # count-model target is non-negative, then shift predictions back.
GLM_DROP_COL = "pos_GK"  # baseline position dummy, dropped to avoid the dummy-variable trap

INITIAL_TRAIN_END = 13   # first fold trains on origin_round in [MIN_ORIGIN_ROUND, 13]
VAL_BLOCK = 4             # each subsequent validation block spans 4 origin gameweeks

GBM_PARAMS = dict(
    max_depth=3,             # shallow, per DesignDoc.md
    max_leaf_nodes=15,
    min_samples_leaf=30,
    l2_regularization=1.0,   # heavily regularized, per DesignDoc.md
    learning_rate=0.05,
    max_iter=300,
    random_state=0,
)


def make_folds(
    origin_rounds, initial_train_end: int = INITIAL_TRAIN_END, val_block: int = VAL_BLOCK
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    rounds = sorted(set(origin_rounds))
    min_r, max_r = rounds[0], rounds[-1]
    folds = []
    val_start = initial_train_end + 1
    while val_start <= max_r:
        val_end = min(val_start + val_block - 1, max_r)
        folds.append(((min_r, val_start - 1), (val_start, val_end)))
        val_start = val_end + 1
    return folds


def fit_predict_glm(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame):
    Xtr = sm.add_constant(X_train.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xva = sm.add_constant(X_val.drop(columns=[GLM_DROP_COL]).astype(float), has_constant="add")
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)

    y_shifted = y_train + GLM_OFFSET
    res = dm.NegativeBinomial(y_shifted, Xtr, loglike_method="nb2").fit(disp=0, maxiter=200)
    pred = np.asarray(res.predict(Xva)) - GLM_OFFSET
    return pred, res


def fit_predict_gbm(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, **overrides):
    params = {**GBM_PARAMS, **overrides}
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    return pred, model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "bias": float(np.mean(y_pred - y_true)),  # mean(pred - actual): + = over-predicts
    }


def run_walkforward(
    horizon_df: pd.DataFrame, X: pd.DataFrame, return_predictions: bool = False
):
    """Fit GLM/GBM/train-mean-baseline per fold and evaluate on that fold's
    validation block. Returns the per-fold/per-horizon aggregated results
    table; pass return_predictions=True to also get back a per-row frame
    (element, origin_round, horizon, target_points, and each model's raw
    prediction) for slicing by attributes not baked into the fold loop
    itself — e.g. ownership-threshold filtering.
    """
    folds = make_folds(horizon_df["origin_round"].unique())
    logger.info("running %d walk-forward folds", len(folds))

    results = []
    pred_rows = []
    for i, (train_range, val_range) in enumerate(folds, start=1):
        train_mask = horizon_df["origin_round"].between(*train_range)
        val_mask = horizon_df["origin_round"].between(*val_range)
        Xtr, ytr = X[train_mask], horizon_df.loc[train_mask, "target_points"].values
        Xva, yva = X[val_mask], horizon_df.loc[val_mask, "target_points"].values
        if len(Xva) == 0 or len(Xtr) == 0:
            continue

        logger.info(
            "fold %d: train origin %s (n=%d) -> val origin %s (n=%d)",
            i, train_range, len(Xtr), val_range, len(Xva),
        )
        glm_pred, _ = fit_predict_glm(Xtr, ytr, Xva)
        gbm_pred, _ = fit_predict_gbm(Xtr, ytr, Xva)
        baseline_pred = np.full(len(yva), ytr.mean())

        horizons_val = horizon_df.loc[val_mask, "horizon"].values
        for horizon_label, mask in [("all", np.ones(len(yva), dtype=bool))] + [
            (h, horizons_val == h) for h in sorted(horizon_df["horizon"].unique())
        ]:
            if mask.sum() == 0:
                continue
            for model_name, pred in [
                ("NB2 GLM", glm_pred), ("Shallow GBM", gbm_pred), ("Baseline: train mean", baseline_pred),
            ]:
                results.append(
                    {
                        "fold": i,
                        "train_range": f"{train_range[0]}-{train_range[1]}",
                        "val_range": f"{val_range[0]}-{val_range[1]}",
                        "horizon": horizon_label,
                        "n_val": int(mask.sum()),
                        "model": model_name,
                        **evaluate(yva[mask], pred[mask]),
                    }
                )

        if return_predictions:
            val_idx = horizon_df.index[val_mask]
            pred_rows.append(
                pd.DataFrame(
                    {
                        "fold": i,
                        "element": horizon_df.loc[val_idx, "element"].values,
                        "origin_round": horizon_df.loc[val_idx, "origin_round"].values,
                        "horizon": horizons_val,
                        "target_points": yva,
                        "glm_pred": glm_pred,
                        "gbm_pred": gbm_pred,
                        "baseline_pred": baseline_pred,
                    },
                    index=val_idx,
                )
            )

    results_df = pd.DataFrame(results)
    if return_predictions:
        predictions_df = pd.concat(pred_rows) if pred_rows else pd.DataFrame()
        return results_df, predictions_df
    return results_df


def ownership_filtered_metrics(
    predictions_df: pd.DataFrame, ownership: pd.Series, thresholds: tuple[float, ...] = (0, 2, 5, 10)
) -> pd.DataFrame:
    """Re-aggregate MAE/RMSE for GLM, GBM, and the baseline after dropping
    players below each season-ownership threshold (selected_by_percent).
    Reuses the predictions already produced by run_walkforward — no refit.
    """
    df = predictions_df.copy()
    df["ownership"] = df["element"].map(ownership)
    n_missing = df["ownership"].isna().sum()
    if n_missing:
        logger.warning(
            "%d/%d prediction rows have no ownership figure (dropped from this comparison)",
            n_missing, len(df),
        )
    df = df.dropna(subset=["ownership"])

    rows = []
    for thresh in thresholds:
        sub = df[df["ownership"] >= thresh]
        n_players = sub["element"].nunique()
        for model_name, col in [
            ("NB2 GLM", "glm_pred"), ("Shallow GBM", "gbm_pred"), ("Baseline: train mean", "baseline_pred"),
        ]:
            m = evaluate(sub["target_points"].values, sub[col].values)
            rows.append(
                {
                    "min_ownership": thresh,
                    "n_rows": len(sub),
                    "n_players": n_players,
                    "model": model_name,
                    **m,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import logging as _logging

    from pipeline.features import build_horizon_table, feature_matrix
    from pipeline.storage import load_processed

    _logging.basicConfig(level=_logging.INFO)
    df = load_processed("player_gameweek_2025_26")
    horizon_df = build_horizon_table(df)
    X, _ = feature_matrix(horizon_df)
    results = run_walkforward(horizon_df, X)
    print(results[results["horizon"] == "all"].to_string())
