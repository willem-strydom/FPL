"""Chip timing as an optimal-stopping problem.

    python -m pipeline.chip_policy --build     # backtest chip gains + forecast errors (~15 min)
    python -m pipeline.chip_policy             # print thresholds by weeks left (no live forecasts)

Each half's chips expire at its deadline (GW19, then GW38) and only one chip
can be played per gameweek. With n weeks left in the half and a set S of
unused chips, the value of the chips still held is, working back from the
deadline,

    V(after deadline, S) = 0
    V(week j, S) = E[ max( V(j+1, S), max over c in S of gain_c(j) + V(j+1, S - c) ) ]

and this week's rule is: play the chip c with the largest
gain_c - (V(next, S) - V(next, S - c)), if that is positive. The threshold
V(next, S) - V(next, S - c) is the value of keeping c: it falls as the
deadline nears and depends on which other chips are still held. This
replaces fixed thresholds from a hindsight schedule, which assumed you could
pick each chip's best week after seeing the whole half (an upward bias).

Future weeks' gains:
- next 8 weeks, Bench Boost and Triple Captain: the planner's own forecast
  for that week (planned bench / captain points) plus a forecast error drawn
  from the backtest at the same lead time, so known double gameweeks count;
- everything else: joint draws of all four chips' weekly gains from the
  2025-26 backtest (one draw is one week of one seed, so the chips'
  correlation is kept).

Gains are measured as in the original threshold derivation, on clean
predictions along the tuned planner's (jittered) trajectory, starting from a
model-built squad, start week excluded:
  WC = planner objective with wildcard - without (decayed 8-week scale)
  FH = best one-week XI (incl. captain) - the planned week-1 XI
  BB = planned week-1 bench points, TC = planned week-1 captain points
"""
import argparse
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR

CHIPS = ["wildcard", "freehit", "bboost", "3xc"]
FORECAST_CHIPS = {"bboost": "bench_points", "3xc": "captain_points"}
HALF_DEADLINES = (19, 38)
NOISE_SD = 0.25
SAMPLES_PER_WEEK = 400
CACHE_PATH = DATA_DIR / "backtest_week_cache.pkl"
GAINS_PATH = DATA_DIR / "chip_gains.parquet"
ERRORS_PATH = DATA_DIR / "chip_forecast_errors.parquet"


# --- building the backtest data ---------------------------------------------

def _season_for_seed(seed: int) -> tuple[list[dict], list[dict]]:
    logging.basicConfig(level=logging.WARNING)
    from pipeline.optimizer import plan_transfers, select_squad
    from pipeline.policy_backtest import SeasonBacktest, next_free_transfers

    bt = SeasonBacktest()
    bt.week_cache = pickle.loads(CACHE_PATH.read_bytes())
    cache = bt.week_cache
    squad, ft = set(bt.initial_squad()), 1
    gains, forecasts = [], []
    weeks = sorted(cache)
    for r in weeks:
        clean = bt._ensure_owned(cache[r], squad, r)
        if r == weeks[0]:
            # start from the model's own squad, not the ownership-built one
            squad, ft = set(plan_transfers(clean, squad, ft, wildcard=True)["squad"]["code"]), 1
            clean = bt._ensure_owned(cache[r], squad, r)
        else:
            normal = plan_transfers(clean, squad, ft)
            wild = plan_transfers(clean, squad, ft, wildcard=True)
            fh = select_squad(clean.assign(predicted_points=clean["predicted_points_gw1"]))
            gains.append({
                "seed": seed, "gw": r + 1,
                "wildcard": wild["objective"] - normal["objective"],
                "freehit": fh["starting_predicted_points"] - normal["plan"][0]["xi_points"],
                "bboost": normal["plan"][0]["bench_points"],
                "3xc": normal["plan"][0]["captain_points"],
            })
            for wk in normal["plan"]:
                forecasts.append({"seed": seed, "made_gw": r + 1, "target_gw": r + wk["week"], "week": wk["week"],
                                  "bboost": wk["bench_points"], "3xc": wk["captain_points"]})
        cols = [c for c in cache[r].columns if c.startswith("predicted_points_gw")]
        rng = np.random.default_rng([seed, r])
        noisy = cache[r].copy()
        noisy[cols] = noisy[cols].values + rng.normal(0, NOISE_SD, (len(noisy), len(cols))) * (noisy[cols].values != 0)
        decision = plan_transfers(bt._ensure_owned(noisy, squad, r), squad, ft)
        squad = set(decision["squad"]["code"])
        ft = next_free_transfers(ft, decision["transfer_info"]["transfers_used"])
    return gains, forecasts


def build(seeds: int = 8, workers: int = 7) -> None:
    from pipeline.policy_backtest import SeasonBacktest

    CACHE_PATH.write_bytes(pickle.dumps(SeasonBacktest().build_week_cache()))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_season_for_seed, range(seeds)))
    gains = pd.DataFrame([g for r in results for g in r[0]])
    forecasts = pd.DataFrame([f for r in results for f in r[1]])
    # forecast error at lead time `week`: the gain actually computed when that
    # week arrived (on the trajectory actually taken) minus the forecast
    realized = gains.melt(id_vars=["seed", "gw"], value_vars=list(FORECAST_CHIPS), var_name="chip", value_name="realized")
    fc = forecasts[forecasts["week"] >= 2].melt(id_vars=["seed", "made_gw", "target_gw", "week"],
                                               value_vars=list(FORECAST_CHIPS), var_name="chip", value_name="forecast")
    errors = fc.merge(realized.rename(columns={"gw": "target_gw"}), on=["seed", "target_gw", "chip"], how="inner")
    errors["error"] = errors["realized"] - errors["forecast"]
    gains.to_parquet(GAINS_PATH)
    errors.to_parquet(ERRORS_PATH)
    logging.getLogger(__name__).warning("saved %d weekly gain rows and %d forecast errors", len(gains), len(errors))


# --- the decision rule ---------------------------------------------------------

def _load():
    if not GAINS_PATH.exists() or not ERRORS_PATH.exists():
        raise FileNotFoundError("chip backtest data missing: run `python -m pipeline.chip_policy --build`")
    return pd.read_parquet(GAINS_PATH), pd.read_parquet(ERRORS_PATH)


def deadline_for(gw: int) -> int:
    return next(d for d in HALF_DEADLINES if gw <= d)


def _future_samples(n_future: int, forecasts: dict | None, rng) -> list[np.ndarray]:
    """Joint gain draws (SAMPLES_PER_WEEK x 4, columns in CHIPS order) for
    each of the next n_future weeks."""
    gains, errors = _load()
    pool = gains[CHIPS].values
    out = []
    for j in range(n_future):
        week = j + 2  # plan week index of that future gameweek
        draw = pool[rng.integers(len(pool), size=SAMPLES_PER_WEEK)].copy()
        for chip, _ in FORECAST_CHIPS.items():
            f = (forecasts or {}).get(chip)
            if f is not None and week <= len(f):
                e = errors.loc[(errors["chip"] == chip) & (errors["week"] == week), "error"].values
                draw[:, CHIPS.index(chip)] = f[week - 1] + rng.choice(e, size=SAMPLES_PER_WEEK)
        out.append(np.clip(draw, 0, None))
    return out


def _subsets(chips):
    return [frozenset(c) for k in range(len(chips) + 1) for c in combinations(chips, k)]


def values_after_this_week(n_left: int, held: set, forecasts: dict | None = None, seed: int = 0) -> dict:
    """V(next week, S) for every subset S of the held chips, given n_left
    weeks left in the half including this one."""
    rng = np.random.default_rng(seed)
    held = [c for c in CHIPS if c in held]
    subsets = _subsets(held)
    V = {s: 0.0 for s in subsets}
    for draw in reversed(_future_samples(n_left - 1, forecasts, rng)):
        nxt = {}
        for s in subsets:
            best = np.full(len(draw), V[s])
            for c in s:
                best = np.maximum(best, draw[:, CHIPS.index(c)] + V[s - {c}])
            nxt[s] = float(best.mean())
        V = nxt
    return V


def decide(gains_now: dict, gw: int, held: set, forecasts: dict | None = None) -> dict:
    """Per held chip: this week's gain, the value of keeping it (threshold),
    and the margin; plus which chip (if any) to play."""
    n_left = deadline_for(gw) - gw + 1
    held = {c for c in held if c in CHIPS}
    V = values_after_this_week(n_left, held, forecasts)
    S = frozenset(held)
    rows = {c: {"gain": gains_now[c], "threshold": V[S] - V[S - {c}]} for c in held}
    for r in rows.values():
        r["margin"] = r["gain"] - r["threshold"]
    best = max(rows, key=lambda c: rows[c]["margin"], default=None)
    return {"weeks_left": n_left, "chips": rows, "play": best if best and rows[best]["margin"] > 0 else None}


def threshold_table(held: set = frozenset(CHIPS)) -> pd.DataFrame:
    """Thresholds by weeks left with all chips held and no live forecasts."""
    rows = []
    for n in (1, 2, 3, 4, 6, 8, 11, 15, 19):
        V = values_after_this_week(n, set(held))
        S = frozenset(held)
        rows.append({"weeks_left": n, **{c: V[S] - V[S - {c}] for c in CHIPS if c in held}})
    return pd.DataFrame(rows).set_index("weeks_left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="rerun the backtest to rebuild chip gains and forecast errors")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.build:
        build(args.seeds, args.workers)
    gains, errors = _load()
    pd.set_option("display.width", 200)
    print(f"{len(gains)} weekly gain draws; mean / sd / 90th pct:")
    print(gains[CHIPS].agg(["mean", "std", lambda s: s.quantile(0.9)]).round(2).to_string())
    print("\nforecast error sd by lead week (bench / captain):")
    print(errors.groupby(["week", "chip"])["error"].std().unstack().round(2).to_string())
    print("\nvalue of keeping each chip (threshold), all four held, by weeks left in the half:")
    print(threshold_table().round(1).to_string())


if __name__ == "__main__":
    main()
