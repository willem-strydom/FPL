"""Derive chip thresholds (recommend.CHIP_THRESHOLDS) from the season backtest.

    python -m pipeline.chip_thresholds [--seeds 8] [--workers 7]

First half of 2025-26 (decision weeks with target <= GW19), along the tuned
planner's trajectory. Per decision week, on clean (un-jittered) predictions:
  WC = planner objective with wildcard - without (8-week decayed scale)
  FH = best one-week XI (incl. captain) - the planned week-1 XI
  BB = planned week-1 bench points
  TC = planned week-1 captain points
Then a hindsight schedule per seed: each chip exactly once, at most one chip
per gameweek, maximizing total gain; a chip's threshold is its gain in its
scheduled week, averaged over seeds. Thresholds are assumed to hold for the
second half too (fresh chip set from GW20).

Three artifacts the first derivation hit, all handled here:
- starting from the ownership-built GW1 squad inflated the wildcard (29.4),
  so the trajectory starts from the model's own wildcard squad;
- measuring gains on jittered predictions let WC/FH cherry-pick lucky noise,
  so gains use clean predictions while the trajectory follows jittered
  decisions (that's what makes seeds differ);
- the model-built starting week itself inflated bench boost (15.1), so it is
  excluded from the schedule.
"""
import argparse
import itertools
import logging
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from pipeline.config import DATA_DIR

LAST_FIRST_HALF_GW = 19
CHIPS = ["WC", "FH", "BB", "TC"]
CACHE_PATH = DATA_DIR / "backtest_week_cache.pkl"
NOISE_SD = 0.25


def gains_for_seed(seed: int) -> list[dict]:
    logging.basicConfig(level=logging.WARNING)
    from pipeline.optimizer import plan_transfers, select_squad
    from pipeline.policy_backtest import SeasonBacktest, next_free_transfers

    bt = SeasonBacktest()
    bt.week_cache = pickle.loads(CACHE_PATH.read_bytes())
    cache = bt.week_cache
    squad, ft, rows = set(bt.initial_squad()), 1, []
    weeks = sorted(k for k in cache if k + 1 <= LAST_FIRST_HALF_GW)
    for r in weeks:
        clean = bt._ensure_owned(cache[r], squad, r)
        if r == weeks[0]:
            squad, ft = set(plan_transfers(clean, squad, ft, wildcard=True)["squad"]["code"]), 1
            clean = bt._ensure_owned(cache[r], squad, r)
        else:
            normal = plan_transfers(clean, squad, ft)
            wild = plan_transfers(clean, squad, ft, wildcard=True)
            fh = select_squad(clean.assign(predicted_points=clean["predicted_points_gw1"]))
            rows.append({
                "seed": seed, "gw": r + 1,
                "WC": wild["objective"] - normal["objective"],
                "FH": fh["starting_predicted_points"] - normal["plan"][0]["xi_points"],
                "BB": float(normal["bench"]["predicted_points"].sum()),
                "TC": float(normal["captain"]["predicted_points"]),
            })
        cols = [c for c in cache[r].columns if c.startswith("predicted_points_gw")]
        rng = np.random.default_rng([seed, r])
        noisy = cache[r].copy()
        noisy[cols] = noisy[cols].values + rng.normal(0, NOISE_SD, (len(noisy), len(cols))) * (noisy[cols].values != 0)
        decision = plan_transfers(bt._ensure_owned(noisy, squad, r), squad, ft)
        squad = set(decision["squad"]["code"])
        ft = next_free_transfers(ft, decision["transfer_info"]["transfers_used"])
    return rows


def schedule(g: pd.DataFrame) -> dict:
    """Best assignment of the four chips to distinct weeks (exhaustive)."""
    weeks, vals = g["gw"].tolist(), g[CHIPS].values
    best, best_total = None, -np.inf
    for combo in itertools.permutations(range(len(weeks)), len(CHIPS)):
        total = sum(vals[w, c] for c, w in enumerate(combo))
        if total > best_total:
            best, best_total = combo, total
    return {chip: (weeks[w], vals[w, c]) for c, (chip, w) in enumerate(zip(CHIPS, best))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from pipeline.policy_backtest import SeasonBacktest

    CACHE_PATH.write_bytes(pickle.dumps(SeasonBacktest().build_week_cache()))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        g = pd.DataFrame([row for batch in ex.map(gains_for_seed, range(args.seeds)) for row in batch])

    out = []
    for seed, gs in g.groupby("seed"):
        sched = schedule(gs)
        for chip in CHIPS:
            out.append({"seed": seed, "chip": chip, "week": sched[chip][0], "gain": sched[chip][1], "max_gain": gs[chip].max()})
    o = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print(o.pivot(index="seed", columns="chip", values="gain").round(2).to_string())
    print(o.pivot(index="seed", columns="chip", values="week").to_string())
    summary = o.groupby("chip").agg(threshold=("gain", "mean"), sd=("gain", "std"), unconstrained_max=("max_gain", "mean"))
    print(summary.round(2).to_string())
    print("\nmean gain by week:")
    print(g.groupby("gw")[CHIPS].mean().round(2).to_string())
    names = {"WC": "wildcard", "FH": "freehit", "BB": "bboost", "TC": "3xc"}
    print("\nCHIP_THRESHOLDS =", {names[c]: round(float(summary.loc[c, "threshold"]), 1) for c in CHIPS})


if __name__ == "__main__":
    main()
