"""Full-season walk-forward backtest of a transfer policy, scored on realized
(actual) points.

Two passes, split so many policies can share the expensive part:

  1. build_week_cache(): at each origin round, refit the GLM/floor on
     origin rounds <= r only and score every player for horizons 1-4.
     Policy-independent, so it's computed once and reused.
  2. run(policy): replay the season from a fixed GW1 squad, letting the
     policy decide each week's transfer count, and sum the starting XI's
     realized points (captain doubled, hard-cost hits subtracted).

Decisions use predicted points; scoring uses realized points. This is the
fitness function a policy search (grid or GA) calls.

Known simplifications: p60 is raw trail_starts5 (no recalibration), no
chips, no auto-subs, and the score covers GW5-38 only (the GW1 squad is held
through GW4 before the first model-driven decision).
"""
import logging
import math

import pandas as pd
import pulp

from pipeline.features import (
    DEFCON_HIT_RATE_COL,
    MIN_ORIGIN_ROUND,
    _PER90_SOURCE_COLS,
    _add_trailing_features,
    _origin_snapshot,
    build_horizon_table,
    feature_matrix,
)
from pipeline.model_compare import GLM_OFFSET
from pipeline.optimizer import TRANSFER_HARD_COST, load_squad_rules, select_squad
from pipeline.predict_next_gw import fit_floor_mae_optimal, fit_production_glm
from pipeline.storage import load_processed
from pipeline.two_stage_pipeline import _predict_with_fitted_glm

logger = logging.getLogger(__name__)

POSITION_MAP = {"GK": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
LAST_ROUND = 38
MAX_HORIZON = 4
DISCOUNT = 0.95
MAX_BANKED_TRANSFERS = 5

FPL_NATIVE_FORM_COLS = [c for c in _PER90_SOURCE_COLS.values() if "understat" not in c] + [DEFCON_HIT_RATE_COL]
UNDERSTAT_COLS = ["form_understat_xG_per90", "form_understat_xA_per90"]

# Best policy from the log-binomial grid search on the 25-26 season (1792 realized
# points, GW5-38). Anchored near alpha=6.57/beta=-1.07 (4 pts at F=1, 0 at F=5).
# Caveat: neighbouring cells score 55-120 points lower, so this is a fragile
# single-season optimum, not a robust one.
N_CHOOSE = 11
LOG_BINOM = {f: math.log(math.comb(N_CHOOSE, f)) for f in range(MAX_BANKED_TRANSFERS + 1)}
CHOSEN_ALPHA = 6.57
CHOSEN_BETA = -0.9


def logbinom_required_gain(alpha: float, beta: float):
    """required_gain(F) = alpha + beta * ln C(11, F): the net predicted gain a
    transfer must clear to be made, given F banked free transfers."""
    def fn(weeks_remaining: int, free_transfers: int) -> float:
        return alpha + beta * LOG_BINOM[min(MAX_BANKED_TRANSFERS, max(0, free_transfers))]
    return fn


def next_free_transfers(free_transfers: int, used: int) -> int:
    # Using more than your bank costs points but never goes negative: next week
    # just starts from the single new free transfer.
    if used <= free_transfers:
        return min(MAX_BANKED_TRANSFERS, free_transfers - used + 1)
    return 1


class SeasonBacktest:
    def __init__(self, season_table: str = "player_gameweek_2025_26"):
        hist = load_processed(season_table)
        hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"])

        origin = _origin_snapshot(_add_trailing_features(hist))
        origin["price"] = origin["value"] / 10.0
        origin["position_name"] = origin["position"].map(POSITION_MAP)
        origin["team_name"] = origin["team"]
        origin["web_name"] = origin["name"]
        self.origin = origin

        self.team_fixture = hist.groupby(["team", "round"], as_index=False).first()[
            ["team", "round", "own_team_difficulty", "was_home"]
        ]
        self.horizon_df = build_horizon_table(hist)
        self.X_glm, _ = feature_matrix(self.horizon_df)
        self.realized = hist.groupby(["code", "round"])["total_points"].sum()

        self.week_cache: dict[int, pd.DataFrame] | None = None
        self._initial_squad: set | None = None

    def _predict_pool(self, r: int) -> pd.DataFrame:
        # No trail_games5 > 0 filter: the cache is shared by policies holding
        # different squads, and hard-excluding an inactive owned player would
        # make "hold" infeasible. Inactive players just score low via p60.
        snap = self.origin[self.origin["round"] == r].copy()
        if snap.empty:
            return snap
        snap["form_unreliable"] = snap["form_points_per90"].isna().astype(int)
        snap[FPL_NATIVE_FORM_COLS] = snap[FPL_NATIVE_FORM_COLS].fillna(0.0)
        snap["understat_form_missing"] = snap["form_understat_xG_per90"].isna().astype(int)
        snap[UNDERSTAT_COLS] = snap[UNDERSTAT_COLS].fillna(0.0)

        blocks = []
        for h in range(1, MAX_HORIZON + 1):
            target = r + h
            if target > LAST_ROUND:
                break
            block = snap.drop(columns=["own_team_difficulty", "was_home"])
            block["horizon"] = h
            fixtures = self.team_fixture[self.team_fixture["round"] == target].drop(columns="round")
            block = block.merge(fixtures, on="team", how="inner").rename(
                columns={"own_team_difficulty": "target_difficulty", "was_home": "target_was_home"}
            )
            blocks.append(block)
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    def _score_week(self, r: int, pool: pd.DataFrame) -> pd.DataFrame | None:
        train = (self.horizon_df["origin_round"] <= r).values
        if train.sum() < 500:
            return None
        train_60 = train & (self.horizon_df["target_minutes"] >= 60).values
        res, _ = fit_production_glm(self.horizon_df[train_60], self.X_glm[train_60])
        floor = fit_floor_mae_optimal(self.horizon_df[train], self.X_glm[train], res)

        Xp, _ = feature_matrix(pool)
        quality = _predict_with_fitted_glm(res, Xp, GLM_OFFSET)
        p60 = pool["trail_starts5"].clip(0, 1).values
        pool = pool.copy()
        pool["pp_h"] = p60 * quality + (1 - p60) * floor
        pool["weighted_pp"] = (DISCOUNT ** (pool["horizon"] - 1)) * pool["pp_h"]

        agg = pool.groupby(["code", "web_name", "team_name", "position_name", "price"], as_index=False)[
            "weighted_pp"
        ].sum().rename(columns={"weighted_pp": "predicted_points"})
        # check_transfer_timing needs at least one predicted_points_gw{h} column.
        gw1 = pool[pool["horizon"] == 1][["code", "pp_h"]].rename(columns={"pp_h": "predicted_points_gw1"})
        agg = agg.merge(gw1, on="code", how="left")
        agg["predicted_points_gw1"] = agg["predicted_points_gw1"].fillna(agg["predicted_points"])
        return agg

    def build_week_cache(self) -> dict[int, pd.DataFrame]:
        if self.week_cache is None:
            cache = {}
            for r in range(MIN_ORIGIN_ROUND, LAST_ROUND):
                pool = self._predict_pool(r)
                if pool.empty:
                    continue
                scored = self._score_week(r, pool)
                if scored is not None:
                    cache[r] = scored
                    logger.info("cached round %d: %d candidates", r, len(scored))
            self.week_cache = cache
        return self.week_cache

    def initial_squad(self) -> set:
        """GW1 squad with no model input: max starting-XI ownership within
        budget, with a tiny bench-cost penalty to push spend into the XI."""
        if self._initial_squad is not None:
            return self._initial_squad
        snap = self.origin[self.origin["round"] == 1].reset_index(drop=True)
        own = snap["selected"].astype(float)
        rules = load_squad_rules()
        idx = snap.index.tolist()

        prob = pulp.LpProblem("initial_squad", pulp.LpMaximize)
        squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
        starter = pulp.LpVariable.dicts("starter", idx, cat="Binary")
        prob += pulp.lpSum(starter[i] * own[i] for i in idx) - 0.001 * pulp.lpSum(
            (squad[i] - starter[i]) * snap.loc[i, "price"] for i in idx
        )
        prob += pulp.lpSum(squad[i] for i in idx) == rules["squad_size"]
        prob += pulp.lpSum(starter[i] for i in idx) == rules["starting_size"]
        prob += pulp.lpSum(squad[i] * snap.loc[i, "price"] for i in idx) <= rules["budget"] + 1e-6
        for i in idx:
            prob += starter[i] <= squad[i]
        for pos, n in rules["squad_select"].items():
            pos_idx = snap.index[snap["position_name"] == pos].tolist()
            prob += pulp.lpSum(squad[i] for i in pos_idx) == n
            prob += pulp.lpSum(starter[i] for i in pos_idx) >= rules["min_play"][pos]
            prob += pulp.lpSum(starter[i] for i in pos_idx) <= rules["max_play"][pos]
        for club in snap["team_name"].dropna().unique():
            club_idx = snap.index[snap["team_name"] == club].tolist()
            prob += pulp.lpSum(squad[i] for i in club_idx) <= rules["club_limit"]
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        self._initial_squad = {snap.loc[i, "code"] for i in idx if round(squad[i].value()) == 1}
        return self._initial_squad

    def _realized_week(self, result: dict, target_round: int) -> float:
        captain = result["captain"]["code"]
        total = 0.0
        for code in result["starters"]["code"]:
            pts = self.realized.get((code, target_round), 0.0)
            total += 2 * pts if code == captain else pts
        return total

    def _threshold_decision(self, pool, squad, free_transfers, required_gain) -> tuple[dict, int]:
        options = {}
        for k in (0, 1, 2):
            try:
                options[k] = select_squad(
                    pool, current_squad_codes=squad, free_transfers=free_transfers, force_transfers=k
                )
            except RuntimeError:
                options[k] = None

        best_k = 0
        if options[0] is not None:
            base = options[0]["starting_predicted_points"]
            for k in (1, 2):
                if options[k] is None:
                    break
                net = options[k]["starting_predicted_points"] - TRANSFER_HARD_COST * max(0, k - free_transfers)
                if net - base > required_gain:
                    best_k, base = k, net
                else:
                    break

        if options[best_k] is None:
            # force_transfers=0 is occasionally infeasible; fall back to the
            # unforced optimizer rather than aborting the season.
            result = select_squad(pool, current_squad_codes=squad, free_transfers=free_transfers)
            return result, result["transfer_info"]["transfers_used"] if result["transfer_info"] else 0
        return options[best_k], best_k

    def run(self, required_gain_fn=None) -> float:
        """Realized GW5-38 points for a policy. required_gain_fn(weeks_remaining,
        free_transfers) -> threshold; None runs the production soft-cost
        heuristic as a baseline."""
        cache = self.build_week_cache()
        squad = set(self.initial_squad())
        free_transfers = 1
        total = 0.0

        for r in sorted(cache):
            target = r + 1
            pool = cache[r]
            if required_gain_fn is None:
                result = select_squad(pool, current_squad_codes=squad, free_transfers=free_transfers)
                used = result["transfer_info"]["transfers_used"] if result["transfer_info"] else 0
            else:
                threshold = required_gain_fn(LAST_ROUND - r, free_transfers)
                result, used = self._threshold_decision(pool, squad, free_transfers, threshold)

            total += self._realized_week(result, target) - TRANSFER_HARD_COST * max(0, used - free_transfers)
            squad = set(result["squad"]["code"])
            free_transfers = next_free_transfers(free_transfers, used)
        return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bt = SeasonBacktest()
    print(f"production heuristic: {bt.run():.0f}")
    print(f"chosen log-binomial policy: {bt.run(logbinom_required_gain(CHOSEN_ALPHA, CHOSEN_BETA)):.0f}")
