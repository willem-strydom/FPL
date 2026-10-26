"""Full-season walk-forward backtest of a transfer policy, scored on realized
(actual) points.

Two passes, split so many policies can share the expensive part:

  1. build_week_cache(): at each decision round r, refit the GLM/floor on
     training rows whose *target* round is <= r (only outcomes known at
     decision time) and predict every player for horizons 1..MAX_HORIZON.
     Policy-independent, so it's computed once and reused.
  2. run(policy): replay the season from a fixed GW1 squad, letting the
     policy decide each week's transfers, and sum the starting XI's realized
     points (captain doubled, hits subtracted).

Decisions use predicted points; scoring uses realized points. A single
replay is path-dependent (tiny prediction changes flip later transfers, and
season totals swing ~±100), so compare policies with compare(): several
seeds of prediction jitter, the same seeds for every policy.

The weekly model mirrors production: the GLM trains on 25-26 rows plus the
2022-24 premium attackers (all from before the season, so nothing leaks),
the player adjustment uses games known by the decision round, and the start
model is refit each week on outcomes known by then.

Known simplifications: no RotoWire lineups (so no next-week lineup carry-
over), no chips, no auto-subs, flat £100m budget at each week's prices, and
scoring starts at the first week with enough known outcomes to fit the model.
"""
import logging

import numpy as np
import pandas as pd
import pulp

from pipeline.features import (
    MAX_HORIZON,
    MIN_ORIGIN_ROUND,
    START_MODEL_COLS,
    START_RATE_COL,
    _add_trailing_features,
    _origin_snapshot,
    build_horizon_table,
    feature_matrix,
    fill_missing_form,
)
from pipeline.model_compare import GLM_OFFSET
from pipeline.optimizer import (
    MAX_BANKED_TRANSFERS,
    PLANNER_DECAY,
    PLANNER_FT_VALUE,
    PLANNER_HORIZON,
    PLANNER_TRANSFER_WEEKS,
    TRANSFER_HARD_COST,
    load_squad_rules,
    plan_transfers,
    select_squad,
)
from pipeline.predict_next_gw import (
    fit_floor_mae_optimal,
    fit_production_glm,
    fit_start_models,
    player_adjustments_from,
    with_premium_seasons,
)
from pipeline.storage import load_processed
from pipeline.two_stage_pipeline import _predict_with_fitted_glm

logger = logging.getLogger(__name__)

POSITION_MAP = {"GK": "GKP", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
LAST_ROUND = 38
MIN_TRAINING_ROWS = 500
BASE_COLS = ["code", "web_name", "team_name", "position_name", "price"]

# The pre-planner production objective: a 4-week, 0.95-discounted sum fed to
# the single-period select_squad.
LEGACY_HORIZON = 4
LEGACY_DISCOUNT = 0.95


def gw_col(h: int) -> str:
    return f"predicted_points_gw{h}"


def next_free_transfers(free_transfers: int, used: int) -> int:
    # Using more than your bank costs points but never goes negative: next week
    # just starts from the single new free transfer.
    if used <= free_transfers:
        return min(MAX_BANKED_TRANSFERS, free_transfers - used + 1)
    return 1


def production_policy():
    def decide(pool, squad, free_transfers):
        cols = [gw_col(h) for h in range(1, LEGACY_HORIZON + 1) if gw_col(h) in pool]
        agg = pool.assign(predicted_points=sum(LEGACY_DISCOUNT ** k * pool[c] for k, c in enumerate(cols)))
        result = select_squad(agg, current_squad_codes=squad, free_transfers=free_transfers)
        return result, result["transfer_info"]["transfers_used"] if result["transfer_info"] else 0
    return decide


def multiperiod_policy(horizon: int = PLANNER_HORIZON, decay: float = PLANNER_DECAY,
                       ft_value: float = PLANNER_FT_VALUE, transfer_weeks: int | None = PLANNER_TRANSFER_WEEKS):
    def decide(pool, squad, free_transfers):
        result = plan_transfers(pool, squad, free_transfers, horizon=horizon, decay=decay,
                                ft_value=ft_value, transfer_weeks=transfer_weeks)
        return result, result["transfer_info"]["transfers_used"]
    return decide


class SeasonBacktest:
    def __init__(self, season_table: str = "player_gameweek_2025_26"):
        hist = load_processed(season_table)
        hist["kickoff_time"] = pd.to_datetime(hist["kickoff_time"])
        from pipeline import availability

        self.snapshots = availability.load("2025-26")
        hist = availability.add_flags(hist, self.snapshots)
        self.hist = hist

        origin = _origin_snapshot(_add_trailing_features(hist))
        origin = origin.merge(availability.origin_table(self.snapshots), on=["element", "round"], how="left")
        origin["avail_now"] = origin["avail_now"].fillna(1.0)
        origin["live_eligible"] = origin["live_eligible"].fillna(True).astype(bool)
        origin["price"] = origin["value"] / 10.0
        origin["position_name"] = origin["position"].map(POSITION_MAP)
        origin["team_name"] = origin["team"]
        origin["web_name"] = origin["name"]
        self.origin = origin

        # One row per (team, round, fixture): double gameweeks keep both fixtures.
        self.team_fixture = hist.groupby(["team", "round", "fixture"], as_index=False).first()[
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
        snap = fill_missing_form(snap)

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
        train = (self.horizon_df["target_round"] <= r).values
        if train.sum() < MIN_TRAINING_ROWS:
            return None
        train_60 = train & (self.horizon_df["target_minutes"] >= 60).values
        y_train, X_train = with_premium_seasons(self.horizon_df[train_60], self.X_glm[train_60])
        res, train_cols = fit_production_glm(y_train, X_train)
        floor = fit_floor_mae_optimal(self.horizon_df[train], self.X_glm[train], res)

        Xp, _ = feature_matrix(pool)
        quality = _predict_with_fitted_glm(res, Xp, GLM_OFFSET)
        h1_known = train_60 & (self.horizon_df["horizon"] == 1).values
        adjust = player_adjustments_from(self.horizon_df[h1_known], res, train_cols)
        quality = (quality + GLM_OFFSET) * np.exp(pool["code"].map(adjust).fillna(0.0).values) - GLM_OFFSET

        start_models = fit_start_models(self.hist, max_target_round=r, snapshots=self.snapshots)
        p60 = pool[START_RATE_COL].clip(0, 1).values.copy()
        for h, model in start_models.items():
            rows = (pool["horizon"] == h).values
            if rows.any():
                p60[rows] = model.predict_proba(pool.loc[rows, START_MODEL_COLS].astype(float))[:, 1]
        # Like the live pool, players flagged out at the deadline aren't scored:
        # they keep their squad slot at no expected minutes.
        p60 = np.where(pool["live_eligible"].values, p60, 0.0)
        pool = pool.assign(pp=p60 * quality + (1 - p60) * floor)

        # Blank weeks -> 0; double gameweeks -> sum of both fixtures.
        wide = pool.pivot_table(index="code", columns="horizon", values="pp", aggfunc="sum").fillna(0.0)
        wide.columns = [gw_col(h) for h in wide.columns]
        base = pool.drop_duplicates("code")[BASE_COLS]
        return base.merge(wide.reset_index(), on="code", how="left")

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

    def _ensure_owned(self, pool: pd.DataFrame, squad: set, r: int) -> pd.DataFrame:
        """A player who's left the league still sits in your squad until sold:
        add them back with zero predictions rather than dropping them."""
        missing = squad - set(pool["code"])
        if not missing:
            return pool
        last = (self.origin[self.origin["code"].isin(missing) & (self.origin["round"] <= r)]
                .sort_values("round").groupby("code").last().reset_index()[BASE_COLS])
        return pd.concat([pool, last], ignore_index=True).fillna(0.0)

    def _realized_week(self, result: dict, target_round: int) -> float:
        captain = result["captain"]["code"]
        total = 0.0
        for code in result["starters"]["code"]:
            pts = self.realized.get((code, target_round), 0.0)
            total += 2 * pts if code == captain else pts
        return total

    def run(self, policy, seed: int | None = None, noise_sd: float = 0.25) -> float:
        """Realized points from the first cached week onward. With a seed, each
        week's predictions get N(0, noise_sd) jitter that depends only on
        (seed, week) — identical for every policy run with that seed."""
        cache = self.build_week_cache()
        squad = set(self.initial_squad())
        free_transfers = 1
        total = 0.0

        for r in sorted(cache):
            pool = cache[r]
            if seed is not None:
                cols = [c for c in pool.columns if c.startswith("predicted_points_gw")]
                rng = np.random.default_rng([seed, r])
                noise = rng.normal(0.0, noise_sd, size=(len(pool), len(cols)))
                pool = pool.copy()
                pool[cols] = pool[cols].values + noise * (pool[cols].values != 0)
            pool = self._ensure_owned(pool, squad, r)

            result, used = policy(pool, squad, free_transfers)
            total += self._realized_week(result, r + 1) - TRANSFER_HARD_COST * max(0, used - free_transfers)
            squad = set(result["squad"]["code"])
            free_transfers = next_free_transfers(free_transfers, used)
        return total

    def compare(self, policies: dict, seeds=range(6), noise_sd: float = 0.25) -> pd.DataFrame:
        rows = []
        for seed in seeds:
            for name, policy in policies.items():
                score = self.run(policy, seed=seed, noise_sd=noise_sd)
                rows.append({"seed": seed, "policy": name, "score": score})
                logger.info("seed %s %s: %.0f", seed, name, score)
        return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, baseline: str) -> pd.DataFrame:
    wide = results.pivot(index="seed", columns="policy", values="score")
    out = pd.DataFrame({"mean": wide.mean(), "sd": wide.std(), "min": wide.min(), "max": wide.max()})
    diff = wide.sub(wide[baseline], axis=0)
    out[f"mean diff vs {baseline}"] = diff.mean()
    out["se of diff"] = diff.std() / np.sqrt(len(wide))
    out["seeds won"] = (diff > 0).sum()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bt = SeasonBacktest()
    first = min(bt.build_week_cache())
    print(f"decisions from GW{first}, scoring GW{first + 1}-{LAST_ROUND}")
    results = bt.compare({
        "production": production_policy(),
        "multiperiod (tuned)": multiperiod_policy(),
    })
    print(results.pivot(index="seed", columns="policy", values="score").to_string())
    print(summarize(results, baseline="production").round(1).to_string())
