"""Mathematical-programming squad selector — DesignDoc.md's "Optimizer
Notes" section, implemented directly:

  - "Standard expected-value maximization ... no variance penalty needed
    here." The objective is a plain linear sum of predicted points; no
    variance/risk term anywhere in the model.
  - "Captaincy: skip variance-adjustment machinery... best/highest-EV
    players rarely come off the bench." The optimizer doesn't get a
    separate captaincy sub-model — captain is just whichever selected
    starter has the highest predicted_points, enforced as a linear
    constraint (captain <= starter) rather than picked post-hoc, so it's
    still part of one joint optimization.
  - "Must exclude cold-start players ... from the candidate pool
    entirely, not just give low/zero predicted scores." Already true by
    construction: the pool this module receives (predict_next_gw.
    predict_all()) was built from the cold-start-filtered candidate list,
    so those players are absent rows, not zero-score rows.
  - "Do not double-penalize bench-risk players by also discounting in the
    optimizer objective function." predicted_points already blends
    p(plays 60+) x quality — the objective here uses that number as-is,
    with no second rotation-risk discount layered on top.

Squad-building rules (squad size, budget, starting-XI size, max players
per club, per-position min/max in the starting XI) are read from the
official API's own game_settings / element_types tables, not hardcoded —
if FPL changes a rule mid-season the optimizer picks it up automatically
on the next ingest.
"""
import logging

import pandas as pd
import pulp

from pipeline.storage import load_raw

logger = logging.getLogger(__name__)

# Real FPL rule: a transfer beyond your banked free-transfer allowance costs
# a flat 4 points, always, regardless of how many you have banked.
TRANSFER_HARD_COST = 4.0

# Separate from the hard rule above: a *soft*, forward-looking opportunity
# cost on every transfer used this round, scaled by how many free transfers
# are left going in. Spending your last banked transfer is more expensive
# in expectation than spending one of five, because it leaves you with
# fewer in reserve to react to next week's news before the hard -4 cost
# kicks in. Linear between the two anchor points requested: 0 at 5 free
# transfers, -1.5 (midpoint of "-1 to -2") at 1; extended down to 0 free
# transfers at the same slope for a single consistent formula.
SOFT_PENALTY_AT_ONE = -1.5
SOFT_PENALTY_FREE_TRANSFERS_ANCHOR = 5

# --- Bench value ---
# A bench player only ever scores real points via FPL's auto-substitution
# rule: a STARTER who gets exactly 0 minutes (not just a bad game) gets
# swapped for the highest-priority bench player who (a) actually played and
# (b) keeps the resulting formation legal. So a bench slot's expected value
# is roughly P(this slot actually gets used) x that player's own quality —
# never the flat "0.5 min * predicted_points" you might guess from just
# reading the objective term below.
#
# BENCH_STARTER_BLANK_RATE is measured directly, not assumed: among rows in
# the 25-26 archive where a player had started literally every one of their
# last 5 games (trail_starts5 == 1.0 — about as "nailed on" as this pipeline
# can call anyone), 7.6% of them still had a genuine 0-minute week right
# after (injury, red card, unexpected rest). That's the single-player base
# rate everything else here is built from.
#
# From that rate, the *unrestricted* probability that at least 1/2/3 of an
# 11-man starting XI blank in a given week (treating starters as roughly
# independent, binomial(11, 0.076)) comes out to 58% / 20% / 4.6% — but a
# bench slot can only legally cover a blank in roughly its own position
# group, not any of the 11, which cuts those figures down a lot (a rough
# "~4 players typically share a given outfield position in the XI" discount
# brings 58/20/4.6 down to ~14/5/~1) — landing almost exactly on a plain,
# hand-picked 12%/5%/2% guess for bench ranks 1/2/3. Good enough agreement
# that there's no real case for a fancier position-and-formation-exact
# model here — see BENCH_RANK_WEIGHTS below for what's actually used and
# why the three ranks get collapsed to one shared weight in the objective.
BENCH_STARTER_BLANK_RATE = 0.076

# The reserve keeper has no rank ambiguity — a squad always carries exactly
# one bench keeper, and they only ever cover the single starting keeper
# blanking — so this uses a measured single-event rate directly rather than
# a rank-decayed guess. NOT the same BENCH_STARTER_BLANK_RATE as above,
# though — that pooled rate is dominated by outfielders, who really do get
# rotated/rested/tactically benched in a way keepers essentially never are
# mid-season. Measured for GKs specifically instead, split by time of
# season since that turned out to matter a lot: 11.1% in the GW30-38
# run-in (dead-rubber rest once safety/relegation is settled) vs. 4.4% (±2%
# at 95% CI, n=436) in GW1-29 in-season play — the number that actually
# describes the gameweeks this pipeline ever predicts for, since it never
# runs against a genuinely dead rubber. Still not "basically 0": real if
# rare events (an untelegraphed injury, a red card, an illness) do show up
# ~1 in 23 reliable-starter-keeper weeks even in-season.
#
# Worth being precise about what this is actually measuring: bench value
# only matters for blanks you had no chance to react to — a foreseeable one
# (an announced injury, a suspension from a prior red card) should really
# be handled by a transfer, not bench depth, and counting it here overstates
# the bench's true insurance value. Checked for this directly rather than
# assuming it doesn't matter: of the 19 GK blank events behind the 4.4%
# figure, zero were preceded by a red card in their prior match, and their
# transfers_out-as-%-of-ownership at the origin week (a proxy for "had the
# news already broken by the deadline") looks statistically indistinguishable
# from non-blanking reliable-starter GKs at the same time (medians 4.4% vs
# 3.1%, no outlier spike). Neither available signal found a meaningful
# foreseeable component to strip out, so this stays at the raw measured
# rate — but the historical archive has no preserved injury-status field to
# check directly, and n=19 is thin, so "found nothing to subtract" is not
# strong evidence that nothing should be subtracted. Revisit if a better
# data source for this ever becomes available.
BENCH_GK_WEIGHT = 0.044

# Bench ranks 1/2/3 (outfield): the *identity* of which selected bench
# player ends up ranked 1st/2nd/3rd (i.e. which one the auto-sub algorithm
# tries first) depends on which players get selected — not knowable before
# the solve — so assigning these three weights to specific candidates would
# need real sorting/assignment machinery inside the MILP for a small
# correction term that doesn't warrant it. Using the plain average of the
# three ranks on every outfield bench slot is the correct expected-value
# treatment when the rank-to-player assignment isn't resolved: unbiased in
# aggregate, at the cost of not rewarding a specific bench slot for being
# "the good one" over the other two.
BENCH_RANK_WEIGHTS = (0.12, 0.05, 0.02)
BENCH_OUTFIELD_WEIGHT = sum(BENCH_RANK_WEIGHTS) / len(BENCH_RANK_WEIGHTS)


def soft_transfer_penalty_rate(free_transfers: int) -> float:
    free_transfers = max(0, min(SOFT_PENALTY_FREE_TRANSFERS_ANCHOR, free_transfers))
    slope = SOFT_PENALTY_AT_ONE / (SOFT_PENALTY_FREE_TRANSFERS_ANCHOR - 1)  # per free transfer below 5
    return slope * (SOFT_PENALTY_FREE_TRANSFERS_ANCHOR - free_transfers)


def load_squad_rules() -> dict:
    gs = load_raw("fpl_live_game_settings").iloc[0]
    positions = load_raw("fpl_live_positions").set_index("singular_name_short")
    return {
        "squad_size": int(gs["squad_squadsize"]),
        "starting_size": int(gs["squad_squadplay"]),
        "budget": float(gs["squad_total_spend"]) / 10,
        "club_limit": int(gs["squad_team_limit"]),
        "squad_select": positions["squad_select"].astype(int).to_dict(),
        "min_play": positions["squad_min_play"].astype(int).to_dict(),
        "max_play": positions["squad_max_play"].astype(int).to_dict(),
    }


def check_transfer_timing(pool: pd.DataFrame, out_code: int, in_code: int, n_near_term_weeks: int = 2) -> dict:
    """Does this specific OUT/IN swap actually hold up on the *immediate*
    fixture(s), or does it only win because of the full discounted-horizon
    aggregate select_squad optimized over?

    Motivating case (see conversation): a player can have a genuinely great
    next fixture but a rough run after it — the aggregate correctly
    recommends moving on from them for the *horizon*, but that can silently
    paper over the fact that keeping them one more week, then transferring
    later, would have captured that good fixture too. This doesn't change
    what select_squad recommends (that would require a real multi-period,
    multi-week transfer-timing optimization, which — deliberately — this
    project isn't building: every week gets re-run fresh with new data
    anyway, so planning several weeks of transfers in advance mostly
    duplicates work next week's run will redo better). It just surfaces the
    near-term picture explicitly so a transfer that only wins "on average
    over 4 weeks" doesn't get treated the same as one that wins every week.
    """
    out_row = pool.loc[pool["code"] == out_code].iloc[0]
    in_row = pool.loc[pool["code"] == in_code].iloc[0]
    gw_cols = sorted(c for c in pool.columns if c.startswith("predicted_points_gw"))
    near_cols = gw_cols[:n_near_term_weeks]

    out_gw1, in_gw1 = float(out_row[gw_cols[0]]), float(in_row[gw_cols[0]])
    out_near, in_near = float(out_row[near_cols].sum()), float(in_row[near_cols].sum())

    return {
        "out_player": out_row["web_name"], "in_player": in_row["web_name"],
        "out_gw1": out_gw1, "in_gw1": in_gw1, "gw1_favors_keeping": out_gw1 > in_gw1,
        "out_near_term": out_near, "in_near_term": in_near, "near_term_favors_keeping": out_near > in_near,
        "out_full_aggregate": float(out_row["predicted_points"]), "in_full_aggregate": float(in_row["predicted_points"]),
    }


def analyze_transfer_timing(pool: pd.DataFrame, transferred_out: pd.DataFrame, transferred_in: pd.DataFrame) -> list[dict]:
    """Run check_transfer_timing on every suggested swap, pairing OUT/IN by
    position (sorted by predicted_points so multiple same-position swaps
    pair sensibly rather than arbitrarily)."""
    results = []
    for pos in transferred_out["position_name"].unique():
        outs_pos = transferred_out[transferred_out["position_name"] == pos].sort_values(
            "predicted_points", ascending=False
        )
        ins_pos = transferred_in[transferred_in["position_name"] == pos].sort_values(
            "predicted_points", ascending=False
        )
        for out_code, in_code in zip(outs_pos["code"], ins_pos["code"]):
            results.append(check_transfer_timing(pool, out_code, in_code))
    return results


def select_squad(
    pool: pd.DataFrame,
    rules: dict | None = None,
    current_squad_codes: set | None = None,
    free_transfers: int | None = None,
) -> dict:
    """pool must have: code, web_name, team_name, position_name, price,
    predicted_points — one row per eligible candidate. Returns the
    optimal 15-man squad, starting XI, bench order, and captain/vice.

    current_squad_codes + free_transfers activate the transfer-penalty
    terms (see module docstring / TRANSFER_HARD_COST /
    soft_transfer_penalty_rate) — pass both to re-optimize an existing
    squad; leave both None to build from scratch with no transfer cost at
    all, which is what an initial/opening squad actually costs in FPL.
    """
    rules = rules or load_squad_rules()
    pool = pool.reset_index(drop=True)
    idx = pool.index.tolist()

    prob = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)
    squad = pulp.LpVariable.dicts("squad", idx, cat="Binary")
    starter = pulp.LpVariable.dicts("starter", idx, cat="Binary")
    captain = pulp.LpVariable.dicts("captain", idx, cat="Binary")

    pred = pool["predicted_points"]
    # captain's points count twice: once as a starter, once again as captain
    objective = pulp.lpSum((starter[i] + captain[i]) * pred[i] for i in idx)

    # Bench players get a real expected-value term now, not just a tie-break
    # epsilon (see BENCH_STARTER_BLANK_RATE / BENCH_GK_WEIGHT /
    # BENCH_OUTFIELD_WEIGHT above for where these numbers come from and why
    # a bench player's true expected contribution is nowhere near their
    # face-value predicted_points). Deliberately NOT capped to be "too small
    # to affect the starting XI" the way the old placeholder was — a real
    # bench player's value can legitimately be worth trading a marginal
    # starting-XI swing for, and that's exactly the trade-off this term is
    # meant to let the solver make.
    is_gk = (pool["position_name"] == "GKP").values
    bench_weight = pd.Series(
        [BENCH_GK_WEIGHT if g else BENCH_OUTFIELD_WEIGHT for g in is_gk], index=pool.index
    )
    objective += pulp.lpSum((squad[i] - starter[i]) * pred[i] * bench_weight[i] for i in idx)

    transfers_used_expr = None
    if current_squad_codes is not None and free_transfers is not None:
        was_in_squad = pool["code"].isin(current_squad_codes)
        # transfers in == transfers out, since squad size is fixed — count
        # new faces only, no separate "transfer_out" variable needed.
        transfers_used_expr = pulp.lpSum(
            squad[i] for i in idx if not was_in_squad.iloc[i]
        )
        soft_rate = soft_transfer_penalty_rate(free_transfers)
        objective += soft_rate * transfers_used_expr

        excess = pulp.LpVariable("excess_transfers", lowBound=0)
        prob += excess >= transfers_used_expr - free_transfers
        objective -= TRANSFER_HARD_COST * excess

        logger.info(
            "transfer penalty active: %d free transfers -> soft rate %.3f/transfer, "
            "hard cost %.1f/transfer beyond the free allowance",
            free_transfers, soft_rate, TRANSFER_HARD_COST,
        )

    prob += objective, "expected_points_net_of_transfer_cost"

    prob += pulp.lpSum(squad[i] for i in idx) == rules["squad_size"]
    prob += pulp.lpSum(squad[i] * pool.loc[i, "price"] for i in idx) <= rules["budget"] + 1e-6
    prob += pulp.lpSum(starter[i] for i in idx) == rules["starting_size"]

    for i in idx:
        prob += starter[i] <= squad[i]
        prob += captain[i] <= starter[i]
    prob += pulp.lpSum(captain[i] for i in idx) == 1

    for pos, n_required in rules["squad_select"].items():
        pos_idx = pool.index[pool["position_name"] == pos].tolist()
        prob += pulp.lpSum(squad[i] for i in pos_idx) == n_required
        prob += pulp.lpSum(starter[i] for i in pos_idx) >= rules["min_play"][pos]
        prob += pulp.lpSum(starter[i] for i in pos_idx) <= rules["max_play"][pos]

    for club in pool["team_name"].dropna().unique():
        club_idx = pool.index[pool["team_name"] == club].tolist()
        prob += pulp.lpSum(squad[i] for i in club_idx) <= rules["club_limit"]

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"solver did not find an optimal solution: {pulp.LpStatus[status]}")

    pool["in_squad"] = [bool(round(squad[i].value())) for i in idx]
    pool["is_starter"] = [bool(round(starter[i].value())) for i in idx]
    pool["is_captain"] = [bool(round(captain[i].value())) for i in idx]

    squad_df = pool[pool["in_squad"]].copy()
    starters = squad_df[squad_df["is_starter"]].sort_values("predicted_points", ascending=False)
    bench = squad_df[~squad_df["is_starter"]].sort_values("predicted_points", ascending=False)
    captain_row = squad_df[squad_df["is_captain"]].iloc[0]
    vice_row = starters[~starters["is_captain"]].iloc[0]  # highest-EV starter who isn't captain

    starting_points = starters["predicted_points"].sum() + captain_row["predicted_points"]

    transfer_info = None
    if transfers_used_expr is not None:
        n_transfers = int(round(pulp.value(transfers_used_expr)))
        excess_n = max(0, n_transfers - free_transfers)
        soft_cost = soft_transfer_penalty_rate(free_transfers) * n_transfers
        hard_cost = TRANSFER_HARD_COST * excess_n
        transferred_in = squad_df[~squad_df["code"].isin(current_squad_codes)]
        transferred_out = pool[
            pool["code"].isin(current_squad_codes) & ~pool["code"].isin(squad_df["code"])
        ]
        transfer_info = {
            "transfers_used": n_transfers,
            "free_transfers": free_transfers,
            "excess_transfers": excess_n,
            "soft_cost": float(soft_cost),
            "hard_cost": float(hard_cost),
            "total_transfer_cost": float(soft_cost + hard_cost),
            "transferred_in": transferred_in["web_name"].tolist(),
            "timing_check": analyze_transfer_timing(pool, transferred_out, transferred_in),
        }

    logger.info(
        "optimal squad: %d players, £%.1fm / £%.1fm, starting XI predicted %.2f pts (captain: %s)",
        len(squad_df), squad_df["price"].sum(), rules["budget"], starting_points, captain_row["web_name"],
    )

    return {
        "squad": squad_df,
        "starters": starters,
        "bench": bench,
        "captain": captain_row,
        "vice_captain": vice_row,
        "total_cost": float(squad_df["price"].sum()),
        "budget": rules["budget"],
        "starting_predicted_points": float(starting_points),
        "transfer_info": transfer_info,
        "rules": rules,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from pipeline.predict_next_gw import predict_multi_gw

    # Initial squad build: no current_squad_codes/free_transfers passed, so
    # the transfer-penalty terms don't activate — an opening squad has no
    # prior squad to transfer from and costs nothing to assemble in real
    # FPL. Pass both (e.g. select_squad(pool, current_squad_codes={...},
    # free_transfers=2)) to re-optimize a later gameweek's transfer(s).
    pool = predict_multi_gw(horizons=(1, 2, 3, 4), discount=0.95)
    result = select_squad(pool)
    cols = ["web_name", "team_name", "position_name", "price", "predicted_points"]
    print("=== STARTING XI (4-gameweek discounted) ===")
    print(result["starters"][cols].to_string(index=False))
    print("\n=== BENCH ===")
    print(result["bench"][cols].to_string(index=False))
    print(f"\nCaptain: {result['captain']['web_name']}  Vice: {result['vice_captain']['web_name']}")
    print(f"Cost: £{result['total_cost']:.1f}m / £{result['budget']:.1f}m")
    print(f"Starting XI predicted points (with captain bonus): {result['starting_predicted_points']:.2f}")
