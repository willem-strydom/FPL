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

    # Bench players contribute nothing to the primary objective (design
    # doc's EV-maximization is about the starting XI), which leaves the
    # solver genuinely indifferent among same-priced bench options — it
    # will happily seat a mediocre bench forward next to a much better one
    # at the identical price, since nothing in the objective says
    # otherwise, and CBC returns *a* vertex, not "the" vertex, among tied
    # solutions. A small epsilon reward for bench predicted_points breaks
    # that tie in favor of the better bench player without ever being
    # large enough to justify a worse starting XI to get it — it's sized
    # far below the smallest realistic single-player starting-XI swing.
    BENCH_EPSILON = 1e-3
    objective += BENCH_EPSILON * pulp.lpSum((squad[i] - starter[i]) * pred[i] for i in idx)

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
        transfer_info = {
            "transfers_used": n_transfers,
            "free_transfers": free_transfers,
            "excess_transfers": excess_n,
            "soft_cost": float(soft_cost),
            "hard_cost": float(hard_cost),
            "total_transfer_cost": float(soft_cost + hard_cost),
            "transferred_in": squad_df.loc[~squad_df["code"].isin(current_squad_codes), "web_name"].tolist(),
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
