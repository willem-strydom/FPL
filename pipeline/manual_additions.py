"""SUPERSEDED — no longer used in the production driver. Kept for reference
and for the (now rare) case of a cold-start-excluded player with real
transfer-market pedigree but zero real current-season minutes yet, where
there's genuinely nothing else to go on.

Once a cold-start-excluded player has actually played real Premier League
minutes this season, pipeline.predict_next_gw.graduated_new_arrivals lets
them into the normal candidate pool and scores them with the same GLM as
everyone else, off their real (if thin) current-season trailing stats —
not this module's foreign-league discount table. That's a strictly better
signal the moment it exists (see the Sangaré/Anderson comparison in
conversation: this module priced Sangaré on his old Ligue 1 numbers while
ignoring 18 points and 2 assists he'd already put up in the Premier
League), so this module no longer runs as part of a normal prediction/
optimizer pass.

Original docstring, still accurate for what this module itself does:

A short, explicit, opt-in list of newly-arrived players the cold-start
filter correctly excludes from the GLM — no 25-26 PL minutes, so there's
no trustworthy way to model their scoring rate — but who the user wants
visible to the optimizer anyway, priced in with a documented discount
for the jump to the Premier League instead of a real prediction.

This is deliberately NOT a general mechanism. DesignDoc.md explicitly
named "hardcoded league-strength adjustment factors" as a complexity to
avoid, and that reasoning still holds: there's no systematic pipeline here
for cross-league data, no principled per-league calibration, nothing that
generalizes. This is a bounded carve-out for a short list of named
players, each with real (web-verified) prior-season stats and a plain,
auditable discount table — never fed to the GLM, never retrained on, and
removed the moment a player has real PL minutes (at which point they age
out of the cold-start filter naturally and this module should drop them).

Methodology: prior-season goals/assists convert to a rough points-per-90
using standard FPL scoring values, then two multiplicative discounts are
applied:

  - League tier: how close the source league plays to the Premier League.
  - Position: how well output at that position tends to travel (goals are
    the most system/league-dependent output there is; last line of
    defense positions less so).

Both tables are judgment calls, not fitted to data — there is no dataset
of "cross-league transfer outcomes" this pipeline has access to. Treat the
resulting numbers as a deliberately conservative placeholder, not a
prediction with the same standing as the GLM's output.
"""
import logging

import pandas as pd

from pipeline.storage import load_raw

logger = logging.getLogger(__name__)

APPEARANCE_POINTS = 2.0  # nominal per-90 return for simply being on the pitch, before goals/assists
GOAL_VALUE_BY_POSITION = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}  # standard FPL scoring
ASSIST_VALUE = 3

# Judgment calls, not fitted — see module docstring.
LEAGUE_TIER_DISCOUNT = {
    "big5": 0.80,             # Bundesliga, La Liga, Serie A, Ligue 1
    "strong_secondary": 0.60,  # Eredivisie, Primeira Liga, Belgian Pro League, Championship
    "other": 0.40,
}
POSITION_ADAPTATION_DISCOUNT = {"GKP": 0.95, "DEF": 0.95, "MID": 0.90, "FWD": 0.85}

# A confirmed RotoWire starter uses the real p(start) like everyone else.
# Absent that, a highly-touted, expensive new arrival still isn't a "bench
# fodder" case (BENCH_PROB=0.12 means "we positively know they're not
# starting") — it's genuine uncertainty, priced separately and more
# generously, but still well short of assuming they walk into the side.
UNCONFIRMED_NEW_SIGNING_PROB = 0.30

# code: FPL's stable player id (see fpl_live_players) — looked up once here
# so the rest of the record can be plain, checkable prior-season stats.
# Sources verified via web search (see conversation), not remembered:
#   Vuskovic: 2024-25 on loan at Westerlo (Belgian Pro League), 35 apps, 7 G, 3 A.
#   Sangare: 2025-26 at RC Lens (Ligue 1), 33 apps, 3 G, 3 A.
#   Toure: 2025-26 at TSG Hoffenheim (Bundesliga), 29 starts, 5 G, 9 A.
MANUAL_PLAYERS = [
    {
        "code": 610799, "web_name": "Vuskovic", "prior_league_tier": "strong_secondary",
        "prior_league": "Belgian Pro League (on loan at Westerlo)",
        "prior_apps": 35, "prior_goals": 7, "prior_assists": 3,
    },
    {
        "code": 513545, "web_name": "Sangaré", "prior_league_tier": "big5",
        "prior_league": "Ligue 1 (RC Lens)",
        "prior_apps": 33, "prior_goals": 3, "prior_assists": 3,
    },
    {
        "code": 611665, "web_name": "Touré", "prior_league_tier": "big5",
        "prior_league": "Bundesliga (TSG Hoffenheim)",
        "prior_apps": 29, "prior_goals": 5, "prior_assists": 9,
    },
]

POSITION_NAME = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def build_manual_candidates(
    horizons: tuple[int, ...] = (1, 2, 3, 4),
    discount: float = 0.95,
    lineup_data: tuple | None = None,
) -> pd.DataFrame:
    live_players = load_raw("fpl_live_players")
    live_teams = load_raw("fpl_live_teams")
    lp = live_players.copy()
    lp["price"] = lp["now_cost"] / 10
    lp["position_name"] = lp["element_type"].map(POSITION_NAME)
    lp["team_name"] = lp["team"].map(live_teams.set_index("id")["name"])
    lp["full_name"] = lp["first_name"] + " " + lp["second_name"]

    if lineup_data is not None:
        probs, _ = lineup_data
    else:
        from pipeline.lineups import get_start_probabilities

        probs, _ = get_start_probabilities(live_players, live_teams)
    lineup_p = probs.set_index("code")["p_plays_60_lineup"] if not probs.empty else pd.Series(dtype=float)

    weight_sum = sum(discount ** (h - 1) for h in horizons)
    rows = []
    for entry in MANUAL_PLAYERS:
        player = lp[lp["code"] == entry["code"]]
        if player.empty:
            logger.warning("manual addition code=%s not found in live_players — skipped", entry["code"])
            continue
        player = player.iloc[0]
        position = player["position_name"]

        goals_per90 = entry["prior_goals"] / entry["prior_apps"]
        assists_per90 = entry["prior_assists"] / entry["prior_apps"]
        raw_quality_per90 = (
            APPEARANCE_POINTS
            + goals_per90 * GOAL_VALUE_BY_POSITION[position]
            + assists_per90 * ASSIST_VALUE
        )
        combined_discount = (
            LEAGUE_TIER_DISCOUNT[entry["prior_league_tier"]] * POSITION_ADAPTATION_DISCOUNT[position]
        )
        discounted_quality = raw_quality_per90 * combined_discount

        if entry["code"] in lineup_p.index:
            p_start = float(lineup_p.loc[entry["code"]])
            p_source = "lineup_starter"
        else:
            p_start = UNCONFIRMED_NEW_SIGNING_PROB
            p_source = "unconfirmed_new_signing"

        predicted_points = (p_start * discounted_quality + (1 - p_start) * 0.5) * weight_sum

        rows.append({
            "code": entry["code"],
            "web_name": player["web_name"],
            "full_name": player["full_name"],
            "team_name": player["team_name"],
            "position_name": position,
            "price": player["price"],
            "selected_by_percent": player.get("selected_by_percent"),
            "predicted_points": predicted_points,
            "p_plays_60": p_start,
            "p_plays_60_source": p_source,
            "is_manual_addition": True,
            "prior_league": entry["prior_league"],
            "raw_quality_per90": raw_quality_per90,
            "combined_discount": combined_discount,
            "discounted_quality_per90": discounted_quality,
        })

    result = pd.DataFrame(rows)
    logger.info(
        "manual additions: %d players (%s)",
        len(result), result["web_name"].tolist() if len(result) else [],
    )
    return result


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    df = build_manual_candidates()
    pd.set_option("display.width", 160)
    print(df[[
        "web_name", "team_name", "position_name", "price", "prior_league",
        "raw_quality_per90", "combined_discount", "discounted_quality_per90",
        "p_plays_60", "p_plays_60_source", "predicted_points",
    ]].to_string(index=False))
