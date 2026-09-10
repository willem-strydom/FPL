"""Validation suite for the ingested data. Run after build_dataset.py.

Checks span three levels:
  1. Raw ingestion sanity (row counts, presence) for each source.
  2. Processed table integrity (schema, nulls, ranges, duplicate keys).
  3. Cross-source / cross-cutting sanity (Understat vs FPL's own xG,
     cold-start filter behaving sensibly, season freshness).

Each check returns (passed, message, details). Nothing here raises on
failure — it's a report, not a gate — but every failure is surfaced clearly
so it's obvious what to look at before trusting the data for training.
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

from pipeline.config import DATA_DIR, HISTORY_SEASON
from pipeline.storage import load_processed, load_raw

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict = field(default_factory=dict)


def _try(name: str, fn) -> CheckResult:
    try:
        passed, message, details = fn()
        return CheckResult(name, passed, message, details)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the suite
        return CheckResult(name, False, f"check raised {exc!r}", {})


# --- raw ingestion checks -------------------------------------------------

def check_raw_row_counts() -> tuple[bool, str, dict]:
    expectations = {
        "fpl_live_players": (100, 5000),
        "fpl_live_teams": (20, 20),
        "fpl_hist_merged_gw": (20000, 40000),
        "fpl_hist_players_raw": (400, 1200),
        "fpl_hist_teams": (20, 20),
        "understat_season_players": (300, 800),
    }
    details, ok = {}, True
    for name, (lo, hi) in expectations.items():
        try:
            n = len(load_raw(name))
        except FileNotFoundError:
            details[name] = "MISSING"
            ok = False
            continue
        in_range = lo <= n <= hi
        details[name] = f"{n} rows ({'ok' if in_range else f'expected {lo}-{hi}'})"
        ok &= in_range
    return ok, "raw source row counts within expected ranges" if ok else \
        "one or more raw sources missing or outside expected row-count range", details


def check_understat_match_coverage() -> tuple[bool, str, dict]:
    matches = load_raw("understat_player_matches")
    id_map = load_raw("understat_fpl_id_map")
    matched_ids_with_logs = set(matches["understat_player_id"].astype(str)) & set(
        id_map["understat_id"].astype(str)
    )
    coverage = len(matched_ids_with_logs) / len(id_map) if len(id_map) else 0
    ok = coverage > 0.8
    return ok, f"Understat match-log coverage: {coverage:.0%} of FPL-matched players", {
        "matched_players_with_match_logs": len(matched_ids_with_logs),
        "total_matched_players": len(id_map),
        "total_match_rows": len(matches),
    }


def check_gameweek_completeness() -> tuple[bool, str, dict]:
    gw = load_raw("fpl_hist_merged_gw")
    rounds = sorted(gw["round"].unique())
    ok = rounds == list(range(1, 39))
    return ok, f"gameweeks present: {rounds[0]}-{rounds[-1]} ({len(rounds)}/38)", {
        "missing_rounds": sorted(set(range(1, 39)) - set(rounds))
    }


# --- processed table checks ------------------------------------------------

REQUIRED_COLUMNS = {
    "element", "round", "minutes", "total_points", "value", "was_home",
    "rolling_5gw_minutes", "own_team_difficulty", "understat_xG", "understat_xA",
    "expected_goals", "expected_assists",
}


def check_processed_schema() -> tuple[bool, str, dict]:
    df = load_processed("player_gameweek_2025_26")
    missing = REQUIRED_COLUMNS - set(df.columns)
    ok = not missing
    return ok, "all required columns present" if ok else f"missing columns: {missing}", {
        "n_columns": len(df.columns)
    }


def check_no_duplicate_player_fixtures() -> tuple[bool, str, dict]:
    """(element, round) is legitimately non-unique — FPL "double gameweeks"
    have a team play twice in one round (usually rearranged postponements).
    (element, fixture) is the real uniqueness constraint: one row per
    player per actual match.
    """
    df = load_processed("player_gameweek_2025_26")
    dupes = df.duplicated(subset=["element", "fixture"]).sum()
    dgw_rows = df.duplicated(subset=["element", "round"], keep=False).sum()
    ok = dupes == 0
    return ok, f"{dupes} duplicate (element, fixture) rows ({dgw_rows} double-gameweek rows are expected)", {
        "n_duplicate_fixtures": int(dupes), "n_double_gameweek_rows": int(dgw_rows)
    }


def check_core_field_nulls() -> tuple[bool, str, dict]:
    df = load_processed("player_gameweek_2025_26")
    core_fields = ["element", "round", "minutes", "total_points", "value", "kickoff_time"]
    null_counts = {c: int(df[c].isna().sum()) for c in core_fields}
    ok = all(v == 0 for v in null_counts.values())
    return ok, "no nulls in core fields" if ok else f"unexpected nulls: {null_counts}", null_counts


def check_value_ranges() -> tuple[bool, str, dict]:
    df = load_processed("player_gameweek_2025_26")
    problems = {}
    if not df["minutes"].between(0, 130).all():
        problems["minutes"] = "outside [0, 130]"
    if not df["round"].between(1, 38).all():
        problems["round"] = "outside [1, 38]"
    if not df["value"].gt(0).all():
        problems["value"] = "non-positive price found"
    if not df["own_team_difficulty"].dropna().between(1, 5).all():
        problems["own_team_difficulty"] = "outside [1, 5]"
    if not df["rolling_5gw_minutes"].dropna().between(0, 130).all():
        problems["rolling_5gw_minutes"] = "outside [0, 130]"
    ok = not problems
    return ok, "all value ranges sane" if ok else f"range violations: {problems}", problems


def check_rolling_minutes_no_leakage() -> tuple[bool, str, dict]:
    """Spot-check: a player's rolling_5gw_minutes at round R must be computable
    purely from rounds < R (never equal to a same-round function of itself).
    We verify this structurally: round-1 rows must always be NaN (no prior
    gameweek exists yet to average).
    """
    df = load_processed("player_gameweek_2025_26")
    round1 = df[df["round"] == 1]
    non_null_round1 = round1["rolling_5gw_minutes"].notna().sum()
    ok = non_null_round1 == 0
    return ok, (
        "round-1 rows correctly have no rolling-minutes value (nothing to average yet)"
        if ok else f"{non_null_round1} round-1 rows have a non-null rolling average (leakage)"
    ), {"non_null_round1": int(non_null_round1)}


# --- cross-source / cross-cutting checks -----------------------------------

def check_xg_source_agreement() -> tuple[bool, str, dict]:
    """FPL's native expected_goals and Understat's xG are independent
    providers (Opta vs StatsBomb-style shot models) — they shouldn't match
    exactly, but should be positively correlated where both exist.
    """
    df = load_processed("player_gameweek_2025_26")
    both = df.dropna(subset=["expected_goals", "understat_xG"])
    both = both[both["minutes"] > 0]
    if len(both) < 50:
        return False, f"too few overlapping rows to check ({len(both)})", {"n": len(both)}
    corr = both["expected_goals"].corr(both["understat_xG"])
    ok = corr > 0.6
    return ok, f"FPL vs Understat expected-goals correlation: {corr:.2f} (n={len(both)})", {
        "correlation": round(float(corr), 3), "n_overlap": len(both)
    }


def check_cold_start_filter_sanity() -> tuple[bool, str, dict]:
    df = load_processed("cold_start_exclusions")
    included = df["include"].sum()
    total = len(df)
    frac = included / total if total else 0
    ok = 0.3 < frac < 0.9
    reasons = dict(df.loc[~df["include"], "reason"].value_counts())
    return ok, f"{included}/{total} live players pass the cold-start filter ({frac:.0%})", {
        "exclusion_reasons": reasons
    }


def check_season_freshness() -> tuple[bool, str, dict]:
    """DesignDoc.md: only train on the past season. Confirm the historical
    archive is the season we configured, and that it's distinct from
    whatever season the live API currently serves.
    """
    from pipeline.fpl_live import current_season_label

    events = load_raw("fpl_live_events")
    live_season = current_season_label(events)
    gw = load_raw("fpl_hist_merged_gw")
    kickoffs = pd.to_datetime(gw["kickoff_time"])
    hist_years = {kickoffs.min().year, kickoffs.max().year}
    expected_start_year = int(HISTORY_SEASON.split("-")[0])
    ok = expected_start_year in hist_years and live_season != HISTORY_SEASON
    return ok, (
        f"training data spans {sorted(hist_years)}, configured as {HISTORY_SEASON}; "
        f"live API is on {live_season}"
    ), {"hist_years": sorted(hist_years), "live_season": live_season}


CHECKS = [
    check_raw_row_counts,
    check_gameweek_completeness,
    check_understat_match_coverage,
    check_processed_schema,
    check_no_duplicate_player_fixtures,
    check_core_field_nulls,
    check_value_ranges,
    check_rolling_minutes_no_leakage,
    check_xg_source_agreement,
    check_cold_start_filter_sanity,
    check_season_freshness,
]


def run() -> list[CheckResult]:
    results = [_try(fn.__name__, fn) for fn in CHECKS]
    n_pass = sum(r.passed for r in results)
    logger.info("validation: %d/%d checks passed", n_pass, len(results))
    for r in results:
        level = logging.INFO if r.passed else logging.WARNING
        logger.log(level, "[%s] %s: %s", "PASS" if r.passed else "FAIL", r.name, r.message)

    _write_report(results)
    return results


def _write_report(results: list[CheckResult]) -> None:
    lines = ["# Data Validation Report", ""]
    n_pass = sum(r.passed for r in results)
    lines.append(f"**{n_pass}/{len(results)} checks passed**")
    lines.append("")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"## [{status}] {r.name}")
        lines.append(r.message)
        if r.details:
            for k, v in r.details.items():
                lines.append(f"- {k}: {v}")
        lines.append("")
    path = DATA_DIR / "validation_report.md"
    path.write_text("\n".join(lines))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
