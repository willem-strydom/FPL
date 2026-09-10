"""Cold-start player exclusion, per DesignDoc.md "Scope cut" section.

Transfers from other leagues, promoted-team players, and academy graduates
with no/limited PL history are excluded from the model and the optimizer's
candidate pool entirely — filtered out here, at the data stage, rather than
downstream via imputed/zero scores.

A player currently in the live squad is excluded if:
  - their club was not in last season's (25-26) top flight (promoted team), or
  - they have no record at all in the 25-26 archive (new-to-PL transfer or
    academy graduate with zero prior top-flight minutes), or
  - they have a record but fall under the minimum starts/minutes bar
    (limited PL history — technically "in the league" but not enough of a
    track record for the quality model to learn from).

Matching live players to their 25-26 record uses FPL's `code` field, which
is stable across seasons (unlike `id`, which gets reassigned).
"""
import logging

import pandas as pd

from pipeline.config import COLD_START_MIN_MINUTES, COLD_START_MIN_STARTS
from pipeline.storage import load_raw, save_processed

logger = logging.getLogger(__name__)


def find_promoted_teams(live_teams: pd.DataFrame, hist_teams: pd.DataFrame) -> set[str]:
    return set(live_teams["name"]) - set(hist_teams["name"])


def classify_players(
    live_players: pd.DataFrame,
    live_teams: pd.DataFrame,
    hist_players_raw: pd.DataFrame,
    hist_teams: pd.DataFrame,
) -> pd.DataFrame:
    promoted = find_promoted_teams(live_teams, hist_teams)
    live_team_name = live_teams.set_index("id")["name"]

    hist_by_code = hist_players_raw.set_index("code")

    rows = []
    for p in live_players.itertuples():
        team_name = live_team_name.get(p.team, "unknown")
        if team_name in promoted:
            rows.append((p.code, p.web_name, team_name, False, "promoted_team"))
            continue

        if p.code not in hist_by_code.index:
            rows.append((p.code, p.web_name, team_name, False, "no_25_26_pl_record"))
            continue

        hist_row = hist_by_code.loc[p.code]
        # A code can map to >1 row only if duplicated upstream; guard defensively.
        if isinstance(hist_row, pd.DataFrame):
            hist_row = hist_row.iloc[0]

        starts = hist_row.get("starts", 0)
        minutes = hist_row.get("minutes", 0)
        if starts < COLD_START_MIN_STARTS or minutes < COLD_START_MIN_MINUTES:
            rows.append((p.code, p.web_name, team_name, False, "insufficient_pl_minutes"))
            continue

        rows.append((p.code, p.web_name, team_name, True, "ok"))

    result = pd.DataFrame(
        rows, columns=["code", "web_name", "team", "include", "reason"]
    )
    logger.info(
        "cold-start filter: %d/%d live players included (%s)",
        result["include"].sum(), len(result),
        dict(result.loc[~result["include"], "reason"].value_counts()),
    )
    return result


def build() -> pd.DataFrame:
    live_players = load_raw("fpl_live_players")
    live_teams = load_raw("fpl_live_teams")
    hist_players_raw = load_raw("fpl_hist_players_raw")
    hist_teams = load_raw("fpl_hist_teams")

    result = classify_players(live_players, live_teams, hist_players_raw, hist_teams)
    save_processed(result, "cold_start_exclusions")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build()
