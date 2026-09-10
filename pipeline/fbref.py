"""FBref ingestion — currently blocked, documented rather than stubbed silently.

DesignDoc.md lists FBref as a historical xG/xA source. In practice fbref.com
serves a Cloudflare JS challenge ("Just a moment...") to any non-browser
request, including a plain `requests.get` with a realistic browser
User-Agent (verified: HTTP 403, `noindex,nofollow` challenge page, even
`/robots.txt` is gated). Defeating that challenge would mean automating
around bot detection, which this pipeline won't do.

Net effect on the model, per DesignDoc.md's own framing: Understat was
already spec'd as "alternative shot-based xG/xA (optional, cross-check vs
FBref)" — so Understat (pipeline/understat.py) fills the xG/xA role, and
FPL's own bootstrap/history data already includes Opta-sourced
expected_goals/expected_assists per gameweek as a second, independent
cross-check. Between those two, the model has two independent xG/xA
estimates without needing FBref specifically.

This module is left in place as a seam: if the user obtains an authorized
path to FBref data (their own manual export, a licensed API, etc.), a
fetch_* function can be dropped in here and wired into build_dataset.py
without touching the rest of the pipeline.
"""
import logging

logger = logging.getLogger(__name__)

STATUS = "blocked"
REASON = (
    "fbref.com returns HTTP 403 behind a Cloudflare JS challenge for "
    "non-browser requests; not bypassed by design."
)


def ingest() -> None:
    logger.warning("FBref ingestion skipped: %s", REASON)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
