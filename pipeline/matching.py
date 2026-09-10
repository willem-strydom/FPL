"""Name/team normalization for joining Understat players onto FPL players.

There's no shared ID between FPL and Understat, so we match on normalized
player name + club. Diacritics, punctuation, and club-naming differences
("Man City" vs "Manchester City") are the main sources of mismatch.
"""
import re
import unicodedata

from pipeline.config import TEAM_NAME_OVERRIDES


# Letters NFKD doesn't decompose into a base + combining accent (they're
# distinct letters, not accented variants), so ascii-encode would silently
# drop them instead of transliterating. Handle them explicitly first.
_NON_DECOMPOSING = str.maketrans(
    {"Ø": "O", "ø": "o", "Æ": "AE", "æ": "ae", "Þ": "Th", "þ": "th",
     "Đ": "D", "đ": "d", "Ł": "L", "ł": "l", "ß": "ss"}
)


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.translate(_NON_DECOMPOSING)
    stripped = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    stripped = re.sub(r"[^a-zA-Z0-9 ]", " ", stripped).lower()
    return re.sub(r"\s+", " ", stripped).strip()


def normalize_team(name: str) -> str:
    base = normalize_name(name)
    return TEAM_NAME_OVERRIDES.get(base, base)
