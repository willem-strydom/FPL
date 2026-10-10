"""A manager's actual squad state, from FPL's public entry endpoints.

Squad comes from the latest gameweek's picks (the gameweek before it if that
week was a Free Hit, since Free Hit squads revert). Transfers made for the
upcoming gameweek stay hidden until its deadline, so this is the squad as of
the last deadline.
"""
import logging

import requests

from pipeline.config import REQUEST_TIMEOUT, USER_AGENT
from pipeline.storage import load_raw

logger = logging.getLogger(__name__)

API = "https://fantasy.premierleague.com/api/entry/{entry_id}/"
MAX_BANKED_TRANSFERS = 5


def _get(path: str) -> dict | list:
    resp = requests.get(path, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def free_transfers_next(events: list[dict], chips: dict[int, str]) -> int:
    """Replay the banking rule over the season: +1 per week (cap 5); using more
    than banked costs hits and resets to 1; Wildcard/Free Hit weeks keep the bank."""
    ft = None
    for e in sorted(events, key=lambda e: e["event"]):
        if ft is None:  # the entry's first gameweek is the free initial squad
            ft = 1
            continue
        used = e["event_transfers"]
        if chips.get(e["event"]) in ("wildcard", "freehit"):
            ft = min(MAX_BANKED_TRANSFERS, ft + 1)
        elif used <= ft:
            ft = min(MAX_BANKED_TRANSFERS, ft - used + 1)
        else:
            ft = 1
    return ft


def selling_price(purchase: int, now: int) -> int:
    """FPL keeps half of any rise (rounded down), in tenths of £m."""
    return now if now <= purchase else purchase + (now - purchase) // 2


def fetch_entry_state(entry_id: int) -> dict:
    base = API.format(entry_id=entry_id)
    history = _get(base + "history/")
    events = history["current"]
    chips = {c["event"]: c["name"] for c in history["chips"]}
    latest = max(e["event"] for e in events)
    squad_event = latest - 1 if chips.get(latest) == "freehit" else latest

    picks = _get(base + f"event/{squad_event}/picks/")["picks"]
    transfers = _get(base + "transfers/")
    players = load_raw("fpl_live_players").set_index("id")

    # Purchase price: latest transfer-in cost, else the player's GW1 price
    # (initial squad). Picks don't expose it without logging in.
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        bought[t["element_in"]] = t["element_in_cost"]
    gw_log = load_raw("fpl_current_gw")
    first_price = gw_log.sort_values("round").groupby("element")["value"].first()

    squad, prices = set(), {}
    for p in picks:
        el = p["element"]
        now = int(players.loc[el, "now_cost"])
        purchase = bought.get(el, int(first_price.get(el, now)))
        code = int(players.loc[el, "code"])
        squad.add(code)
        prices[code] = selling_price(purchase, now) / 10

    bank = next(e["bank"] for e in events if e["event"] == squad_event) / 10
    state = {
        "squad": squad,
        "selling_prices": prices,
        "bank": bank,
        "free_transfers": free_transfers_next(events, chips),
        "as_of_gw": squad_event,
        "chips_used": chips,
    }
    logger.info("entry %s: squad as of GW%d, bank £%.1fm, %d FT", entry_id, squad_event, bank, state["free_transfers"])
    return state
