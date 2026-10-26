# Squad History — 2026-27 Season

A record of the actual squad shipped each gameweek and the transfers made
between them. The live source of truth is now the FPL API for entry
7592422 (`pipeline/fpl_entry.py`, used by `python -m pipeline.recommend`);
this file is the human-readable log, refreshed from the API.

Confidence key: **CONFIRMED (API)** — read from the FPL entry endpoints.
**CONFIRMED** — willem stated it directly. **PENDING** — recommended, not
yet visible (upcoming-gameweek transfers stay hidden until the deadline).

---

## Transfer log

| GW | Deadline | Out | In | FT used | Status |
|----|----------|-----|-----|---------|--------|
| 1 | 2026-08-21 | — (initial squad) | — | — | CONFIRMED |
| 2 | 2026-08-28 | none | none | 0 | CONFIRMED (willem: "another gameweek went by and I didn't make any transfers") |
| 3 | 2026-09-04 | none | none | 0 | CONFIRMED (same statement covers GW2 and GW3) |
| 4 | 2026-09-12 | Pedro Porro, O'Nien, E.Le Fée | Calafiori, Thomas, Bobb | 3 | CONFIRMED (API) — 3 transfers, no hit |
| 5 | 2026-09-18 | none | none | 0 | CONFIRMED (API) — the recommended De Cuyper → Hall was not made; FT rolled |
| 6 | 2026-10-10 | Szoboszlai, Anderson | Ødegaard, Rudoni | 2 | CONFIRMED (API) — the planner's recommendation, no hit |
| 7 | 2026-10-17 | none | none | 0 | CONFIRMED (API) — FT rolled (the planner had Ødegaard → Semenyo as a near-zero-value move); captain Haaland |
| 8 | 2026-10-23 | Dubravka | Kinsky | 1 | CONFIRMED (API) — the planner's recommendation; second FT rolled |

---

## Squad as of GW1 (initial, confirmed)

**Starting XI**: Haaland (C), B.Fernandes (VC), Mbeumo, Gabriel, Szoboszlai, De Cuyper, E.Le Fée, Anderson, Pedro Porro, Georginio, Lammens
**Bench**: O'Nien, van Ewijk, Dubravka, Mheuka
**Cost**: £100.0m

## Squad as of GW5 (CONFIRMED via API)

**Starting XI**: Haaland (C), Lammens (VC), Gabriel, De Cuyper, Calafiori, B.Fernandes, Mbeumo, Szoboszlai, Bobb, Anderson, Georginio
**Bench**: Dubravka, Thomas, van Ewijk, Mheuka
**Bank**: £0.0m | **Selling value**: £99.3m (De Cuyper has risen £0.5m but sells for £4.7m)

## Squad as of GW6 (CONFIRMED via API)

**Starting XI**: Haaland (C), Lammens (VC), Gabriel, De Cuyper, Calafiori, B.Fernandes, Mbeumo, Ødegaard, Bobb, Rudoni, Georginio
**Bench**: Dubravka, Thomas, van Ewijk, Mheuka
**Bank**: £1.5m | **Team value**: £99.7m

---

## Maintenance notes

- Update this file whenever willem confirms a real transfer was made — move
  the relevant PENDING row to CONFIRMED, add the next gameweek's row, and
  refresh the "squad as of" section.
- If a recommendation is *not* taken (a different transfer made, or none
  at all), record what actually happened, not what was recommended — this
  file tracks reality, not the model's suggestions.
- The corresponding Claude memory file (`fpl-squad-state.md`) should point
  here as the source of truth rather than duplicating this table, to avoid
  the two drifting apart.
