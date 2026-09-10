# FPL Points Prediction & Team Optimization — Implementation Report

This document summarizes what was actually built, validated, and shipped against the goals set out in [DesignDoc.md](DesignDoc.md): a points-prediction model and a mathematical-programming squad optimizer that uses it. It focuses on modeling, validation, and system design decisions — not code-level detail — and deliberately includes the dead ends and negative results that shaped the final design, since several of them are the reason the system looks the way it does.

---

## 1. Data Infrastructure

Two genuinely different data sources feed the system, kept structurally separate because they answer different questions:

- **Historical archive (2025-26 season, per-gameweek)**: ~29,750 rows across all 38 gameweeks, 841 distinct players who appeared at some point that season. This is the only data used to *train* anything — it has known outcomes to learn from. Enriched with xG/xA from FBref and Understat (used as a cross-check against each other), matched to FPL player records by name/team.
- **Live FPL API (bootstrap-static + fixtures)**: the *current* season's squad membership, prices, ownership, fitness/status flags, and official squad-building rules (budget, formation limits, per-club cap). This is the source of truth for "who's actually available to pick right now" — re-fetched fresh on every optimizer run rather than cached, since prices and injury status change frequently, especially close to a season's start.
- **RotoWire (scraped)**: predicted/confirmed starting lineups for the next upcoming gameweek. Legally scraped per RotoWire's own robots.txt/llms.txt terms — distinct from bypassing bot detection. Used only as a start-probability signal (see §7), not for any points modeling.

All raw pulls and derived tables are cached to disk as Parquet (`data/raw/`, `data/processed/`) via a small storage layer, so re-running any stage doesn't require re-fetching upstream data unless it's genuinely stale. An automated data-validation pass checks the ingested historical archive on every build — row counts within expected ranges, all 38 gameweeks present, no unintended duplicate (player, fixture) rows (818 legitimate double-gameweek rows are expected and distinguished from bugs), full Understat match-log coverage for matched players, and no nulls in core fields (minutes, points, kickoff time, etc.).

**Player identity across sources** is matched via FPL's stable `code` field (not `id`, which gets reassigned across seasons) and, where name-based matching against FBref/Understat is needed, a token-containment match with a tiebreak on combined web-name + full-name token overlap — added after an early bug where a common first name (multiple "Gabriel"s across different clubs) matched to the wrong player and silently dropped the correct one from model output.

---

## 2. Feature Engineering & Leakage-Safe Design

### Training table construction

The core training table is **horizon-as-feature**: one row per (player, origin gameweek, horizon), where horizon ∈ {1, 2, 3, 4} and the target is that player's points scored *horizon* gameweeks after the origin. This lets one model learn "points N weeks out" as a function of how far out N is, rather than training four separate models or one multi-output model — maximizing usable training rows per horizon and letting the model learn its own regression-to-the-mean pattern as horizon grows, per the original design doc's reasoning. `horizon` enters the model as a plain numeric feature, not one-hot encoded.

Every feature is computed strictly from data available as of the origin gameweek — never the target gameweek. Double gameweeks (two fixtures in one round) are summed into a single gameweek-total target, and the origin snapshot is taken after the round's last fixture so trailing features already reflect both matches.

### Trailing features

- **`trail_minutes5`**: rolling mean of minutes played over the last 5 gameweeks, *including* zero-minute weeks. This is the "will they start" signal — playing-time trend, not quality.
- **`trail_games5`**: appearance count in the last 5 gameweeks; used as an origin-eligibility gate (a player needs at least one recent appearance to get an origin row at all).
- **`trail_points5`**: rolling mean of actual points scored over the last 5 gameweeks. Used only as the "naive forecast" baseline comparator in validation, never as a model input.
- **Per-90 form rates** (points, goals, assists, BPS, ICT, FPL's own xG/xA, Understat's xG/xA): pooled over the player's last 5 *appearances* (minutes > 0 only), computed as summed-stat / summed-minutes × 90 rather than an average of per-match rates — this avoids a 2-minute cameo goal being read as a 45-points-per-90 extrapolation. Below a reliability floor (30 pooled minutes), the rate is treated as noise and the last reliable value is carried forward instead.

### Cumulative-to-date features (added during the validation work — see §5)

Two additional features track each player's **career-to-date** tenure using an *expanding* (not rolling) window: cumulative starts and cumulative minutes, strictly up to and including each row's own origin round. These are the leakage-safe, walk-forward-compatible version of the "has this player actually proven a squad role" question that the live cold-start filter answers using full-season hindsight — necessary because reusing the full-season version inside a chronological validation fold would leak future information.

### Leakage-safe cross-validation

Because the horizon-as-feature design produces near-duplicate rows (the same origin snapshot repeated 4 times with only the target fixture changing), naive row-level CV would leak: two rows from the same decision point could land on opposite sides of a train/validation split. All validation in this project uses **walk-forward, expanding-window CV grouped by (player, origin gameweek)** — every horizon-row from one decision point moves together into the same fold, and folds only ever train on gameweeks strictly before their validation window.

---

## 3. Points Model: Architecture & Selection

### The point-decomposition formula

Per the design doc, raw points are not predicted directly (which would conflate "bad player" with "good player who doesn't play"). Points are decomposed as:

```
E[points] = P(plays 60+ min) × E[points | plays 60+ min]
          + P(plays 1-59 min) × E[points | plays 1-59 min]
          + P(doesn't play)   × baseline appearance floor
```

Start probability is sourced from data (trailing minutes, later upgraded with lineup data — see §7), not modeled from scratch. The "quality" piece — E[points | plays 60+ minutes] — is where the real modeling effort went.

### GLM vs. GBM

Per the design doc's own open question, a Poisson/Negative-Binomial GLM was compared empirically against a shallow, heavily regularized GBM via the same walk-forward validation, on the 60+-minute band. Measuring Pearson dispersion on that band came out at ~1.64-1.65 — genuinely overdispersed — so Negative Binomial (NB2), not plain Poisson, was used. The GLM and GBM came out comparably accurate; the GLM was kept as the production model for being far more data-efficient and interpretable, consistent with the design doc's stated bias against complexity that doesn't earn its place with a measurable accuracy gain.

**Final GLM feature set**: horizon, trail_minutes5, trail_games5, price, target fixture difficulty, target home/away, the per-90 form rates listed in §2, position (one-hot, goalkeeper dropped as the baseline category), and two data-quality indicator flags (whether the form rate was unreliable / whether Understat coverage was missing) so the model can distinguish "genuinely low form" from "insufficient data to know." No formal feature-pruning pass was performed — the walk-forward validation didn't surface an overfitting problem that pruning would fix, so the added complexity of a selection step wasn't judged to have earned its place either.

### What originally covered the other two terms — and why that was a gap

Until partway through this project, only the 60+-minute term had a real model. The 1-59-minute and 0-minute terms were both served by a single hardcoded constant (0.5 points), never fit to any data. More importantly, **every validation number reported up to that point had only ever scored the raw 60+ GLM against rows that were already known to have reached 60+ minutes** — never the full blended formula against real, unfiltered outcomes. This was caught directly by a pointed question about what "validated" actually meant, and prompted the work in §4-5.

A genuine second GLM was then built for the 1-59-minute band. Its dispersion came out the *opposite* way from the 60+ band — Pearson dispersion ~0.34 (underdispersed), NB2's overdispersion parameter collapsed toward zero and its AIC was worse than plain Poisson's — so a plain Poisson GLM was used for that band specifically. Combined with a fitted DNP baseline (the training-data mean for the 0-minute band), this produced a proper three-term blended pipeline, validated end to end against every real outcome (0, 1-59, and 60+ minutes together) for the first time.

**Result**: the 3-term pipeline came out statistically tied with the simpler 2-term (hardcoded floor) pipeline overall — about a 0.001 MAE difference, both clearly ahead of a naive rolling-average baseline. The sub-60 GLM was genuinely good *within its own band* (roughly half the error of the rolling-average baseline there), but that gain doesn't survive being blended in at low weight against the dominant terms. Broken out by position, DEF and GK modestly favored the 3-term version, MID favored the simpler one, FWD was mixed. Given the tie, and the project's standing bias toward simplicity, the added sub-60 GLM was **not** adopted into production — see §6 for what was adopted instead.

---

## 4. Floor Calibration

With the full pipeline finally being validated honestly, the obvious next question was whether the 0.5-point hardcoded floor constant was any good. It hadn't been calibrated at all, so several approaches were tried:

- **Naive fix — replace 0.5 with the training data's actual mean points for the sub-60-minute band.** This made MAE and bias *worse* in every single validation fold. Diagnosis: the formula's `(1 - P(60+))` weighting concentrates its influence on low-probability (rotation-risk) players, whose true conditional outcome skews toward zero — but the unconditional mean is dragged upward by high-probability players who only occasionally miss the 60-minute cutoff while still picking up decent partial minutes. The naive mean describes a population the floor barely applies to.
- **A properly weighted mean** (least-squares optimal given the formula's own weighting) — still worse on MAE, though slightly better on RMSE. Points in this band are heavily right-skewed and zero-inflated (roughly 61% literal zeros), so mean and median diverge substantially, and MAE — not RMSE — is the more decision-relevant metric here.
- **A weighted median** — the true MAE-optimal single constant for this kind of weighted problem. This produced a stable value of roughly **-0.10** across every validation fold (a small negative correction, not a positive "token appearance" bonus) and delivered a real, consistent improvement: overall MAE of 1.919, against 1.981 for the hardcoded floor and 1.980 for the (more complex) 3-term pipeline, with a naive rolling-average baseline at 2.064. This edge held up across every position, every third of the season, and every ownership threshold checked.

### A second attempt that backfired, and why

A large fraction of every player ever appearing in the archive (roughly half) is a fringe squad player who essentially never gets real minutes — reserves, academy players, long-term absentees the FPL API still tracks. Measuring precisely: this contamination is mostly filtered out already by construction (a fringe player rarely reaches 60+ minutes, so that model's training set self-cleans to ~99% real players), but it concentrates heavily in the 0-minute outcome band (~33% of those training rows), which is exactly what the floor and the sub-60 rate constant are fit on.

The natural next fix — refit the floor's weighted median restricted to "established" (leakage-safe cumulative tenure) players only — was tried and made things **substantially worse**, not better: the fitted value became unstable (swinging between -0.95 and -0.39 across folds instead of the previous stable -0.10) and underperformed even the plain hardcoded floor. The reason: established players mostly carry *high* start-probability, hence *small* weight in this specific fit — restricting to them doesn't just shrink the row count, it shrinks the total weight backing the estimate by roughly 8x, leaving a noisy fit dominated by a handful of currently-out-of-form regulars. This was reverted; the deployed floor is fit on the full, unfiltered population. The same restriction, applied only to the smaller *rate* constant (P(plays partial minutes | not 60+)), was safe and mildly beneficial — that estimate doesn't have the same weight-instability problem.

This distinction — population filtering helps some parts of the pipeline and actively hurts others, depending on how sensitive each specific estimator is to sample size and weight concentration — was the main lesson of this phase, and is documented directly in the code (`two_stage_pipeline.py`) so it isn't rediscovered the hard way again.

---

## 5. Validation Methodology: From Row Averages to Deployment Reality

Several rounds of scrutiny progressively tightened what "validated" actually means for this project:

1. **Raw GLM only → full blended pipeline.** Described in §3 — the original numbers only ever tested the 60+ model against already-known 60+ outcomes.
2. **Aggregate accuracy → partitioned accuracy.** Errors were broken out by ownership tier, position, and time-of-season rather than reported as one blended number, surfacing that the model's edge over baseline *widens* with ownership even though absolute error is worse there (high-ownership players are intrinsically higher-variance, across every model tested including the naive baseline — not a symptom of undertrained), and that different positions favor different formula variants.
3. **Training population → deployment population.** The live pipeline's cold-start filter means roughly a quarter of every validation fold's rows — and ~42% of its 0-minute-outcome rows — are fringe players the deployed system would never actually be asked to score in the first place. Recomputing every headline number restricted to only the rows a real deployment would ever see raised every model's error by roughly 0.25-0.29 MAE (every number reported up to that point had been measured against an easier-than-real population) — but the relative ranking between candidate formulas held, and the calibrated floor's bias advantage got *cleaner*, not weaker, on the realistic slice.
4. **Outcome-level population filtering was explicitly rejected.** A tempting further step — excluding low trailing-points-average players from training, not just validation — was checked and found to be unsafe: roughly a quarter of genuinely established players who go on to play 60+ minutes in a given week have a trailing average under 2 points. These are real, legitimate low-output performances (defenders, defensive midfielders), not noise, and removing them from training would bias the model toward over-predicting everyone. The line drawn: filtering the *population* using information known before the outcome (trailing minutes, cumulative tenure) is safe; filtering training rows by their own *outcome* is not, regardless of how it's framed.

### An open, acknowledged gap: point-wise error isn't decision quality

A more fundamental critique surfaced late in the project and is recorded here rather than papered over: every validation number in this report is a point-wise average error (MAE/RMSE/bias) across a set of rows, treating every row as equally important. But predictions here only matter through the downstream optimizer's *choices* — a large error on a player who was never a realistic pick is invisible to the optimizer, while a small error on two closely-priced, closely-ranked alternatives can flip a real transfer decision. Point-wise accuracy is a proxy for decision quality, not the same thing, and a model can improve on one while not moving the other.

Two more direct measures were discussed as the correct way to close this gap, neither built yet:
- **Ranking-based validation** (e.g., per-gameweek rank correlation, or top-N overlap between predicted and actual scorers among plausible candidates) — tests the property squad selection actually depends on, independent of absolute error size.
- **A full walk-forward backtest through the actual optimizer** — simulate gameweek by gameweek using only past data, run the real squad-selection MILP, and score the *realized* points of the chosen squad against history. This is the only method that directly answers "did this modeling choice produce a better team," but requires deciding how to model an evolving squad/transfer state across simulated gameweeks, which was flagged as a real design decision rather than pure implementation work.

---

## 6. Start-Probability & Lineup Integration

The base start-probability signal is `trail_minutes5 / 90`, clipped to [0, 1] — a purely backward-looking proxy. This was upgraded with scraped RotoWire predicted/confirmed lineups, which carry forward-looking information no trailing statistic can (summer transfers, a new signing walking straight into the XI, this week's team news) — concretely, this is what correctly identifies a new signing as a club's starting goalkeeper on day one, a case the trailing-minutes proxy handled wrong before this fix.

Priority order per player:
1. **Confirmed/predicted lineup starter** — overrides the trailing proxy outright (not averaged with it; a confirmed lineup and a stale minutes trend aren't two votes of equal standing).
2. **Team has a published lineup, player isn't in it** — treated as a firm "not starting" (a fixed low bench probability), catching cases where a player's historical minutes look fine but they've been squeezed out of the current XI entirely.
3. **No lineup published yet for that team** — falls back to the trailing-minutes proxy.

RotoWire only ever publishes the *next* upcoming gameweek's lineup. For multi-gameweek predictions, that single snapshot is carried forward and applied to every horizon in the window rather than reverting to the trailing-minutes proxy for weeks 2-4 — a confirmed current role at a club is judged more durable over a few weeks than the pre-season trailing trend it replaces.

---

## 7. Cold-Start Filtering & Manual Additions

Per the design doc's explicit scope cut, players with no trustworthy Premier League track record — transfers from other leagues, promoted-team players, academy graduates — are excluded from the model and the optimizer's candidate pool **entirely**, at the data stage, rather than given a low or zero predicted score. The bar (fewer than 5 starts or 450 minutes in the prior PL season) is read from the historical archive and checked against the live squad on every run, so newly-promoted clubs and fresh transfers are excluded automatically without any manual list-keeping.

Two deliberate, bounded carve-outs sit outside this filter, both explicitly *not* general mechanisms:

- **A short, named list of newly-arrived players** the user wanted visible to the optimizer despite having no PL history. Prior-season output (goals/assists at their previous club) converts to a rough points-per-90 using standard FPL scoring values, then two judgment-call discount tables apply — how close the source league plays to the Premier League, and how well output at that specific position tends to travel across leagues (goals travel worst; defensive output travels best). These are explicitly documented as a conservative placeholder, never fed into GLM training, and meant to be dropped the moment a player accumulates real PL minutes.
- **A small "enabler" pool** of cheap, cold-start-excluded players, re-surfaced purely as legal squad-filler — not as scored quality candidates. The cold-start filter correctly protects the quality model from untrustworthy predictions, but it also strips out the genuinely cheapest players at each position, leaving the optimizer with an artificially high budget floor. Enablers are scored by a flat nominal per-start value, split into "confirmed lineup starter" and "unconfirmed but available" confidence tiers, so a confirmed cheap starter still ranks above an unconfirmed one at the same price.

---

## 8. Squad Optimizer

A mixed-integer linear program (PuLP with the CBC solver), verified to reach a provably optimal solution (not a heuristic) via solver logs. Binary decision variables cover squad membership, starting-XI membership, and captaincy; the objective is a linear sum of predicted points, with the captain's points counted twice (once as a starter, once again as captain) and a small epsilon reward on bench predicted-points solely to break otherwise-indifferent ties in favor of the better bench option — never large enough to justify a worse starting XI to get it.

All squad-building rules — squad size, budget, formation min/max per position, per-club cap — are read live from the official FPL API's own rules tables rather than hardcoded, so a mid-season rule change would be picked up automatically. Cold-start-excluded players are absent from the candidate pool passed to the solver, not present with a zero score — matching the design doc's explicit requirement.

**Multi-gameweek extension**: the objective sums predicted points across the next several gameweeks with a small per-week discount (default 0.95 — a judgment call, not fit to data, reflecting that a squad decided today has more certain value next week than four weeks out), plus a transfer-cost term when re-optimizing an existing squad rather than building from scratch. This combines the real FPL rule (a hard -4 point cost for any transfer beyond the free allowance) with a smaller, forward-looking *soft* cost on every transfer used, scaled by how many free transfers remain banked — zero at 5 free transfers, growing to roughly -1.5 with only one left, reflecting that spending your last banked transfer forecloses more future flexibility than spending a spare one.

No variance or risk penalty appears anywhere in the objective (a deliberate design-doc decision — best-EV players are rarely genuine bench risks), and rotation risk is never discounted twice: it's already priced once into `predicted_points` via the start-probability blend, so the optimizer's objective doesn't apply a second penalty on top.

---

## 9. Current Production Pipeline

End to end, a live squad-selection run does the following:

1. Fetch fresh live data (squad, prices, fixtures, status, official rules) from the FPL API.
2. Rebuild the cold-start eligibility list against the current live squad.
3. Fetch current RotoWire lineups.
4. Fit the 60+-minute NB2 GLM and the MAE-optimal floor constant (§4) once, on the complete historical season — no held-out fold, since this is genuine forward forecasting into gameweeks with no ground truth yet, not a backtest.
5. Score every eligible candidate across the prediction horizon, blending the GLM's quality prediction with lineup-informed (or trailing-minutes-fallback) start probability and the calibrated floor.
6. Add the manual-addition and enabler candidate pools (§7).
7. Solve the multi-gameweek MILP over the combined pool under the live-fetched squad rules.

The 3-term pipeline with its own sub-60 GLM (§3) was fully built and validated but was **not** adopted into production — it was statistically indistinguishable from the simpler, now-recalibrated 2-term formula, so the simpler design was kept per the project's standing bias against unearned complexity.

---

## 10. Known Limitations / Open Items

- Validation remains fundamentally point-wise (MAE/RMSE/bias over rows), not a direct measure of squad-selection decision quality. Ranking-based validation and a full optimizer backtest (§5) are the identified next steps, not yet built.
- The leakage-safe "established/plausible candidate" population filter has been validated for reporting and for the smaller rate constant, but has not yet been extended to the GLMs' own training rows — flagged as a reasonable, safe next step (origin-time population filtering only, never outcome-based) but not yet implemented in production.
- No systematic feature-selection/pruning pass has been performed on the GLM; not shown to be necessary, but also not rigorously ruled out.
- Cold-start exclusion of promoted-team and new-to-PL players remains a deliberate scope cut per the original design doc, intended to be revisited once real current-season minutes accumulate for those players.
- The design doc's "Option B" fixture-adjustment approach (a second-stage model learning a fixture-difficulty *ratio* rather than fixture difficulty as a direct input feature) was never pursued — the simpler Option A was sufficient and no evidence emerged that the added complexity was needed.
