# FPL Points Prediction & Team Optimization — Implementation Report

*Last updated 2026-10-19 (2026-27 season, ahead of GW8).*

This document summarizes what was built, validated, and shipped against the goals in [DesignDoc.md](DesignDoc.md): a points-prediction model and a mathematical-programming optimizer that turns it into weekly squad, transfer and chip decisions. It covers modeling, validation and system design rather than code, and deliberately records the dead ends, negative results and reversals — several of them are the reason the system looks the way it does.

The short version of where things stand:

- **Points model**: a Negative Binomial (NB2) GLM for points given 60+ minutes, trained on the complete 2025-26 season plus 2022-24 premium attackers, with horizons 1-8 and a per-player adjustment shrunk toward the population, combined with a separately modeled probability of playing 60+ minutes (recent start rate, minutes when starting, season start share) and a calibrated floor for everything else.
- **Decisions**: a multi-period mixed-integer planner that schedules transfers over the next 8 gameweeks with the real free-transfer banking rules, tuned on a full-season backtest; chips are recommended against thresholds derived the same way.
- **Weekly use**: `python -m pipeline.recommend --refresh` pulls the manager's actual squad from the FPL API and prints this week's transfers, the multi-week plan, a wildcard squad, and a chip check.

---

## 1. Data Infrastructure

Several sources feed the system, kept structurally separate because they answer different questions:

- **Historical archive (2025-26, per gameweek)** — ~29,750 rows, all 38 gameweeks, 841 players, enriched with Understat xG/xA. This is the core training season, because it has known outcomes under the current rules.
- **Earlier seasons (2022-23 to 2024-25, `extra_seasons.py`)** — the same vaastav mirror, with player ids kept season-specific so rolling features never cross a season boundary. FPL's own xG/xA only start in 2022-23. Only 2022-24 premium attackers are used in training (§3).
  - **DefCon for earlier seasons.** FPL only began recording defensive contributions in 2025-26. Per-match clearances, blocks, interceptions, tackles and recoveries for 2024-25 come from FotMob (Opta-sourced), as published by [FPL-Core-Insights](https://github.com/olbauday/FPL-Core-Insights), which asks to be credited as the source.
  - **Validation against FPL's official 2025-26 numbers:** the DefCon threshold is reproduced in 99.2% (DEF), 99.3% (MID) and 99.9% (FWD) of games, using *total* tackles (tackles won drops this to ~96%). 2024-25 is rescored to 2025-26 rules (+2 per hypothetical DefCon hit, 2,506 points in total).
  - **Nothing earlier is usable:** FotMob's terms forbid automated access, FBref lost its Opta feed, and Fantasy Football Scout's backfill is members-only.
- **Live FPL API** (bootstrap-static, fixtures) — the current season's players, prices, ownership, fitness flags and the official squad rules (budget, formation limits, club cap). Rules are read live, never hardcoded.
- **Current-season gameweek history** (FPL `element-summary` per player) — this season's per-gameweek rows, used to build live features. The ingest retries each player and refuses to save a partial pull: a transient network outage once overwrote good data with a pull missing 248 players, which would have silently degraded every prediction. Fixtures that haven't been played yet are dropped: once a gameweek opens, the API lists every fixture in it at 0 minutes for every player, so a mid-gameweek run read Haaland and Rudoni (whose GW6 games were still to come) as having missed a game — Haaland's GW7 projection fell from 7.4 to 4.8 and the wildcard sold him.
- **Manager entry** (FPL `entry` endpoints) — the user's actual squad, bank, chips used and transfer history (§10).
- **RotoWire lineups** (scraped, within RotoWire's published terms) — predicted/confirmed XIs for the next gameweek, used only as a start-probability signal. Each fetch also saves a dated snapshot. First accuracy check: of the 218 players RotoWire listed as GW5 starters the day before the deadline, 91% actually started.

Everything is cached as Parquet under `data/raw/` and `data/processed/`. An automated validation pass checks the historical archive on every build (row counts, all gameweeks present, legitimate vs. duplicate double-gameweek rows, Understat coverage, no nulls in core fields).

**Player identity**: matched on FPL's season-stable `code`, never the per-season `id`/`element` (which is reassigned each season — an early bug merged two unrelated players' histories). Name matching against Understat uses token containment with a web-name/full-name overlap tiebreak, added after common first names matched the wrong player.

---

## 2. Feature Engineering & Leakage-Safe Design

### Training table

The core table is **horizon-as-feature**: one row per (player, origin gameweek, horizon), target = points scored *horizon* gameweeks after the origin. One model learns "points N weeks out" with `horizon` as a numeric feature. Horizons now run **1-8** (originally 1-4) so the planner has real projections eight weeks ahead. Every feature uses only data up to the origin gameweek; double gameweeks are summed into one gameweek target.

### Features

- **Playing-time trend**: `trail_minutes5` (mean minutes over the last 5 gameweeks, zeros included) and `trail_games5` (appearances, also the eligibility gate).
- **Form rates (5 appearances)**: pooled per-90 rates — summed stat / summed minutes × 90, so a cameo goal isn't read as a 45-per-90 player — for goals, assists, "points from everything else" (below), xGA, plus a **DefCon hit rate** (share of recent appearances clearing the position's defensive-contribution bonus threshold: 10 for DEF, 12 for MID/FWD). The hit rate replaced raw DefCon per 90 because the bonus is a hard threshold, so expected bonus is linear in the hit probability, not the raw count. Below 30 pooled minutes a rate is treated as noise and the last reliable value is carried forward.
- **Season-to-date rates**: expanding (not rolling) per-90 xG, xA, xGA and points-from-everything-else, null under 90 season minutes. Added after the expert-team analysis (§9) as a longer-run "talent" read alongside 5-game form: walk-forward MAE 2.3926 → 2.3908, 4/6 folds.
- **Fixture**: FPL's official difficulty rating (FDR) and home/away for the target fixture, plus position dummies and two clean-sheet interaction terms (§3).
- **Data-quality flags**: unreliable form and unreliable season-to-date rates — so the model can tell "genuinely low" from "not enough data". A missing-Understat flag was dropped as a feature (2026-10-19): its Understat rates had been pruned, it was 1 for every live player, and it duplicated the season columns once earlier seasons were added. All three flag-then-fill steps live in one shared helper used by training, live scoring and the backtest.

### Feature pruning

A VIF diagnosis found the attacking-output cluster (points, goals, BPS, ICT per 90, and Understat xG/xA duplicating FPL's own) badly collinear: goals per 90 carried a significantly *negative* coefficient. Dropping only the Understat pair didn't fix it; dropping the whole cluster did — every remaining coefficient took the expected sign (xG and xA positive and significant), walk-forward accuracy was unchanged, and BIC was the best of all candidates.

**Goals and assists back in (GW7).** The collinearity came from points per 90 already *containing* goal and assist points. Splitting points into the goal/assist part and everything else (appearance, clean sheets, bonus, DefCon, cards) removes the overlap, so actual output can sit alongside the rest without fighting it. Tested variants (purged walk-forward, MAE vs 2.4056 for the xG/xA model):

| Variant | MAE change | Folds | GW6-9 | Wrong-signed coefficients |
|---|---|---|---|---|
| Goals/assists replace xG/xA everywhere (no split) | −0.0073 ± 0.0024 | 6/6 | +0.0042 | 2 |
| Same, with points split | −0.0075 ± 0.0025 | 6/6 | +0.0043 | 2 |
| Split, goals/assists + season xG/xA | −0.0025 ± 0.0015 | 5/6 | +0.0017 | 2 |
| **Split, 5-game goals/assists, season xG/xA (adopted)** | **−0.0027 ± 0.0017** | **6/6** | **+0.0004** | **0** |

The GW6-9 column is a leave-one-block-out check with near-full training data. It shows the full swap's bigger overall gain is a mid-season effect: over a handful of games actual goals are mostly finishing noise, xG settles faster, and the swap only starts to win around GW10. The adopted version keeps xG/xA for the season-long read, uses actual goals/assists for recent form, has max VIF 3.0 and every attacking coefficient positive. The full swap is a candidate to revisit mid-season.

### Leakage-safe validation

Horizon-as-feature creates near-duplicate rows, so all validation is **walk-forward, expanding-window CV grouped by (player, origin gameweek)**. A further leak was found when extending to 8 horizons: folds split on the *origin* round only, so a training row from origin 13, horizon 4 had its target in the next validation block. Training rows are now **purged** unless their target round precedes the validation block. The leak turned out to be worth only ~0.002 MAE at horizons 1-4, so earlier conclusions stand, but it would have grown with longer horizons.

---

## 3. Points Model

### Decomposition

Raw points are not predicted directly (that conflates "bad player" with "good player who doesn't play"):

```
E[points] = P(plays 60+) × E[points | plays 60+] + (1 − P(plays 60+)) × floor
```

The start probability comes from data (§5), the quality term from the GLM, and the floor is a calibrated constant (§4).

### Model choice and architecture

- **GLM vs. GBM**: Pearson dispersion on the 60+ band is ~1.65, so NB2 rather than Poisson. A shallow regularized GBM was comparably accurate; the GLM was kept for data-efficiency and interpretability.
- **Position architecture**: pooled model vs. a full split into GK+DEF and MID+FWD models vs. pooled with clean-sheet interaction terms (`xGA × defensive`, `fixture difficulty × defensive`). The interaction version won outright (best overall, best for DEF and GK); the split lost even to plain pooling.
- **Horizons 1-8** (purged walk-forward): horizons 1-4 unchanged by also training on 5-8 (2.3928 vs 2.3934 MAE); horizons 5-8 clearly better than extrapolating a 1-4 model (2.4231 vs 2.4364). Error rises only gently with distance — 2.384 at one week out, 2.435 at eight.

### Tested and rejected

Each of these was a reasonable idea, tested with the same walk-forward framework, and not adopted:

| Idea | Result |
|---|---|
| Replace FDR with opponents' rolling xG for/against | Worse at every position (2.4161 vs 2.3926); rolling goals worse still (2.4359) |
| Rolling window other than 5 (3-10) | 8-10 marginally better (2.3910 at 10), mostly DEF/GK; deferred because longer windows delay feature maturity early in a season |
| GBM residual-boosting on top of the GLM | No gain (2.3941), won only 2/6 folds; worse, it applied a −1.95-point "correction" to Haaland — trees can't extrapolate to extreme players, so the most decision-relevant prediction was the most distorted |
| GBM residual-boosting, retried properly (GW7) | LightGBM Poisson starting from the GLM's prediction (multiplicative correction), residuals cross-fit by player, ±15% cap, monotone variant. The first attempt's early stopping used a random holdout, which leaked: each outcome appears in up to 8 horizon rows. With a time-based holdout it stops after 1-48 rounds instead of 500. Still worse than the GLM (+0.0012 ± 0.0004, 1/6 folds), and its largest correction was still Haaland downward (−0.55) in weeks he outscored the GLM. Closed. |
| Penalty-taker feature (cumulative penalty xG) | Significant, correctly signed coefficient but no accuracy gain — already captured by form rates |
| xG/xA × defender interactions | Null; slightly worse for defenders themselves |
| Separate 1-59-minute GLM (3-term pipeline) | Statistically tied with the 2-term formula; simpler version kept |

### Earlier seasons: premium attackers from 2022-24 (adopted 2026-10-19)

The GLM now also trains on 2022-23 and 2023-24 games of midfielders and forwards priced £7.0m+ at the origin gameweek (8,218 rows, 51 players; DefCon is worth only ~0.1-0.2 points a game for them, so its absence there costs little). Each of those seasons gets its own intercept and price slope. Live rows and 2025-26 rows have those columns at 0, so the premium *level* is still learned from 2025-26. The extra rows teach how form, xG and fixtures move premium predictions; recent goals (0.030 → 0.054) and season xG (0.094 → 0.125) both strengthened.

| Training data (purged walk-forward on 2025-26) | RMSE | Bias | £10m+ resid | ≤£5.5m / £5.5-7.5m resid | Top-3 realized per pick |
|---|---|---|---|---|---|
| 2025-26 only | 3.1249 | −0.08 | −0.42 | +0.08 / +0.04 | — |
| + full 2024-25 | 3.1838 | −0.33 | −1.58 | +0.45 / +0.11 | −0.42 ± 0.13 |
| + 2022-24 premiums | 3.1244 | −0.09 | −0.54 | +0.10 / +0.02 | +0.15 ± 0.05 |
| **+ 2022-24 premiums, own price slope per season** | 3.1242 | −0.09 | **−0.28** | +0.09 / +0.03 | **+0.10 ± 0.04** |

- **Season-ahead check.** With no 2025-26 rows in training at all, adding the premiums to 2024-25 improved the top-3 picks by +0.23 ± 0.07 and £7.5m+ MAE by −0.05 ± 0.02.
- **Full 2024-25 rejected.** Even with its own intercept and price slope, it over-predicted 2025-26 premiums badly (a 2024-25-only model put Haaland at 8.8 vs 6.5 actual) and everyone by about 0.3. It also lowered MAE while biasing predictions down, a reminder that MAE rewards under-prediction of a right-skewed target; RMSE, per-band bias and the top-3 picks are the better guides here. Likely causes are Salah's record 2024-25 season and rule changes between seasons, and it may be worth revisiting with a per-event model.
- **Not used for the floor or the start model**, which stay on 2025-26: a premium-only sample would skew a population constant and a start model.

### Player-specific adjustment (adopted 2026-10-15)

The GLM is population-only, so it can't learn that a particular player keeps beating or missing his prediction. Each player now gets a multiplier from his own one-week-ahead 60+ minute games this season: log(actual / predicted), with a points offset of 2, shrunk toward zero by n / (n + 80). This is empirical-Bayes partial pooling, the cheap version of a hierarchical player-ability model.

Purged walk-forward, adjustments built only from games before each validation block:

| Shrinkage k | MAE vs GLM | £7.5m+ MAE | Top-3 realized per pick |
|---|---|---|---|
| 20 | +0.0019 ± 0.0061 | +0.034 | +0.01 ± 0.06 |
| 40 | −0.0008 ± 0.0037 | +0.018 | +0.11 ± 0.04 |
| **80 (chosen on folds 1-3)** | −0.0012 ± 0.0021 | +0.010 | **+0.09 ± 0.05** |
| 160 | −0.0008 ± 0.0012 | +0.005 | +0.08 ± 0.03 |

An additive version behaved the same. The gain is in the picks: about +0.1 realized points per top-3 pick for any k ≥ 40, with MAE unchanged. **It does not fix the premium over-prediction.** Haaland's one-week-ahead errors flip sign from block to block (+1.8, −4.5, +1.4, −3.9, +6.4, +3.7), so there's no stable trait to learn, and premium MAE gets slightly worse. Bruno, by contrast, was under-predicted in every block last season. Only this season's games are used, matching what was validated, so the adjustment is near zero early on (±2% at GW7) and grows through the season. Carrying last season's player effects into this one is untested.

### Calibration checks

Stratifying out-of-fold residuals by 5-game form, xGI and season-to-date output showed **no systematic under-prediction of in-form players** (calibration slopes 0.84-1.0; slopes below 1 mean predictions are if anything slightly too spread, not compressed). The one exception is in-form defenders (top quintile under-predicted by 0.33 ± 0.11 per game). The players an optimizer would buy — top 3 predicted per position each week — came in at −0.02 ± 0.12 points versus prediction: no detectable winner's curse. **Caveat:** the first validation fold starts at GW14, so early-season calibration is untested.

**Is the model too skeptical of actual output?** (GW7, prompted by experts owning Groß, João Pedro, Palmer and Rogers.) Midfielders and forwards with 450+ minutes were grouped by how far their season goals plus assists exceed their xG plus xA. The most over-performing fifth came in at −0.02 ± 0.06 versus prediction, and no group was clearly under-predicted. With season xG in the model, season goals take a *negative* weight: over-performance against xG doesn't persist. So the shrinkage on players like Groß is supported by last season's data.

---

## 4. Floor Calibration

The original hardcoded floor (0.5 points for anyone not reaching 60 minutes) was never fit to data. A naive mean made things worse (it describes high-probability players who rarely trigger the floor), a weighted mean was worse on MAE, and the **weighted median** — the MAE-optimal constant under the formula's own weighting — gave a stable ~−0.10 and a consistent improvement (MAE 1.919 vs 1.981 for the hardcoded floor) across every position, season third and ownership tier.

Refitting the floor on "established" players only made it unstable and worse: those players carry high start probability, hence small weight in this fit, so restriction shrank the weight backing the estimate ~8×. The lesson — population filtering helps some estimators and hurts others depending on how their weight concentrates — is documented in the code.

---

## 5. Start Probability (P(plays 60+))

In priority order per player:

1. **RotoWire starter** (next gameweek only) — overrides everything else.
2. **Team has a published lineup for that fixture but the player isn't in it** — a fixed low bench probability.
3. Otherwise, and for every gameweek beyond the next: a **per-horizon start model** — a shallow gradient-boosted classifier, constrained to be increasing in all three inputs:
   - the **exponentially weighted start rate** (half-life 1.5 games);
   - **average minutes over the last 5 starts** (a full-90 regular vs. a starter usually subbed around the hour);
   - **season start share** (so one recent gap weighs less for an established starter).
4. For weeks 2+, a published next-week lineup is first folded into the start rate as one more game (named = start, left out = no start), and the model one week closer to the target is used.

How it got there:

- The original proxy was `trail_minutes5 / 90`, then a 5-game starts fraction, which separates a 90-minute regular from one often subbed early.
- RotoWire's single next-week snapshot used to be carried forward to all horizons; it now applies only to horizon 1. Carrying it forward gave every confirmed starter the same flat probability for weeks 2-4 regardless of their own reliability.
- **Recalibration**: the raw rate was overconfident at both extremes, worse further out — rows at ~0 still played 60+ 13-19% of the time; rows at ~1.0 fell short 16-26%. A one-feature logistic map per horizon, validated out of sample, improved Brier score at every horizon (e.g. 0.242 → 0.210 at horizon 4).
- **Recency weighting**: a flat 5-game window punished players who had just won a starting place (e.g. Konsa: 0, 11, 90, 90, 90 minutes → ~0.53). The EWM rate beat the 5-game fraction in **24 of 24** walk-forward fold × horizon cells (~5% lower Brier). Side effect: a single recent injury absence weighed heavily.
- **The João Pedro case (GW7).** He started every game he was fit for, missed GW5 injured and returned in GW6, and the one-input model put him at 0.68 next week falling to 0.57. Last season's data showed what it was missing:

  | Pattern (last 6 games) | Played 60+ next week | Old model | New model |
  |---|---|---|---|
  | 6 straight starts, 85+ min when starting | 0.89 | 0.85 | 0.90 |
  | 6 straight starts, under 85 min when starting | 0.68 | 0.85 | 0.75 |
  | One missed game, 85+ min when starting | 0.84 | 0.69 | 0.86 |

  Minutes when starting also stands in for "absent through injury" vs "dropped", which the archive can't record directly. The three-input model beat the one-input recalibration in **44 of 44** fold × horizon cells (Brier 0.1952 vs 0.2045); a logistic model with interaction terms was a close second (0.1955). Trained on GW18+ only and tested on GW4-9, it was well calibrated next week (most nailed bin: predicted 0.94, actual 0.94) and slightly optimistic further out (0.90 vs 0.87 at four weeks). Even fully nailed players only reach 60+ minutes about 89% of the time next week and 77% eight weeks out (injuries, suspensions, rotation, early subs), so p(60+) sits well below managers' "90%+ to start".
- **Lineup carry-over**: archive validation with the actual next start flipped 10% of the time (RotoWire's measured error): weeks 2-8 Brier 0.2000 → 0.1954 on top of the new model, better in 31/38 cells. Combined with the new model: 0.2099 → 0.1954.
- A RotoWire double-fixture bug (a rearranged match live alongside next week's preview gave some players two rows) is resolved by matching on player *and* opponent. The same opponent matching now decides whether a team "has a lineup": mid-gameweek, RotoWire still shows the current round's lineups for teams yet to play, which would otherwise have marked all their players as benched for the next round.

---

## 6. Live Inference: Season Transition and New Players

- **Season blending**: early in 2026-27, live features blended last season's tail with this season's games (various weightings were tried). Once enough current-season games existed, the historical blend was **dropped entirely** — live features now use current-season rows only. The GLM itself is still fit on the complete 2025-26 season.
- **New arrivals**: the original hand-curated discount table for players new to the league (`manual_additions.py`) is **superseded**. Any cold-start-excluded player automatically graduates into the main model once they have 30 real current-season minutes, scored like everyone else on their real data.
- **Cold-start and enablers**: players with no trustworthy PL record are excluded from the scored pool at the data stage rather than given a zero score. A small pool of cheap excluded players is re-added as legal bench filler at a flat nominal value, so the optimizer isn't left with an artificially high price floor. Owned players who drop out of the scored pool (injured, unavailable) stay in the squad at zero projected points instead of vanishing.

---

## 7. Optimizers

### Single-period squad selector (`select_squad`)

A PuLP mixed-integer program over squad, XI and captain (captain counted twice), with all rules read live from the FPL API. It is still used for single-week questions such as the Free Hit check.

**Bench value**, replacing an earlier tie-break epsilon, is grounded in measured rates. A bench player only scores when a starter plays exactly 0 minutes. Among reliable starters, 7.6% still blanked the following week; for goalkeepers it was 4.4% in GW1-29 vs. 11.1% in the end-of-season run-in. An analysis of foreseeability (red cards, transfer-market spikes) found no foreseeable component to strip out. Outfield bench slots get the average of 12% / 5% / 2% rank weights; the bench GK gets 4.4%.

### Multi-period planner (`plan_transfers`) — the production decision engine

One mixed-integer program over the next H gameweeks, with per-player, per-week squad / XI / captain / transfer variables and the **exact free-transfer rule** (+1 per week, cap 5; using more than banked costs 4 points each and leaves 1 next week). The objective is decayed weekly XI points plus bench value, minus hits, plus a small value per transfer still banked after the horizon. Only week 1 is executed; the whole thing is re-solved every week.

Other features:
- **Transfer window**: transfers are only allowed in the first `transfer_weeks` of the plan, so late-window moves aren't judged on truncated payoff.
- **Wildcard mode**: unlimited free transfers in week 1, with banked transfers kept.
- **Pre-filtering**: the candidate pool is cut to ~110 players (top EV, top EV per £, and cheapest per position, plus everyone owned).
- **Solver**: HiGHS, with identical plans to CBC at about 1.5-2× the speed.
- **Owned-player guard**: the planner refuses to run if an owned player is missing from the pool; otherwise that player would be treated as sold for free.

**Tuned settings**: horizon 8, decay 0.7, FT value 0.75, transfers allowed in the first 6 weeks (§8).

This dissolves the problem that sank the earlier policy work: the squad trajectory itself is a decision variable, so choices like "bank this week to make a free double transfer later" come out of the optimization instead of needing a hand-built rule.

---

## 8. Transfer Policy and Chips: Backtesting and Tuning

### The backtest harness (`policy_backtest.py`)

The harness replays the 2025-26 season:
- **Weekly refit**: at each decision week the model is refit only on outcomes known by then.
- **Decisions vs. scoring**: the policy decides using predicted points, and each week is scored on **realized** points (captain doubled, hits subtracted).
- **Start**: the season begins from a fixed GW1 squad, chosen without the model by maximizing ownership within budget.

Fixes along the way:
- the weekly refit originally trained on outcomes up to four weeks in the future;
- double gameweeks were predicted as one fixture but scored as two;
- managers' free-transfer counts could go negative.

Decisions start at GW6, the first week with enough known outcomes to fit a model.

**The central methodological lesson: single replays are path-dependent.** A tiny prediction change flips one transfer, which changes every later week; season totals swing by roughly ±100 points. Policies are therefore compared over multiple seeds of small prediction jitter, with the same seeds for every policy so comparisons are paired. With 8 seeds, differences below roughly 45 points can't be resolved.

### History of transfer policies

1. **Soft-cost heuristic** (original): a single-week optimization with a hand-picked "cost of using a transfer" curve (−1.5 at one free transfer down to 0 at five). Never validated.
2. **Threshold rules**: transfer only if the predicted gain clears a required margin, with families in weeks remaining and banked transfers (linear, then log-binomial in banked transfers, anchored at "a transfer with no free ones costs 4" and "the fifth banked transfer is worth ~0"). The best grid cell scored 1792 vs. 1630 for the heuristic — but after a small model change the ranking **reversed** (1681 vs. 1756). Every conclusion from that phase was within path-dependence noise and has been discarded. Along the way: the first transfer's predicted gain is almost always 2-11 points, so no tested threshold ever held back a free transfer; and a full dynamic program over the transfer bank is intractable because the true state includes the squad itself.
3. **Research review**: the community-standard approach (Sertalp Çay's open-source FPL solver, FPLReview's solver) is a multi-period MILP with a small flat FT value (~1.5-1.75) and decay (~0.85). Their "sensitivity analysis" (vote over many noisy re-solves) was considered and **rejected**: with calibrated predictions it only moves the decision away from expected-value maximization, and no published evidence quantifies a gain.
4. **Multi-period planner** (current, §7).

### Results (2025-26, GW7-38, realized points)

| Policy | Mean | Seed sd | vs. production |
|---|---|---|---|
| Production soft-cost heuristic | 1507 | 65 | — |
| Planner, 4 weeks, default settings | 1567 | 47 | +60 ± 20 (6/8 seeds) |
| Planner, 8 weeks, default settings | 1577 | 32 | +70 ± 22 (7/8 seeds) |
| **Planner, tuned** | **1594** (seeds 0-7) | — | **+86 ± 32 (7/8 seeds)** |

**Tuning** — a coordinate-descent sweep with 16 seeds, taking the best mean at each stage (the tuned planner averaged 1597 over 16 seeds):

| Parameter | Values (mean points) | Chosen |
|---|---|---|
| Decay | 0.7: 1585, 0.8: 1578, 0.85: 1576, 0.9: 1574, 0.95: 1545, 1.0: 1512 | 0.7 |
| FT value | 0: 1591, 0.75: 1595, 1.5: 1585, 2.25: 1588, 3.0: 1590 | 0.75 |
| Transfer weeks | 1: 1580, 2: 1591, 4: 1592, 6: 1597, 8: 1595 | 6 |
| Horizon | 4: 1559, 6: 1576, 8: 1597 | 8 |

What the sweep showed:
- **Discounting matters.** Too little hurts clearly: decay 0.95 cost 32 points and 1.0 cost 64.
- **The 8-week horizon pays off only once decay is tuned.** It beats 4 weeks by 37 ± 13 at decay 0.7, but tied at the default 0.85.
- **Most of the space is flat.** Decay 0.7-0.9, every FT value tested, and transfer weeks 2-8 are all within noise of each other.

The tuned set is +20 ± 14 over the defaults. Both the chosen decay and the chosen horizon sit at the edge of their grids, so neither optimum is bracketed yet.

### Chip thresholds

**How they're derived**: first half of 2025-26, 8 seeds, along the tuned planner's trajectory. Gains are measured each week on un-jittered predictions:
- **Wildcard**: planner objective with the chip minus without.
- **Free Hit**: best one-week XI minus the planned XI.
- **Bench Boost**: the planned bench's points.
- **Triple Captain**: the planned captain's points.

Each chip is then placed exactly once in GW8-19, at most one per gameweek, maximizing total gain. A chip's threshold is its gain in its scheduled week. The one-chip-per-week rule moved at least one chip off its best week in all 8 seeds. Thresholds are assumed to hold for the second half, when a fresh set of chips arrives (GW20).

| Chip | Threshold | Seed sd | Old (obsolete) |
|---|---|---|---|
| Wildcard | 18.7 | 1.7 | 21.15 |
| Bench Boost | 14.0 | 0.6 | 14.24 |
| Free Hit | 9.2 | 1.0 | 11.23 |
| Triple Captain | 8.3 | 0.1 | 8.77 |

Three artifacts found and fixed in the first attempt:
- starting from the ownership-based squad inflated the Wildcard (29.4);
- measuring gains on jittered predictions let Wildcard and Free Hit cherry-pick lucky noise;
- the free model-built starting week inflated Bench Boost (15.1).

---

## 9. Benchmarking Against Expert Squads (GW6)

Two "expert" squads (the most-owned picks among 100 consistently top-1% managers, overall and among those wildcarding) were scored under our own model. Over 4 weeks:
- **Our wildcard**: 187.4
- **Expert wildcard squad**: 175.6
- **Expert overall squad**: 171.3

So the model thinks it's only ~3 points a week better — small relative to its own error. The large individual disagreements clustered into four patterns:

1. **Strong recent output discounted.** Examples: Groß 47 points in 5 GWs, model MID rank 33; De Cuyper rank 31. Calibration analysis (§3) later showed this shrinkage is historically justified, and the over-performance test (§3) found no under-prediction of players beating their xG.
2. **Fixture-driven cheap picks.** A triple-up on Fulham's run against promoted sides, which the experts avoided.
3. **Recently-nailed players under-rated.** Konsa, Muñoz; fixed by the EWM start rate.
4. **Injury doubts weighted near zero.** João Pedro and Palmer at 75%; FPL's own chance-of-playing field isn't used yet.

**Revisited at GW7.** Palmer and Rogers were already rated highly (MID #4 and #7) and lose out on budget and squad structure, not on prediction. João Pedro's gap was start probability, not output (§5): he moved from FWD #16 to #3 once the start model could tell an injury gap from being dropped. Groß rose to MID #8 (start reliability, and the goals/assists features) and is now this week's recommended transfer.

---

## 10. Production Pipeline

```bash
python -m pipeline.recommend --refresh
```

1. Refresh the live FPL data and current-season gameweek history (both fail loudly rather than save partial data).
2. Read `my_team.json` (`{"entry_id": 7592422}`) and fetch the manager's state from the FPL API:
   - **Squad**: as of the last deadline. Upcoming-week transfers stay hidden until the deadline, and a Free Hit week reverts to the previous squad.
   - **Bank**.
   - **Free transfers**: replayed from transfer history.
   - **Selling prices**: FPL keeps half of any rise, e.g. De Cuyper risen £0.5m sells for £4.7m.
   - **Chips used.**
3. Fit the GLM, floor and start-probability model on the complete 2025-26 season; build current-season features from played fixtures only; fetch RotoWire lineups.
4. Score every eligible player for the next 8 gameweeks (blank weeks → 0, double gameweeks summed), add enablers, keep owned players at selling price.
5. Solve the planner for transfers and for a wildcard squad; print the XI, captain, bench and week-by-week plan.
6. Compare each chip's current gain to its threshold, check which chips remain in the current half, and recommend at most one.

`SquadHistory.md` is the human-readable transfer log, refreshed from the API (the API is the source of truth — the hand-kept record once assumed a GW5 transfer that was never made).

**GW6 (made)**: Szoboszlai and Anderson out, Ødegaard and Rudoni in, as recommended; £1.5m left in the bank.

**GW7 (made)**: no transfer; the free transfer was rolled (the planner's Ødegaard → Semenyo was worth almost nothing either way).

**Current state (GW8)**: 2 free transfers, £1.5m in the bank, £99.1m budget at selling prices.
- **Recommended**: De Cuyper out, Murillo in, keeping the second transfer; captain B.Fernandes.
- **Haaland** is projected only 4.2 because of injury doubt (FPL 75%, RotoWire "questionable", so p(60+) 0.54); his points given 60+ minutes is still 7.8.
- **Planner's later moves**: Van Hecke and Tonali (GW9), Hill (GW10), Verbruggen (GW11).
- **Chips**: none clear their thresholds (Wildcard 8.7 against 18.7).

---

## 11. Known Limitations and Open Items

- **Mostly one season of history.** The GLM now also uses 2022-24 premium attackers (§3), but the floor, start model, policy comparisons and chip thresholds rest on 2025-26 alone, and the full 2024-25 season made predictions worse when added. The only held-out-season check is the season-ahead test in §3.
- **Early-season calibration of the points model is untested** (validation starts at GW14) — the period where the largest expert disagreements occur. The start model has been checked early season (§5); the points model hasn't.
- **Policy backtest uses the raw start rate**, not the start model, so backtested policies and chip thresholds predate both start-probability changes.
- **Planner simplifications**:
  - Owned players are priced at their selling price in every week, so a sell-then-rebuy inside a plan is costed at the selling price.
  - The backtest uses a flat £100m budget at each week's prices.
  - There are no auto-substitutions.
- **Chip simplifications**: playing a chip doesn't alter the simulated trajectory, and the second half is assumed to share the first half's thresholds.
- **Tuning edges**: decay 0.7 and horizon 8 are both at the edges of their grids.
- **Not yet used**:
  - FPL's `chance_of_playing_next_round` (injury percentages)
  - Rolling window 8-10 (deferred)
  - Season-to-date goals/assists in place of xG/xA (better from ~GW10 on; revisit mid-season)
- **Understat is missing for the current season.** It isn't ingested for 2026-27, so the missing-Understat flag is set for every live player and shifts all quality predictions up by about 4% (same for everyone, so rankings barely move).
- **`requirements.txt` is incomplete.** It lists pandas, pyarrow and requests, but not pulp, highspy, statsmodels or scikit-learn. LightGBM is installed in the venv for the hybrid experiment but nothing in the pipeline uses it.
- **Design-doc items not pursued**: the second-stage fixture-ratio model ("Option B") and a variance/risk term in the objective. Neither has been shown to be needed.

### Premium-player error: what's been ruled out, and the roadmap (2026-10-15)

**The problem.** Out-of-sample, the highest predictions run high: the top 1% by about 1.2 points, mostly Haaland (−0.98) and Salah (−0.90), the only £12m+ players last season. £7.5-10m players run about 0.5 low. The weekly top 3 per position are still unbiased (−0.00 ± 0.24), so this is about levels at the top, not ranking within a position. It matters for premium-vs-spread decisions and captaincy margins.

**Tried and rejected** (all purged walk-forward):
- **Price shape:** log price, an extra slope above £9m, a spline, price above the positional floor (identical by construction), and per-position price slopes.
- **Price-move features:** rise/fall flags or £ change over the last 1-2 GWs. A tiny, consistent gain (−0.0025, 6/6 folds), but no gain in picks.
- **Dropping price:** much worse (top-3 −0.39 per pick).
- **Price fading through the season:** price matters just as much late as early.
- **Link function:** square-root or identity instead of log.
- **Rebalancing training rows:** weights by price or ownership, dropping low-minute rows, premium-tier slopes, and 50% dropout of under-£6m rows. Price weighting improves £7.5m+ MAE (−0.03 to −0.04) but costs overall accuracy and picks.

**Roadmap from a review of professional and academic practice**, in order of expected impact:
1. **Multi-season training.** *Partly done (2026-10-19, §1, §3): 2022-24 premium attackers adopted; full 2024-25 tested and rejected.* OpenFPL trains on four seasons and matches FPL Review's paid model, beating it for 3+ point returns ([arXiv 2508.09992](https://arxiv.org/html/2508.09992v1)). Our one-season rule (DesignDoc) leaves two £12m+ players. Needs older vaastav and Understat seasons, rescoring or season indicators for rule changes (DefCon only from 2025-26), and season-held-out validation.
2. **Player-specific adjustment, pulled toward the average.** Hierarchical football models give each player an ability shrunk toward the population ([Whitaker et al.](https://arxiv.org/pdf/1710.00001)). A population-only GLM can't learn a player's persistent over- or under-prediction. Cheap version: empirical-Bayes shrinkage of each player's residual. **Done (2026-10-15), see §3:** it improves picks but doesn't fix the premium level error.
3. **Per-event model.** Model goals, assists, clean sheets, bonus and minutes separately and combine them with the scoring rules (FPL Review; AIrsenal's team-scoreline × player-involvement split). Lets finishing be shrunk properly: xG per 90 carries over season to season about 0.6, finishing above xG about 0.1 ([Evidently](https://evidently.substack.com/p/how-can-we-tell-if-a-footballer-will), [Stats Perform](https://www.statsperform.com/resource/re-examining-finishing-skill)).
4. **Bookmaker odds** (anytime-goalscorer, clean-sheet markets). The commercial gold standard, but player-prop feeds are paid and need history.
- **Cheap tweaks:** outcome-binned sample weights, balanced across score bins and clipped at the 95th percentile (OpenFPL); form windows of 1, 3, 5, 10 and 38 games alongside the current ones.
