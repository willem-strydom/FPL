# FPL Points Prediction & Team Optimization — Design Doc

## Goal
Build a two-part system:
1. ML model predicting expected points per player over the next N gameweeks (N ≥ 4)
2. Mathematical programming optimizer for team selection / transfers using those predictions

Philosophy: keep the dataset and pipeline as simple as possible. FPL is low signal-to-noise; added complexity should earn its place with a measurable accuracy gain, not just theoretical soundness.

---

## Data Sources (deliberately minimal)

- **Official FPL API**: player status, "chance of playing" flag, DefCon stats (built in natively), price, ownership
- **FBref**: historical xG, xA per player per match
- **Understat**: alternative shot-based xG/xA (optional, cross-check vs FBref)
- **Rolling minutes history** (derived from FPL API): last-5-game average minutes as core "will they start" feature
- **Data Freshenss** Only use data from the past season (25-26). Older data is less informative and has drastically more issues regarding transfers, minutes and injuries.

**Explicitly excluded for now:**
- Betting odds as start-probability proxy — considered, deferred as "nice to have," not core
- Scraped predicted lineups (RotoWire, FF Scout) — same, deferred
- No dedicated injury-news scraping beyond the official chance-of-playing flag

**Scope cut — cold start players excluded from v1:**
- Transfers from other leagues, promoted-team players, and academy graduates with no/limited PL history are **excluded from the model and the optimizer's candidate pool entirely** until they accumulate real PL minutes (a few gameweeks into the season).
- Rationale: avoids cross-league data merging, hardcoded league-strength adjustment factors, and low-confidence predictions that could fool a pure expected-value optimizer.
- Implementation note: filter these players out at the data stage so the optimizer never sees them as candidates (not handled via imputed/zero scores downstream).

---

## Points Model Architecture

### Two-stage decomposition (core design decision)

Instead of one model predicting raw points directly (which implicitly conflates "bad player" with "good player who doesn't play much"):

```
E[points] = P(plays 60+ min) × E[points | plays 60+ min]
          + P(plays 1-59 min) × E[points | plays 1-59 min]
          + P(doesn't play) × baseline appearance floor
```

- **Start probability**: sourced from data directly (rolling minutes, chance-of-playing flag), NOT modeled from scratch — decision made to avoid building a start-likelihood classifier.
- **Quality model** ("form model"): trained only on gameweeks where the player actually played, so per-90 quality rate isn't dragged down by rotation risk. This is the piece that gets ML treatment.
- Combined expected value fed to optimizer as a single coefficient — **do not double-penalize** bench-risk players by also discounting in the optimizer objective function.

### Fixture adjustment

Two options under consideration, leaning toward the simpler one:
- **Option A (preferred default)**: single model with fixture difficulty rating as a direct input feature alongside form/xG features. Simpler, no leakage risk to manage.
- **Option B (if interaction structure is suspected)**: two-stage stacking — form model produces a baseline rate, second-stage model learns a **ratio** (`actual_points / form_prediction`), not raw points, as a function of fixture difficulty. Requires out-of-fold predictions from the form model (via time-respecting CV) to avoid leakage — the naive version (predicting on same data the form model trained on) makes fixture difficulty look artificially unimportant.

Decision: try Option A first; only add Option B's complexity if empirically justified.

### Multi-gameweek horizon

- **Approach**: one model, with `horizon` (1, 2, 3, 4...) as an explicit input feature — not separate models per horizon, not multi-output/vector prediction.
- Rationale: maximizes usable training rows per horizon (vs. splitting limited history across N separate models), and lets the model learn the "regression to the mean as horizon grows" pattern directly from data rather than hand-tuning decay.
- Multi-output rejected: requires all N weeks' fixture data as separate columns per row, complicates leakage-safe CV, no clear benefit over horizon-as-feature here.

### Model choice

- **Not a single default choice** — plan to empirically compare:
  1. Poisson (or negative binomial if overdispersed) GLM — natural fit for goals/assists as sparse count data, data-efficient, interpretable
  2. Shallow, heavily regularized GBM — captures FPL's genuine scoring discontinuities (60-min threshold, DefCon threshold, bonus point non-monotonicity) that a linear model can't represent without manual threshold features
  3. Position-specific submodels of either — scoring rules differ enough by position that this may capture much of what nonlinearity would otherwise be needed for
- Decision on which wins is deferred to empirical comparison on validation data, not decided in the abstract.

---

## Validation Methodology (critical — a real bug risk, not just theory)

The horizon-as-feature design creates near-duplicate rows: each (player, origin-gameweek) produces N rows differing by only ~2 features (horizon, target gameweek's fixture). This does **not** invalidate the point-prediction approach itself (tree models / GBMs don't require IID to fit validly), but it **does** create a serious cross-validation leakage risk.

**Required CV design:**
1. **Group folds by (player, origin gameweek)** — all N horizon-rows generated from one decision point must move together into the same fold. Never split them across train/validation.
2. **Time-respecting split** — never validate on the past using a model trained on the future (standard time-series discipline, stacked on top of constraint 1).

Also note for planning purposes: effective sample size is smaller than raw row count suggests once horizon-duplication is accounted for — don't expect accuracy gains that a naive row-count multiplication would imply.

---

## Optimizer Notes

- Standard expected-value maximization for transfers/squad selection — no variance penalty needed here.
- **Captaincy**: decision made to skip variance-adjustment machinery. Rationale: best/highest-EV players rarely come off the bench, so the EV-vs-variance tradeoff that matters for rotation-risk players is largely moot for realistic captain candidates.
- Must exclude cold-start players (see Data Sources) from the candidate pool entirely, not just give them low/zero predicted scores.

---

## Open Items / To Decide During Implementation

- [ ] Empirically compare Poisson GLM vs. shallow GBM vs. position-split versions of both
- [ ] Decide fixture-adjustment approach (Option A vs B above) based on initial single-model performance
- [ ] Confirm DefCon threshold value and current scoring rules directly from official FPL rules (rule details can change season to season)
- [ ] Revisit cold-start player exclusion after first few gameweeks of new season — reintroduce with real PL data once available