# Improvement Proposals

## Current Repo Snapshot

This repo already has a solid baseline:

- Data pipeline: Understat match data + football-data odds + ClubElo + FBRef schedule + player-level aggregates.
- Tasks: `over/under 2.5` and `match result`.
- Current production model family: gated residual neural nets that start from bookmaker implied probabilities and learn when to deviate from the market.
- Current categorical inputs: `league_idx`, `home_promoted`, `away_promoted`.
- Current continuous features: a hand-picked subset of rolling xG/goals/shots/pressing/form, Elo, schedule, season progress, and player aggregate features.
- Evaluation protocol: rolling expanding-window CV, fixed epoch-selection season, fixed latest held-out season.

Relevant code anchors:

- Training loop: `training/train_main_model.py`
- Main configs: `training/configs/main_models/*.json`
- Model family: `training/models/neural_net.py`
- Data prep and feature selection: `training/train_utils.py`
- Feature build: `preprocessing/build_understat_features.py`
- Feature engineering: `preprocessing/feature_engineering.py`

## How To Use This List

- Run one idea at a time unless the idea is explicitly a bundle.
- Keep the acceptance protocol frozen.
- Promote only ideas that improve proper scoring metrics first.
- Use betting metrics as secondary diagnostics, not the main optimization target.
- If an idea is large, split it into the smallest falsifiable experiment.

## Suggested First Pass

1. `P001` Separate feature sets for the two tasks.
2. `P002` Separate architecture search spaces for the two tasks.
3. `P004` Tune gate hyperparameters again on top of the frozen architecture/config.
4. `P007` Add missing-value indicators instead of only zero-filling after scaling.
5. `P008` Compare `StandardScaler` vs robust / quantile scaling.
6. `P009` Add explicit home-away difference, ratio, and interaction features.
7. `P010` Add probability calibration on the epoch-selection season.
8. `P013` Add recency weighting in the loss.
9. `P021` Try strong tree-model baselines (`LightGBM`, `CatBoost`, `XGBoost`) with market features.
10. `P043` Use opening odds and closing odds as separate inputs instead of a single coalesced price.
11. `P056` Add exponentially weighted rolling features.
12. `P063` and `P064` Add trend and volatility features.
13. `P080` Expand player-derived features beyond the current `r15` aggregate block.
14. `P027` or `P116` Train league-specific or league-head variants.
15. `P128` Rotate acceptance across more than one held-out season.

## Proposal Backlog

### A. Near-Term, High-Leverage Experiments

- `P001`: Separate feature subsets for `over_under` and `result`. Right now both tasks rely on the same `select_feature_columns()` logic even though the two targets likely want different windows, different player features, and different market inputs.
- `P002`: Separate architecture search spaces per task. The repo already stores separate final configs, but the model family is still the same; the result model and the totals model may want different depth, width, norms, and gate behavior.
- `P003`: Re-run feature-family search specifically for the current canonical baseline and then promote only accepted feature changes into source-controlled configs.
- `P004`: Re-tune gate hyperparameters on top of the frozen architecture. The fixed-arch sweep already exists, so this is a practical next experiment rather than a new system.
- `P005`: Search `league_embed_dim` instead of hard-coding `3`.
- `P006`: Search `N_CV_FOLDS` and the minimum-history cutoff instead of assuming `3 folds` and `>= 5 prior games` are optimal.
- `P007`: Add per-feature missingness flags. The current pipeline scales, then `nan_to_num(..., 0.0)`; a model cannot tell "real zero" from "missing then zero-filled."
- `P008`: Compare `StandardScaler` against `RobustScaler`, `QuantileTransformer`, or rank-gaussian transforms for heavy-tailed football features.
- `P009`: Add explicit home-away deltas, ratios, sums, and products for key features instead of forcing the model to learn those interactions from separate columns.
- `P010`: Add post-hoc calibration on the epoch-selection season. Try temperature scaling for both tasks, plus Dirichlet or vector scaling for the multiclass result model.
- `P011`: For over/under, compare the current `log_loss`-based promotion rule against candidates trained with alternative proper-scoring objectives such as blended `log_loss + brier`.
- `P012`: For result prediction, compare `log_loss`-optimized training against a direct `RPS` surrogate or a blended `log_loss + RPS` objective.
- `P013`: Weight recent seasons more heavily in training so the model adapts faster to current football environments and bookmaker behavior.
- `P014`: Add per-league validation gates. A model that gains overall but collapses in one league may not be operationally better.
- `P015`: Average predictions across multiple seeds for the same accepted config. Tabular neural nets often gain stability from simple seed ensembling.

### B. Architecture And Model-Family Experiments

- `P016`: Replace the single flat tabular MLP with a two-tower model: one tower for home-team features, one for away-team features, then a match-up head.
- `P017`: Use a siamese/shared-weight home-away encoder so the model learns a reusable representation of team state before combining the two teams.
- `P018`: Add explicit cross layers or bilinear interaction layers after the home/away encoders.
- `P019`: Add residual/skip connections inside the MLP trunk.
- `P020`: Try GeGLU blocks or gated linear units inside hidden layers. The repo already has a `GeGLU` implementation that is not currently used in the main architecture.
- `P021`: Train strong tree baselines: `LightGBM`, `CatBoost`, `XGBoost`. They are often very competitive on engineered sports tabular data and can expose feature-value nonlinearities cleanly.
- `P022`: Add linear and multinomial-logit baselines with carefully engineered interactions. If they are close to the neural net, that says a lot about where the remaining edge is.
- `P023`: Try CatBoost with categorical columns directly instead of only embedding `league_idx` and promoted flags.
- `P024`: Train a stacked ensemble: market baseline + neural net + one or more tree models + linear model.
- `P025`: Distill an ensemble into a smaller production model once a better ensemble exists.
- `P026`: Use a mixture-of-experts model with experts specialized by league, season phase, or market-confidence regime.
- `P027`: Use league-specific output heads on top of a shared trunk.
- `P028`: Use a shared trunk with two heads and train `result` and `over_under` jointly as a multi-task model.
- `P029`: Predict home goals and away goals jointly, then derive both `1X2` and `over/under` from the same latent goal model.
- `P030`: Replace direct classification with a bivariate Poisson or Dixon-Coles style goal model, then derive all downstream markets from predicted score distributions.
- `P031`: Try a hurdle or zero-inflated goal model if scoreline distributions look poorly captured by plain Poisson assumptions.

### C. Loss Function, Calibration, And Objective Experiments

- `P032`: Add a loss term that keeps predictions close to market unless data supports a deviation, but tune that term instead of relying only on the gate.
- `P033`: Train on a blended objective: `cross_entropy/log_loss + alpha * brier`.
- `P034`: Use focal loss or class-balanced loss for the multiclass result task, especially to improve draw handling.
- `P035`: Upweight draw samples or hard-draw samples directly. Draws are usually the hardest class and often where market edge, if any, hides.
- `P036`: Use label smoothing for the result task to reduce overconfident tails.
- `P037`: Add calibration-aware penalties for high-confidence mispredictions.
- `P038`: Tune `lambda_repulsion` and `lambda_corr` instead of leaving them at zero forever. They may matter now that the rest of the architecture is more stable.
- `P039`: Try KL regularization toward market probabilities rather than only adding market logits through the gated residual path.
- `P040`: Use group DRO or worst-group validation loss across leagues so gains are not carried by one or two easy slices.
- `P041`: Use importance weighting or domain adaptation to bias training toward the latest season distribution.
- `P042`: Add uncertainty estimation with MC dropout, deep ensembles, or evidential outputs, then use uncertainty as a post-filter.

### D. Market Integration Experiments

- `P043`: Use opening odds and closing odds as separate features rather than collapsing them into a single fallback price.
- `P044`: Keep bookmaker-specific odds as separate inputs (`Bet365`, `Pinnacle`, averages, max) instead of only "first valid" coalesced odds.
- `P045`: Add line-movement features: opening-to-closing delta, percent move, and direction of move.
- `P046`: Add bookmaker-dispersion features: max-min spread, max-avg gap, B365-vs-Pinnacle gap, and number of valid books.
- `P047`: Use overround by market and by bookmaker as an input rather than only one raw margin.
- `P048`: Add Asian handicap line and AH prices from `match_history`; they encode team-strength differences that 1X2 odds alone may not fully expose.
- `P049`: Add BTTS or alternative totals lines if they exist in the raw odds source and can be joined reliably.
- `P050`: Model market movement itself as an auxiliary target. If the feature set predicts future closing movement, that can be a useful signal.
- `P051`: Train a model to beat the closing line, not just the match outcome. Closing-line beating is often a cleaner signal of edge than realized short-run profit.
- `P052`: Let the model choose between several market priors, not just one implied-probability vector.
- `P053`: Compare additive corrections in probability space, logit space, and odds-ratio space.
- `P054`: Condition the gate directly on price movement and bookmaker disagreement, not just implied probs and margin.

### E. Feature Engineering From Existing Match-Level Data

- `P055`: Stop relying mostly on one fixed hand-picked window set. Let the model see multiple windows or let task-specific search choose the best window per feature.
- `P056`: Replace some fixed rolling means with exponentially weighted moving averages so recent matches matter more.
- `P057`: Add expanding-season features for early rounds when `r5` windows are noisy.
- `P058`: Add cross-season carryover priors instead of resetting team state completely at season boundaries.
- `P059`: Blend early-season team features with previous-season team strength, league average, or Elo priors.
- `P060`: Add home-advantage features per team, per league, and per season.
- `P061`: Add schedule differential features directly: home rest minus away rest, home congestion minus away congestion.
- `P062`: Add strength-of-schedule features using the rolling quality of recent opponents.
- `P063`: Add trend features: last-5 minus last-10, slope over recent matches, acceleration, and rolling change rates.
- `P064`: Add volatility features: rolling std, interquartile range, max-min range for xG, goals, shots, and Elo.
- `P065`: Add shot conversion features: goals/xG, goals/shots, shots-on-target/shots, opponent finishing allowed.
- `P066`: Add clean-sheet rate, failed-to-score rate, and BTTS rate.
- `P067`: Add draw propensity features explicitly; draws are not always captured well by generic xG difference features.
- `P068`: Add quantile or median rolling features, not just means and sums.
- `P069`: Add opponent-adjusted venue-specific features. Right now adjusted rolling stats are overall-only; home-specific and away-specific adjusted form may help.

### F. Feature Engineering From Match History / Football-Data Columns

- `P070`: Add rolling cards, fouls, and corners as style and referee-interaction proxies.
- `P071`: Add rolling first-half features from `HTHG`, `HTAG`, `HTR`.
- `P072`: Add set-piece proxies from corners and foul counts.
- `P073`: Add discipline mismatch features such as aggressive team vs card-prone referee.
- `P074`: Add referee features directly if coverage is good enough.
- `P075`: Add shot and shot-on-target features from football-data separately from Understat-derived xG blocks and let search decide what survives.
- `P076`: Add rolling pace proxies such as total shots, total corners, total fouls for the over/under model.
- `P077`: Add closing-vs-opening movement for each market as its own feature family.
- `P078`: Add bookmaker availability count and sparse-book flags. Missing odds structure can itself signal league-season quirks.
- `P079`: Add max price, average price, and sharp-book price separately instead of only a single chosen price.

### G. Player-Derived Feature Extensions

- `P080`: Add short-window player features (`r3`, `r5`) instead of only `r15` plus one `r5_sum` count feature.
- `P081`: Add longer-window player features (`r25` or season-to-date`) to capture structural team quality.
- `P082`: Split player features into attack and defense facing groups if player stats support it.
- `P083`: Add starter continuity features: overlap with previous XI, minutes continuity, repeated core players.
- `P084`: Add top-player dependency features: share of xG/xA/minutes from top 1, top 3, top 5 contributors.
- `P085`: Add player concentration trend features: is the squad becoming more concentrated or more spread out over time?
- `P086`: Add positional aggregates if positions are available or can be inferred.
- `P087`: Add youth/experience features: average age, age-weighted minutes, experience-weighted minutes.
- `P088`: Add injury or suspension features from an external source if obtainable.
- `P089`: Add transfer-window shock features: new minutes share, outgoing minutes share, squad turnover.
- `P090`: Add payroll / wage bill features. There is already a `Capology_payroll_scrapping.py` script in the repo, so this is a natural extension.
- `P091`: Add interaction features between player concentration and schedule congestion; thin squads should react differently to rest compression.

### H. Elo, Strength, And Team-State Experiments

- `P092`: Replace or augment ClubElo with your own learned attack/defense ratings updated sequentially from match results and xG.
- `P093`: Add rolling Elo momentum features: recent Elo change, cumulative Elo delta, Elo volatility.
- `P094`: Add opponent-quality dispersion: recent sequence difficulty, not just average difficulty.
- `P095`: Add previous-season final Elo or final-table priors for early-season stabilization.
- `P096`: Learn league-strength normalization so a single scale works better across the five leagues.
- `P097`: Use per-league or per-season normalized feature z-scores in addition to raw values.
- `P098`: Add latent team embeddings trained from historical team identity, regularized heavily to avoid leakage into memorization.
- `P099`: Use dynamic team-state models where each team has a hidden state updated after every match.
- `P100`: Add promoted-team priors richer than a binary flag, such as estimated promoted-team strength, expected relegation risk, or prior-division performance if data can be collected.

### I. Data Construction, Preprocessing, And Null Handling

- `P101`: Replace blind zero-fill with learned imputation, median imputation, or feature-wise imputation plus missing flags.
- `P102`: Clip or winsorize extreme feature values before scaling.
- `P103`: Use league-specific scalers or league-season scalers for certain families.
- `P104`: Add target-encoding style priors for teams or referees, but only with strict time-safe computation.
- `P105`: Audit all joins for subtle leakage around dates, postponed matches, or duplicated fixtures.
- `P106`: Revisit the season-boundary reset in rolling features; some signals should probably carry across summers.
- `P107`: Add row-quality flags for suspicious joins, suspicious odds, or missing upstream data.
- `P108`: Compare training on complete seasons only versus including partial current-season history inside the train pool.
- `P109`: Add synthetic home-away swap augmentation where valid: swap home and away features, invert target, and mirror the market inputs.
- `P110`: Add noise-based augmentation to continuous features to improve robustness on small league-specific slices.

### J. Search, Ensembling, And Specialization

- `P111`: Jointly search feature families and architecture in a constrained space instead of staging them separately forever.
- `P112`: Search gate parameters, regularization, and architecture together once a narrower feature set is fixed.
- `P113`: Search optimizer and schedule choices: `OneCycle`, warmup + cosine, `ReduceLROnPlateau`, different `AdamW` betas, maybe `RAdam`.
- `P114`: Search batch size and gradient accumulation as part of the final recipe.
- `P115`: Use stability as an objective: penalize configs with high variance across seeds, not just the lowest mean validation loss.
- `P116`: Train league-specific challenger models and compare them with the pooled multi-league model.
- `P117`: Train season-phase specialists: early-season model, mid-season model, late-season model, or a learned router between them.
- `P118`: Train market-confidence specialists: one model for tight lines, one for wide-favorite matches, one for balanced matches.
- `P119`: Use snapshot ensembling, stochastic weight averaging, or fold ensembling for the production candidate.
- `P120`: Add a simple meta-model that combines predictions from several strong challengers plus the market baseline.

### K. Evaluation And Research Process Improvements

- `P121`: Keep per-match prediction archives for every serious experiment so you can compare paired log-loss deltas, not just aggregate metrics.
- `P122`: Add bootstrap confidence intervals for all key metrics and for model-minus-market deltas.
- `P123`: Use paired significance tests on per-match log-loss differences before accepting a change.
- `P124`: Evaluate by league, season, month, and market-confidence decile as standard output.
- `P125`: Evaluate draw-only slices, underdog-only slices, favorites, high-total matches, and low-total matches.
- `P126`: Evaluate calibration separately for each result class, not only aggregate metrics.
- `P127`: Evaluate model edge by bookmaker source if multiple bookmaker prices are available.
- `P128`: Rotate held-out acceptance over multiple recent seasons, not only the single latest season.
- `P129`: Add forward-only backtests that mimic production retraining cadence more closely.
- `P130`: Add experiment dashboards for error clustering, drift, calibration, and market-relative residuals.

### L. Betting / Decision-Layer Experiments

- `P131`: Keep model training focused on scoring metrics, but separately tune the decision layer: EV threshold, minimum edge, and uncertainty filters.
- `P132`: Add uncertainty-aware abstention: only bet when the model is both positive EV and confident in its own calibration.
- `P133`: Use different post-filters for over/under and result; the two markets have different noise structures.
- `P134`: For result bets, compare "best EV only" against "multiple positive EV outcomes allowed" and against capped exposure rules.
- `P135`: Compare fixed-threshold betting with threshold schedules that depend on overround or bookmaker disagreement.
- `P136`: Add calibration-adjusted EV rather than raw EV from raw predicted probabilities.
- `P137`: Use ensemble disagreement as a risk control signal before placing a bet.
- `P138`: Add minimum-closing-line-value style acceptance criteria for any strategy that claims market edge.

### M. Production And Operational Improvements

- `P139`: Build a champion-challenger workflow where a challenger writes outputs side by side with the current production model.
- `P140`: Add data hashes and feature-column hashes to model artifacts so production and training are easy to audit.
- `P141`: Persist feature snapshots used for each accepted model so exact reproduction is possible later.
- `P142`: Add drift monitoring for feature distributions, market distributions, and calibration.
- `P143`: Add production checks for missing feature families, missing odds, and unexpected league-team mapping failures.
- `P144`: Log forward predictions and later join outcomes automatically so live out-of-sample evaluation never depends on manual backfills.

### N. Ambitious / Hard / New-Data Ideas

- `P145`: Add injury, suspension, and predicted-lineup data.
- `P146`: Add manager-change and tactical-style data.
- `P147`: Add weather, temperature, wind, and precipitation for totals prediction.
- `P148`: Add travel distance and rest-travel interaction features.
- `P149`: Add attendance and stadium features from FBRef schedule data.
- `P150`: Add squad market value, transfer spend, or wage-bill features as structural priors.
- `P151`: Add news sentiment or team-specific shock indicators if you can source them reliably.
- `P152`: Add international-break flags and national-team minutes load.
- `P153`: Add cup-match and European-match intensity flags beyond the current simple congestion counts.
- `P154`: Add bookmaker market-depth or liquidity proxies if you can source them.
- `P155`: Train a proper latent score-distribution model using several derivative markets at once: `1X2`, `OU`, `AH`, maybe `BTTS`.
- `P156`: Build sequential team-state models that update online after each match and feed the downstream market model.

## Extra Repo-Specific Notes

- The current canonical model is already market-aware. Many useful experiments will come from improving "when to trust the market" more than from trying to beat the market everywhere.
- The current training parquet is much richer than the final hand-picked feature list. A lot of potential improvement may come simply from better selection and better task-specific subsets.
- The repo intentionally keeps permanent research tooling light. Several of the best ideas above will be cleaner as narrow branch-specific scripts built on top of the canonical pipeline rather than as new long-lived repo utilities.
- The most promising medium-effort direction is probably: better market decomposition + better task-specific feature sets + better calibration.
- The most promising high-effort direction is probably: move from direct classification toward a latent score-distribution model that powers both tasks jointly.

## Simple Experiment Log Template

Use something like this for each candidate:

| Field | Value |
|---|---|
| Idea ID | `P0XX` |
| Task | `binary`, `multiclass`, or both |
| Change | Short description |
| Expected upside | Why this might help |
| Validation metrics | `brier`, `log_loss`, `rps`, calibration notes |
| Held-out test metrics | Same as above |
| Betting diagnostics | Secondary only |
| Accepted? | `yes/no` |
| Follow-up | Next variant to try |
