# Evaluation Policy

## Purpose

This document defines how model changes are evaluated, what metrics decide promotion, and how the repo should handle iterative experimentation without overfitting to the latest held-out season.

This policy is meant to sit on top of the current canonical workflow implemented in:

- `training/train_main_model.py`
- `training/train_all_models.py`
- `training/train_utils.py`
- `training/configs/main_models/baselines.json`

## Current Canonical Structure

The repo currently uses:

- rolling expanding-window CV as the research-time selection protocol
- one fixed pre-test season for epoch selection
- the latest season as the held-out acceptance season

Important clarification:

- For search and research scripts, rolling CV is the main model-selection tool.
- For the frozen source-controlled main configs, the canonical training script does not re-search hyperparameters. It uses the fixed config, picks the epoch on the final pre-test season, retrains on all pre-test seasons, and evaluates once on the latest season.

## Core Rule

Optimize on rolling CV.

Use the held-out season only as a promotion gate.

Do not tune against the held-out season. If a model change is repeatedly selected because it looked best on the held-out season, that season is no longer held out.

## Why

Rolling CV should drive iteration because:

- it gives more than one time split
- it reduces variance relative to a single season
- it tests whether gains are persistent across time
- it is the only safe place to compare many candidate ideas

The held-out season should only answer:

- "After the candidate was frozen, does it still hold up on unseen recent data?"

It should not answer:

- which of several close variants to prefer
- which hyperparameter value to use
- which feature family edge case to keep

## Latest-Season Rule

As of March 7, 2026, the repo's current held-out season is `2526`, and that season is still in progress.

That means `2526` should be treated as:

- a held-out audit set
- a forward-monitoring set
- not the sole authority for promotion if the signal is weak

Practical implication:

- strong CV win plus neutral `2526` result: challenger or cautious promotion
- strong CV win plus strong `2526` win: promote
- strong CV win plus clear `2526` loss: reject
- weak CV win plus good `2526` result: do not trust it; likely noise

Once `2526` is complete, freeze it as a true acceptance season and use the next season as the forward-monitoring season.

## Metrics Policy

Model inclusion is decided by proper scoring metrics first.

Betting metrics are diagnostics only.

### Over/Under

- Primary metric: `log_loss`
- Secondary metric: `brier`
- Diagnostics: calibration, daily-budget ROI, number of bets, profit

### Match Result

- Primary metric: `log_loss`
- Secondary metric: `rps`
- Diagnostics: classwise calibration, number of bets, profit

## Important Implementation Note

The current code still labels binary comparison with `brier` in the canonical config metadata.

Policy-wise, this repo should still judge binary challengers using:

- `log_loss` as the main promotion metric
- `brier` as a required secondary check

If code and policy disagree, follow the policy for promotion decisions until the implementation is updated.

## Research-Time Selection Rule

During iterative development, choose candidates using rolling CV only.

Recommended ranking rule for 3 folds:

- oldest fold weight: `0.2`
- middle fold weight: `0.3`
- most recent fold weight: `0.5`

Use:

- weighted mean of the primary metric as the main ranker
- latest fold as a guardrail

Reason:

- deployment resembles recent seasons more than old seasons
- but a model that only works on one recent slice is not robust enough

## Promotion Rule

A candidate is accepted only if all of the following are true.

### Mandatory checks

1. The weighted rolling-CV primary metric improves versus the current champion.
2. At least 2 of the 3 CV folds improve on the primary metric.
3. The most recent CV fold does not show a material regression.
4. The secondary proper metric is non-worse in rolling CV.
5. The held-out season does not show a clear regression.
6. No major league-level collapse appears in the evaluation slices.

### Stability check

For any serious promotion candidate:

7. Re-run with at least 3 seeds.
8. If the gain disappears or flips sign across seeds, do not promote.

## Candidate Statuses

Every experiment should end in one of three states.

### Accept

Use `accept` when:

- CV is clearly better
- latest-fold guardrail passes
- held-out is non-worse or better
- the result is stable enough across seeds

### Reject

Use `reject` when any of the following happens:

- the gain comes from only one fold
- the latest CV fold is clearly worse
- the held-out season is clearly worse
- proper scoring degrades even if betting profit improves
- the result is unstable across seeds and the delta is small

### Challenger / Watchlist

Use `challenger` when:

- CV is clearly better
- held-out is inconclusive because the season is partial or the delta is tiny

This is the correct bucket for many in-season candidates.

## What Counts As "Clearly Better"

Best practice:

- compare paired per-match loss deltas
- bootstrap confidence intervals
- promote only when uncertainty supports the gain

Practical minimum if bootstrap is not yet implemented:

- weighted CV primary metric improves
- at least 2 of 3 folds improve
- most recent fold is non-worse
- held-out is non-worse

## Recommended Slices To Always Review

Do not look only at aggregate metrics. Review:

- by league
- by season
- by month or season phase
- by market-confidence decile
- by favorite / balanced / underdog segment
- for result: by class, especially draws

Any candidate with one severe slice regression should be treated cautiously even if the aggregate metric improved.

## What Must Not Decide Promotion

Do not promote a candidate because:

- profit improved over a small sample
- one held-out season looked lucky
- one bookmaker or one league carried the entire gain
- the model placed fewer bets and therefore looked "safer"
- accuracy improved while log-loss worsened

## Operating Workflow

Use this workflow for each idea:

1. Run the experiment against rolling CV.
2. Compare against the current champion on the primary and secondary proper metrics.
3. If it wins in CV, run the canonical retrain and score the held-out season.
4. If it still passes, rerun with 3 seeds for stability.
5. Mark the candidate as `accept`, `reject`, or `challenger`.
6. Only accepted candidates should replace the source-controlled baseline entry.

## Baseline Logging Rule

When a candidate is accepted:

- copy the metrics into `training/configs/main_models/baselines.json`
- record a short description of the change
- keep the previous accepted model in history

When a candidate is not accepted:

- do not overwrite the accepted baseline
- keep the challenger result in runtime artifacts or experiment logs only

## Summary

The repo should behave as follows:

- optimize on rolling CV
- use the held-out season for promotion only
- treat the in-progress `2526` season as a noisy audit set
- decide inclusion on proper scoring metrics first
- require recency robustness and multi-fold consistency
- use betting metrics only as secondary diagnostics

