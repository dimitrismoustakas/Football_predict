# Evaluation Policy

## Scope

This repo evaluates one canonical production model: match result (`Home/Draw/Away`).

## Canonical workflow

Use the fixed trainer in `training/train_main_model.py`.
Do not replace it with ad hoc sweeps when deciding whether a branch is better.

The protocol is source-controlled in `training/configs/main_models/evaluation.json`.
The acceptance path is:
1. rolling expanding-window CV mean `log_loss` as the single experiment objective
2. fixed epoch-selection season for picking training length
3. fixed test season as watch-only monitoring output

The trainer appends each canonical run to `artifacts/experiment_metrics/result_main_runs.tsv`.
Treat that TSV as the single append-only experiment ledger.

## Decision rule

Branch decisions are driven by the CV objective only.
The fixed test season is not part of branch acceptance.

Use this rule unless the user explicitly changes the protocol:
- compare a candidate against the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv`
- any lower CV `log_loss` is an improvement signal worth recording and iterating on
- only compare rows when the protocol and dataset fingerprint match
- if the dataset fingerprint changes, treat the run as a new benchmark series

## Local support work

Cheap local prescreens are allowed to rank nearby ideas before a full canonical run.

Preferred support path for close comparisons:
- choose `best_epoch` on the fixed epoch-selection season
- retrain each candidate for that fixed epoch count on the objective folds
- compare mean CV `log_loss` locally without appending to `artifacts/experiment_metrics/result_main_runs.tsv`

Do not trust the epoch-selection season alone as promotion evidence for nearby candidates.

## Primary metrics

Acceptance is driven by proper scoring metrics first:
- `log_loss`
- `rps`
- `brier`

## Secondary diagnostics

Betting diagnostics are secondary only:
- total profit
- average profit per bet
- number of bets
- bet mix by outcome class

A branch should not be accepted just because short-run betting profit improved while proper scoring got worse.

## Snapshot versioning

Each canonical run records:
- dataset fingerprint for the evaluated training table
- per-season row counts

This is required because the fixed test season can still change when the underlying data is refreshed.
If the dataset fingerprint changes, compare the test number only as context, not as a like-for-like benchmark.

## Runtime outputs

Use `artifacts/models/latest_main_model_metrics.json` as the runtime-generated snapshot of the most recent candidate.
Use `artifacts/experiment_metrics/result_main_runs.tsv` as the canonical experiment ledger.
