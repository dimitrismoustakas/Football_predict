# Evaluation Policy

## Scope

This repo evaluates one canonical production model: match result (`Home/Draw/Away`).

## Canonical workflow

Use the fixed trainer in `training/train_main_model.py`.
The acceptance path is:
1. rolling expanding-window CV mean `log_loss` as the single experiment objective
2. fixed epoch-selection season for picking the training length
3. fixed held-out latest season for the final acceptance check

Do not replace this with ad hoc sweeps when deciding whether a branch is better.

The trainer appends each canonical run to `artifacts/experiment_metrics/result_main_runs.tsv`.
Treat that TSV as the append-only experiment ledger for the scalar objective and headline diagnostics.

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

## Promotion record

Record accepted baselines in `training/configs/main_models/baselines.json`.
Use `artifacts/models/latest_main_model_metrics.json` as the runtime-generated source for the latest candidate metrics.
