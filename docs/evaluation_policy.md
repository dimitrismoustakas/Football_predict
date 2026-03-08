# Evaluation Policy

## Scope

This repo evaluates one canonical production model: match result (`Home/Draw/Away`).

## Canonical workflow

Use the fixed trainer in `training/train_main_model.py`.
The acceptance path is:
1. rolling expanding-window CV for model selection
2. fixed epoch-selection season for picking the training length
3. fixed held-out latest season for the final acceptance check

Do not replace this with ad hoc sweeps when deciding whether a branch is better.

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

Record accepted baselines and new challengers in `training/configs/main_models/baselines.json`.
Use `artifacts/models/latest_main_model_metrics.json` as the runtime-generated source for the latest candidate metrics.
