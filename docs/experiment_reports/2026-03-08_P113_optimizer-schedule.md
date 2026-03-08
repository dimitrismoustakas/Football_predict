# Standard Experiment Report

## Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID | `P113-optimizer-schedule` |
| Date | `2026-03-08` |
| Proposal ID | `P113` |
| Proposal title | `Search optimizer and schedule choices` |
| Task | both |
| Branch | `exp/P113-optimizer-schedule-20260308` |
| Status | `reject` |

## Proposal Chosen

> Search optimizer and schedule choices: `OneCycle`, warmup + cosine, `ReduceLROnPlateau`, different `AdamW` betas, maybe `RAdam`.

## Why This Idea Was Chosen

- The user explicitly asked to tune learning rate, related optimizer hyperparameters, scheduler behavior, and final retraining epoch behavior.
- `P113` is self-contained, fits the canonical workflow, and does not require new data sources.
- The current main configs had a single fixed `AdamW` + cosine recipe, so this proposal was still open in the codebase.

## Implementation Summary

- Added configurable optimizer support to the canonical trainer with `adamw` and `radam` options, plus `beta2`, `optimizer_eps`, and optional gradient clipping.
- Added configurable scheduler support with `cosine`, `warmup_cosine`, `onecycle`, `plateau`, and `none` options.
- Added `final_epoch_mode` and `final_epoch_buffer` so the final retrain can use a deterministic rule derived from the epoch-selection season instead of always using the raw best epoch directly.
- Added a narrow branch-only search runner, `training/evaluate_optimizer_schedule_experiment.py`, to compare compact recipe sets on rolling CV and the held-out season.
- Updated the branch configs to the CV-selected challengers:
  - binary: `adamw_warmup_cosine_stable`
  - multiclass: `radam_warmup_cosine`

## Files Changed

- `training/models/neural_net.py`
- `training/train_utils.py`
- `training/train_main_model.py`
- `training/configs/main_models/over_under.json`
- `training/configs/main_models/result.json`
- `training/evaluate_optimizer_schedule_experiment.py`
- `docs/experiment_reports/2026-03-08_P113_optimizer-schedule.md`

## Commands Run

1. `uv run python training/evaluate_optimizer_schedule_experiment.py`
2. `uv run python -m training.train_all_models`
3. `git status --short`

## Evidence Sources

- `artifacts/models/latest_main_model_metrics.json`
- `artifacts/experiment_metrics/p113_optimizer_schedule_binary.json`
- `artifacts/experiment_metrics/p113_optimizer_schedule_multiclass.json`
- `training/configs/main_models/baselines.json`

## Proper Scoring Metrics

### Rolling CV — Over/Under

| Split | Champion primary | Candidate primary | Delta | Champion secondary | Candidate secondary | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 oldest (`2223`) | 0.671180 | 0.670984 | -0.000195 | 0.239151 | 0.239050 | -0.000101 |
| Fold 2 middle (`2324`) | 0.658366 | 0.658535 | +0.000169 | 0.233069 | 0.233135 | +0.000067 |
| Fold 3 latest (`2425`) | 0.667814 | 0.667699 | -0.000114 | 0.237681 | 0.237618 | -0.000063 |
| Weighted mean | 0.665653 | 0.665607 | -0.000046 | 0.236591 | 0.236560 | -0.000032 |

### Held-Out Season — Over/Under (`2526`)

| Metric | Champion | Candidate | Delta |
|---|---:|---:|---:|
| Primary (`log_loss`) | 0.679639 | 0.681330 | +0.001691 |
| Secondary (`brier`) | 0.243550 | 0.244178 | +0.000628 |

### Rolling CV — Match Result

| Split | Champion primary | Candidate primary | Delta | Champion secondary | Candidate secondary | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 oldest (`2223`) | 0.978861 | 0.977320 | -0.001541 | 0.201581 | 0.201238 | -0.000343 |
| Fold 2 middle (`2324`) | 0.954770 | 0.953456 | -0.001315 | 0.187576 | 0.187043 | -0.000533 |
| Fold 3 latest (`2425`) | 0.964766 | 0.964574 | -0.000192 | 0.195710 | 0.195651 | -0.000060 |
| Weighted mean | 0.964586 | 0.963787 | -0.000799 | 0.194444 | 0.194186 | -0.000258 |

### Held-Out Season — Match Result (`2526`)

| Metric | Champion | Candidate | Delta |
|---|---:|---:|---:|
| Primary (`log_loss`) | 0.973204 | 0.973296 | +0.000093 |
| Secondary (`rps`) | 0.195549 | 0.195483 | -0.000066 |

## Stability

Binary stability was not extended beyond seed `42` because the held-out primary metric was already worse than the champion. Multiclass stability was rerun for seeds `42`, `43`, and `44`; all three runs reproduced the same held-out primary metric.

| Seed | Candidate primary | Delta vs champion |
|---|---:|---:|
| 42 | binary `0.681330`, multiclass `0.973296` | binary `+0.001691`, multiclass `+0.000093` |
| 43 | binary not run, multiclass `0.973296` | binary not run, multiclass `+0.000093` |
| 44 | binary not run, multiclass `0.973296` | binary not run, multiclass `+0.000093` |
| Mean | binary not run, multiclass `0.973296` | binary not run, multiclass `+0.000093` |

## Slice Checks

| Slice | Result | Notes |
|---|---|---|
| League breakdown | mixed | Binary latest CV improved in `3/5` leagues but held-out improved in only `2/5`; multiclass latest CV improved in `3/5` leagues and held-out improved in `2/5`. |
| Latest CV fold | pass | Both selected challengers improved the latest CV primary metric. |
| Draw slice if multiclass | fail | Multiclass draw-only log loss worsened from `1.334995` to `1.366794` on the latest CV fold and from `1.332560` to `1.347791` on held-out. |
| Market-confidence deciles | mixed | Binary improved `4/9` latest-fold deciles and `6/9` held-out deciles; multiclass improved `6/10` latest-fold deciles but only `3/10` held-out deciles. |
| Calibration | mixed | Binary latest-fold calibration improved (`0.00990` -> `0.00862`) but held-out worsened (`0.02490` -> `0.03863`); multiclass latest-fold calibration improved (`0.02956` -> `0.02685`) while held-out was roughly flat/slightly worse (`0.01838` -> `0.01859`). |

## Betting Diagnostics

| Diagnostic | Champion | Candidate | Notes |
|---|---:|---:|---|
| Over/under total profit | -5.38 | -7.09 | Candidate placed many more bets (`285` vs `32`) and lost more total units. |
| Over/under ROI | -0.1253 | -0.0911 | Less-negative daily ROI, but proper scoring worsened on held-out. |
| Over/under number of bets | 32 | 285 | Secondary only. |
| Match-result total profit | 20.61 | -2.80 | Candidate reduced bet volume sharply (`23` vs `172`) and lost money on held-out. |
| Match-result avg profit | 0.1198 | -0.1217 | Secondary only; not used for promotion. |
| Match-result number of bets | 172 | 23 | Secondary only. |

## Scorecard Draft

Reuse the structure and checks from `docs/model_acceptance_scorecard_template.md`, but leave the decision unresolved here.

| Field | Value |
|---|---|
| Final decision | `reject` |
| Why | The experiment fails the promotion gates. Binary has a small weighted-CV win but a clear held-out regression on both `log_loss` and `brier`. Multiclass improves rolling CV, but held-out `log_loss` is still slightly worse and the draw slice regresses on both the latest CV fold and held-out season. |
| Risks | Binary held-out calibration worsened materially; multiclass draw handling worsened on both the latest CV fold and held-out season; both challengers materially change betting activity. |
| Follow-up experiment | Split `P113` by task and continue around the multiclass `radam + warmup_cosine` recipe while leaving binary closer to the incumbent. |

## Notes For Promotion Review

- Evidence looks **mixed to weak** overall.
- Binary does not currently look promotion-ready from the held-out results.
- Multiclass is closer: rolling CV improved clearly and held-out `rps` improved slightly, but held-out `log_loss` and draw-slice behavior still need review.
- `training/configs/main_models/baselines.json` was left unchanged.
- Generated model bundles under `artifacts/models/` were not staged for commit.
- The runtime experiment JSON files under `artifacts/experiment_metrics/` were used as evidence outputs and can be regenerated with the commands above.

## Promotion Review Outcome

- Outcome: `reject`
- Baseline history update: not allowed because the experiment did not pass the proper-scoring promotion gates.
- Main reason: the combined proposal is not promotion-safe. Binary clearly regressed on the held-out season, and the multiclass candidate still showed worse held-out `log_loss` plus a repeated draw-slice regression.
- Next action: keep the incumbent baseline unchanged and carry only this finalized report onto `main` as the minimal review record.
