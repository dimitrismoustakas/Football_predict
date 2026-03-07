# Standard Experiment Report

## Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID | `P007-missing-flags` |
| Date | `2026-03-07` |
| Proposal ID | `P007` |
| Proposal title | `Add per-feature missingness flags` |
| Task | both |
| Branch | `exp/P007-missing-flags-20260307` |
| Status | `reject` |

## Proposal Chosen

> Add per-feature missingness flags. The current pipeline scales, then `nan_to_num(..., 0.0)`; a model cannot tell "real zero" from "missing then zero-filled."

## Why This Idea Was Chosen

- It is in the `Suggested First Pass` section of `improvement_proposals.md`.
- It is self-contained, touches the canonical pipeline directly, and does not require new data sources.
- The current training logs already show persistent feature missingness, so the idea was testable immediately.

## Implementation Summary

- Added derived `__is_missing` indicator columns for every selected continuous feature when the config enables the proposal.
- Kept the existing `StandardScaler` flow for base continuous features, then appended unscaled binary missingness flags after scaling and before model input.
- Added config support through `add_missing_indicators` in both frozen main-model configs, with an environment override in the canonical trainer for fair champion/candidate comparison.
- Updated production inference to rebuild the same augmented input matrix and to stop dropping rows solely because base continuous features are missing.
- Added a narrow reproducibility script to compare champion (`add_missing_indicators=false`) versus candidate (`true`) on rolling CV, held-out season, and slice diagnostics.

## Files Changed

- `training/train_utils.py`
- `training/train_main_model.py`
- `training/configs/main_models/over_under.json`
- `training/configs/main_models/result.json`
- `prod_run/pipeline.py`
- `training/evaluate_missing_indicator_experiment.py`
- `docs/experiment_reports/2026-03-07_P007_missing-flags.md`

## Commands Run

1. `uv run python training/evaluate_missing_indicator_experiment.py`
2. `TASK_TYPE=multiclass uv run python training/evaluate_missing_indicator_experiment.py`
3. `uv run python -m training.train_all_models`
4. `uv run python -c "import prod_run.pipeline; print('ok')"`

## Evidence Sources

- `artifacts/models/latest_main_model_metrics.json`
- `artifacts/experiment_metrics/p007_missing_flags_binary.json`
- `artifacts/experiment_metrics/p007_missing_flags_multiclass.json`
- `training/configs/main_models/baselines.json`

## Proper Scoring Metrics

### Rolling CV — Over/Under

| Split | Champion primary | Candidate primary | Delta | Champion secondary | Candidate secondary | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 oldest (`2223`) | 0.671180 | 0.671174 | -0.000006 | 0.239151 | 0.239126 | -0.000024 |
| Fold 2 middle (`2324`) | 0.658366 | 0.658483 | +0.000117 | 0.233069 | 0.233078 | +0.000009 |
| Fold 3 latest (`2425`) | 0.667814 | 0.667329 | -0.000485 | 0.237681 | 0.237446 | -0.000235 |
| Weighted mean | 0.665653 | 0.665444 | -0.000209 | 0.236591 | 0.236472 | -0.000120 |

### Held-Out Season — Over/Under (`2526`)

| Metric | Champion | Candidate | Delta |
|---|---:|---:|---:|
| Primary (`log_loss`) | 0.679639 | 0.680405 | +0.000766 |
| Secondary (`brier`) | 0.243550 | 0.243894 | +0.000344 |

### Rolling CV — Match Result

| Split | Champion primary | Candidate primary | Delta | Champion secondary | Candidate secondary | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 oldest (`2223`) | 0.978861 | 0.976920 | -0.001940 | 0.201581 | 0.201155 | -0.000425 |
| Fold 2 middle (`2324`) | 0.954770 | 0.954369 | -0.000401 | 0.187576 | 0.187102 | -0.000474 |
| Fold 3 latest (`2425`) | 0.964766 | 0.965288 | +0.000522 | 0.195710 | 0.195727 | +0.000017 |
| Weighted mean | 0.964586 | 0.964339 | -0.000247 | 0.194444 | 0.194225 | -0.000219 |

### Held-Out Season — Match Result (`2526`)

| Metric | Champion | Candidate | Delta |
|---|---:|---:|---:|
| Primary (`log_loss`) | 0.973204 | 0.975374 | +0.002171 |
| Secondary (`rps`) | 0.195549 | 0.195816 | +0.000267 |

## Stability

Three-seed stability was **not run**. Under the repo policy this proposal did not clear the held-out gate on the primary metric for either task, so it was not treated as a serious promotion challenger.

| Seed | Candidate primary | Delta vs champion |
|---|---:|---:|
| 42 | binary `0.680405`, multiclass `0.975374` | binary `+0.000766`, multiclass `+0.002171` |
| 43 | not run | not run |
| 44 | not run | not run |
| Mean | not run | not run |

## Slice Checks

| Slice | Result | Notes |
|---|---|---|
| League breakdown | fail | Binary held-out regressed in 4/5 leagues; multiclass held-out regressed materially in `ENG-Premier League` and `GER-Bundesliga` despite gains in some other leagues. |
| Latest CV fold | mixed | Binary latest fold improved; multiclass latest fold regressed on both `log_loss` and `rps`. |
| Draw slice if multiclass | fail | Held-out draw-only log loss moved from `1.332560` to `1.332753`; latest CV draw slice moved from `1.334995` to `1.385597`. |
| Market-confidence deciles | mixed | Binary latest CV improved in 7/10 deciles, but held-out worsened in 6/10; multiclass held-out worsened in 5/10 deciles including several mid/high-confidence bins. |
| Calibration | fail | Binary held-out calibration error rose from `0.033885` to `0.037223`; multiclass held-out calibration error rose from `0.039390` to `0.049631`. |

## Betting Diagnostics

| Diagnostic | Champion | Candidate | Notes |
|---|---:|---:|---|
| Over/under total profit | -5.38 | -8.84 | Candidate placed more bets (`129` vs `32`) and lost more total units. |
| Over/under ROI | -0.1253 | -0.0967 | Less-negative ROI, but proper scoring and total profit both worsened on held-out. |
| Over/under number of bets | 32 | 129 | Daily-budget diagnostics remain secondary only. |
| Match-result total profit | 20.61 | -11.55 | Candidate flipped a positive held-out diagnostic into a loss. |
| Match-result avg profit | 0.1198 | -0.0173 | Worse secondary betting outcome with much higher bet count (`667` vs `172`). |
| Match-result number of bets | 172 | 667 | Secondary only; not used for promotion. |

## Final Scorecard

Applied the policy checks from `docs/model_acceptance_scorecard_template.md` and `docs/evaluation_policy.md`.

| Field | Value |
|---|---|
| Final decision | `reject` |
| Why | Binary fails the held-out gate because both `log_loss` and `brier` worsened on `2526`; multiclass fails both the held-out gate and the latest-fold guardrail because `2425` and `2526` both regressed on the primary metric. |
| Risks | The candidate increases model dependence on sparse-feature patterns, worsens held-out calibration, and changes betting behavior sharply without scoring gains. |
| Follow-up experiment | If this family is revisited, constrain indicators to high-missingness feature subsets instead of all continuous features. |

## Notes For Promotion Review

- Final status: **reject**.
- Evidence looks **weak** for promotion despite small weighted-CV gains.
- Missing evidence: multi-seed stability was intentionally skipped because the held-out gate already failed.
- `training/configs/main_models/baselines.json` was left unchanged.
- No PR was opened because rejected experiments do not advance.
- The canonical held-out run was executed via `uv run python -m training.train_all_models`; the latest candidate numbers were written to `artifacts/models/latest_main_model_metrics.json`.
