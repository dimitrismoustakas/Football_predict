# Standard Experiment Report

## Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID | |
| Date | |
| Proposal ID | |
| Proposal title | |
| Task | `binary` / `multiclass` / both |
| Branch | |
| Status | `pending-promotion-review` |

## Proposal Chosen

> Paste the exact proposal text from `improvement_proposals.md`.

## Why This Idea Was Chosen

- 
- 

## Implementation Summary

- 
- 

## Files Changed

- 
- 

## Commands Run

1. 
2. 
3. 

## Evidence Sources

- `artifacts/models/latest_main_model_metrics.json`
- add any MLflow run IDs, tables, temp outputs, or custom analysis files used

## Proper Scoring Metrics

### Rolling CV

| Split | Champion primary | Candidate primary | Delta | Champion secondary | Candidate secondary | Delta |
|---|---:|---:|---:|---:|---:|---:|
| Fold 1 oldest | | | | | | |
| Fold 2 middle | | | | | | |
| Fold 3 latest | | | | | | |
| Weighted mean | | | | | | |

### Held-Out Season

| Metric | Champion | Candidate | Delta |
|---|---:|---:|---:|
| Primary | | | |
| Secondary | | | |

## Stability

| Seed | Candidate primary | Delta vs champion |
|---|---:|---:|
| 42 | | |
| 43 | | |
| 44 | | |
| Mean | | |

## Slice Checks

| Slice | Result | Notes |
|---|---|---|
| League breakdown | pass / fail / not-run | |
| Latest CV fold | pass / fail / not-run | |
| Draw slice if multiclass | pass / fail / not-run | |
| Market-confidence deciles | pass / fail / not-run | |
| Calibration | pass / fail / not-run | |

## Betting Diagnostics

| Diagnostic | Champion | Candidate | Notes |
|---|---:|---:|---|
| Total profit | | | |
| ROI | | | |
| Number of bets | | | |
| Daily-budget metrics | | | |

## Scorecard Draft

Reuse the structure and checks from `docs/model_acceptance_scorecard_template.md`, but leave the decision unresolved here.

| Field | Value |
|---|---|
| Final decision | `pending-promotion-review` |
| Why | Raw evidence only. Do not promote here. |
| Risks | |
| Follow-up experiment | |

## Notes For Promotion Review

- State whether the evidence looks strong, mixed, or weak.
- Call out any missing evidence.
- Explicitly mention whether `training/configs/main_models/baselines.json` was left unchanged.
