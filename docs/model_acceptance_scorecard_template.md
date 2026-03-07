# Model Acceptance Scorecard Template

## Experiment Metadata

| Field | Value |
|---|---|
| Experiment ID | |
| Date | |
| Task | `binary` / `multiclass` / both |
| Change summary | |
| Compared against | current champion |
| Status | `accept` / `reject` / `challenger` |

## Proper Scoring Metrics

### Rolling CV

| Split | Champion primary | Candidate primary | Delta | Pass? |
|---|---:|---:|---:|---|
| Fold 1 oldest | | | | |
| Fold 2 middle | | | | |
| Fold 3 latest | | | | |
| Weighted mean | | | | |

### Secondary Metric

| Split | Champion secondary | Candidate secondary | Delta | Pass? |
|---|---:|---:|---:|---|
| Fold 1 oldest | | | | |
| Fold 2 middle | | | | |
| Fold 3 latest | | | | |
| Weighted mean | | | | |

### Held-Out Season

| Metric | Champion | Candidate | Delta | Pass? |
|---|---:|---:|---:|---|
| Primary | | | | |
| Secondary | | | | |

## Stability

| Seed | Champion primary | Candidate primary | Delta |
|---|---:|---:|---:|
| 42 | | | |
| 43 | | | |
| 44 | | | |
| Mean | | | |

## Slice Checks

| Slice | Result | Notes |
|---|---|---|
| League breakdown | pass / fail | |
| Latest CV fold | pass / fail | |
| Draw slice if multiclass | pass / fail | |
| Market-confidence deciles | pass / fail | |
| Calibration | pass / fail | |

## Betting Diagnostics

| Diagnostic | Champion | Candidate | Notes |
|---|---:|---:|---|
| Total profit | | | |
| ROI | | | |
| Number of bets | | | |
| Daily-budget metrics | | | |

## Decision Rule

Mark `accept` only if all are true:

- weighted rolling-CV primary metric improves
- at least 2 of 3 folds improve on the primary metric
- latest fold is not materially worse
- secondary proper metric is non-worse
- held-out season is not clearly worse
- no severe slice collapse appears
- seed stability check passes

When evidence is contradicting between match_result and over_under, choose what is best for the match_result models. The over_under model is not that imporant.

Otherwise:

- mark `challenger` if CV is clearly better but held-out is inconclusive
- mark `reject` if the candidate fails any critical guardrail

## Decision Notes

| Field | Value |
|---|---|
| Final decision | |
| Why | |
| Risks | |
| Follow-up experiment | |

