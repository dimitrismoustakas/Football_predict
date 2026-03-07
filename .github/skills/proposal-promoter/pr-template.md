# Accepted Experiment PR Template

## Summary

Promote proposal `<idea-id>`: `<short-title>`.

## What changed

- 
- 

## Evidence

- Report: `docs/experiment_reports/YYYY-MM-DD_<idea-id>_<slug>.md`
- Baseline manifest updated: `training/configs/main_models/baselines.json`

## Proper scoring deltas

### Rolling CV

- Primary metric delta:
- Secondary metric delta:
- Latest fold guardrail:

### Held-out season

- Primary metric delta:
- Secondary metric delta:

## Slice / stability summary

- League slices:
- Calibration:
- Seed stability:

## Policy outcome

- Final decision: `accept`
- Reason:

## Notes

- Betting diagnostics were reviewed as secondary evidence only.
- Generated model bundles under `artifacts/models/` were not committed.
