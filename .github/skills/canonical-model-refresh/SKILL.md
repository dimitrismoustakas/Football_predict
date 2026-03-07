---
name: canonical-model-refresh
description: Refresh football data, rebuild features, retrain and evaluate the canonical models, compare candidates against accepted baselines, and update the source-controlled baseline log. Use this when asked to refresh the pipeline, assess a new canonical candidate, or update model baselines.
---

# Canonical model refresh workflow

Use this skill for the repo's repeatable canonical model maintenance flow. Prefer this workflow over research sweeps unless the user explicitly asks for exploratory work.

## Non-negotiable repo rules

- Preserve the canonical evaluation protocol unless the user explicitly asks to change it:
  - rolling expanding-window CV for model selection
  - fixed final validation season for epoch selection
  - fixed held-out latest season for acceptance
- Treat `training/train_main_model.py` and `training/train_all_models.py` as the canonical training entry points.
- Frozen source-controlled hyperparameters live in `training/configs/main_models/`.
- Runtime-generated model bundles and latest metrics under `artifacts/models/` are outputs, not source assets.
- Never commit generated model bundles from `artifacts/models/`.
- Proper scoring metrics are the primary gate:
  - over/under: `log_loss`, `brier`
  - match result: `log_loss`, `rps`
- Betting diagnostics are secondary only, using fixed daily budget logic. Do not optimize around Sharpe-style objectives.

## Core files to know

- `.github/copilot-instructions.md` — repository-wide rules and workflow summary
- `README.md` — command summary and repo hygiene notes
- `preprocessing/build_understat_features.py` — canonical training feature builder
- `preprocessing/feature_engineering.py` — shared feature logic
- `training/train_main_model.py` — canonical single-model train/eval entry point
- `training/train_all_models.py` — canonical two-model train/eval entry point
- `training/configs/main_models/over_under.json` — frozen over/under config
- `training/configs/main_models/result.json` — frozen result config
- `training/configs/main_models/baselines.json` — accepted baseline history
- `artifacts/models/latest_main_model_metrics.json` — latest evaluated candidate metrics to compare against baselines
- `prod_run/build_prod_features.py` and `prod_run/pipeline.py` — production feature/prediction flow

## Standard sequence

1. Refresh raw data if the task is a full pipeline refresh:
   - `uv run python data_collection/collect_understat.py`
   - `uv run python data_collection/collect_full_schedule.py`
   - `uv run python data_collection/collect_match_history.py`
   - `uv run python data_collection/collect_elo.py`
2. Rebuild training features:
   - `uv run python preprocessing/build_understat_features.py`
3. Retrain/evaluate canonical models:
   - both models: `uv run python training/train_all_models.py`
   - one model only: run `training/train_main_model.py` with the appropriate task configuration already expected by the repo
4. Read `artifacts/models/latest_main_model_metrics.json` and compare the candidate with the latest accepted entry in `training/configs/main_models/baselines.json`.
5. Only if the candidate is accepted, append a new entry to `training/configs/main_models/baselines.json` with:
   - date
   - kind
   - short description of the change
   - per-model config path, epoch selection season, held-out season, best epoch, best validation loss, validation metrics, and test metrics
6. If the task touches production paths, confirm code still points to `artifacts/models/` and `artifacts/mappings/` rather than old `data/models` or `data/mappings` locations.

## Acceptance guidance

When deciding whether to accept a candidate as the new canonical baseline:

- Start with the task's proper scoring metrics, not profit.
- Check that the held-out latest season remains untouched during model selection.
- Confirm that any baseline update is tied to a real model/config/data change and document it briefly.
- Keep the first baseline entry intact; append new entries instead of rewriting history.
- If results are mixed, prefer preserving the current baseline and summarizing the trade-offs.

## Editing guidance

When making code changes during this workflow:

- Keep Python files tab-indented.
- Prefer simple, direct changes.
- Prefer Polars for feature engineering work.
- Do not add backwards-compatibility wrappers unless explicitly requested.
- Avoid changing research scripts unless the user specifically asks for research/sweep work.

## What to report back

Summaries should usually include:

- whether data was refreshed
- whether features were rebuilt
- which canonical training entry point was used
- where the candidate metrics were read from
- whether `training/configs/main_models/baselines.json` was updated
- whether any production path references were corrected
- any follow-up validation still worth running
