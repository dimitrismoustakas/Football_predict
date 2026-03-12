# Football Prediction Pipeline - AI Agent Instructions

## Project overview
This repo trains and serves one Torch model:
- match result (`Home/Draw/Away`)

The canonical training and evaluation loop is fixed and should be the default path for any branch experiment.

## Canonical workflow

### Feature pipeline
- Training features are built by `preprocessing/build_understat_features.py`
- Shared feature logic lives in `preprocessing/feature_engineering.py`
- Production features are built by `prod_run/build_prod_features.py`

### Main training entry point
- `training/train_main_model.py` trains the canonical model
- Frozen hyperparameters live in `training/configs/main_models/`

### Evaluation protocol
Always preserve this unless the user explicitly asks to change it:
- use `training/configs/main_models/evaluation.json` as the source of truth
- rolling expanding-window CV mean `log_loss` is the decision metric
- the last pre-test season is reserved for epoch selection
- use the latest comparable row in `artifacts/experiment_metrics/result_main_runs.tsv` as the default comparison point

## Production
- `prod_run/pipeline.py` loads the canonical model bundle from `artifacts/models/`
- `prod_run/fetch_odds.py` fetches match-result prices
- Generated model bundles under `artifacts/models/` are runtime outputs and should not be committed
- `artifacts/experiment_metrics/result_main_runs.tsv` is the single experiment ledger and should be kept append-only

## GitHub tooling
- GitHub CLI is available at `C:/Program Files/GitHub CLI/gh.exe` on this machine even if `gh` is not on `PATH`
- use it for branch/PR actions when needed

## General experiment guidance
- Keep the repo surface lean.
- Prefer editing canonical files over adding permanent experiment scaffolding.
- If experiment-specific search or analysis code is needed, keep it narrow and branch-local unless it clearly belongs in the canonical path.
- Use `artifacts/experiment_metrics/result_main_runs.tsv` as the default experiment ledger and keep it append-only.
- Do not add report workflows or extra experiment registries unless the user explicitly asks for them.

## Betting diagnostics
Use proper scoring metrics as the primary quality gate.
Betting diagnostics are secondary.

## Coding conventions
- Keep code simple and direct
- Prefer Polars for feature engineering
- Use `uv` for Python commands
- Python files should use tabs for indentation in this repo
- Avoid defensive wrappers that add no value
- Backwards compatibility is not required during cleanup
