# Football Prediction Pipeline - AI Agent Instructions

## Project overview
This repo trains and serves two Torch models:
- match result (`Home/Draw/Away`)
- over/under 2.5 goals

The canonical training/evaluation loop is fixed and should be the default path for any branch experiment.

## Canonical workflow

### Feature pipeline
- Training features are built by `preprocessing/build_understat_features.py`
- Shared feature logic lives in `preprocessing/feature_engineering.py`
- Production features are built by `prod_run/build_prod_features.py`

### Main training entry points
- `training/train_main_model.py` trains one canonical model
- `training/train_all_models.py` trains both canonical models
- Frozen model hyperparameters live in `training/configs/main_models/`

### Evaluation protocol
Always preserve this unless the user explicitly asks to change it:
- rolling expanding-window CV for model selection
- fixed final validation season for epoch selection
- fixed held-out latest season for acceptance

Do not treat ad hoc research sweeps as the canonical merge gate.

## Research workflow
Keep the repo surface small.

If an experiment needs custom search or analysis code, prefer a narrow branch-specific helper over keeping many permanent research scripts in the repo.

## Production
- `prod_run/pipeline.py` now loads both canonical model bundles from `artifacts/models/`
- `prod_run/fetch_odds.py` fetches both totals and match-result prices
- Generated model bundles under `artifacts/models/` are runtime outputs and should not be committed

## Betting diagnostics
Use proper scoring metrics as the primary quality gate.

Betting diagnostics are secondary and use daily fixed-budget logic:
- fixed budget per day
- split equally across positive-EV bets for that day
- no minimum-games-per-day filter

Avoid reintroducing Sharpe-based optimization as the main acceptance rule.

## Coding conventions
- Keep code simple and direct
- Prefer Polars for feature engineering
- Use `uv` for Python commands
- Python files should use tabs for indentation in this repo
- Avoid defensive wrappers that add no value
- Backwards compatibility is not required during cleanup
