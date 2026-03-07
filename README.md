# Football Predict

Torch-based football prediction pipeline for two core tasks:
- match result (`Home/Draw/Away`)
- over/under 2.5 goals

## Canonical workflow

The repo now has one stable training/evaluation loop for both production models.

### Training data
1. Collect/update raw data.
2. Build training features into `data/training/understat_df.parquet`.
3. Train the canonical models with a fixed evaluation protocol.

### Evaluation protocol
The main training loop is frozen and should be used for branch comparisons:
- rolling expanding-window CV for model selection
- fixed final validation season for epoch selection
- fixed held-out latest season for acceptance

Decision rules for iterative promotion live in `docs/evaluation_policy.md`.
Use `docs/model_acceptance_scorecard_template.md` when deciding whether a challenger should replace the current champion.

This is implemented in:
- `training/train_main_model.py`
- `training/train_all_models.py`

### Main model configs
Frozen source-controlled model configs live in:
- `training/configs/main_models/over_under.json`
- `training/configs/main_models/result.json`

Generated runtime artifacts are written to `artifacts/models/` and are not meant to be committed.

## Commands

### Data refresh
- `uv run python data_collection/collect_understat.py`
- `uv run python data_collection/collect_full_schedule.py`
- `uv run python data_collection/collect_match_history.py`
- `uv run python data_collection/collect_elo.py`
- `uv run python preprocessing/build_understat_features.py`

### Train canonical models
- Over/under only: `uv run python training/train_main_model.py`
- Result only: `TASK_TYPE=multiclass uv run python training/train_main_model.py`
- Both models: `uv run python training/train_all_models.py`

### Optional search / research scripts
These are not the canonical acceptance path. Use them only when intentionally sweeping:
- `training/fixed_arch_sweep.py`
- `training/architecture_search.py`
- `training/result_architecture_search.py`
- `training/feature_selection_search.py`
- `training/analyze_residuals_by_decile.py`

## Production

`prod_run/pipeline.py` builds production features, fetches odds, loads both canonical models, and writes predictions to `data/predictions/`.

The output now includes:
- result probabilities and value side diagnostics
- over/under probabilities
- over/under daily-budget allocation percentages

## Betting diagnostics

Model acceptance should be driven by proper scoring metrics first:
- over/under: `log_loss`, `brier`
- result: `log_loss`, `rps`

Betting metrics are secondary diagnostics. The repo now uses daily fixed-budget evaluation instead of Sharpe-based portfolio selection:
- fixed budget per calendar day
- split equally across all positive-EV bets on that day
- no minimum-games-per-day rule

## Repo hygiene

Tracked source assets:
- code
- mappings under `artifacts/mappings/`
- frozen source configs under `training/configs/main_models/`

Ignored/generated outputs:
- `artifacts/models/`
- `mlruns/`
- `downloaded_files/`
- `data/`
