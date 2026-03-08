# Football Predict

Torch-based football prediction pipeline for one core task: match result (`Home/Draw/Away`).

## Canonical workflow

The repo keeps one stable training and evaluation loop for the production model.

### Training data
1. Collect or refresh raw data.
2. Build training features into `data/training/understat_df.parquet`.
3. Train the canonical result model with the fixed evaluation protocol.

### Evaluation protocol
The main training loop is frozen and should be used for branch comparisons:
- rolling expanding-window CV for model selection
- fixed final validation season for epoch selection
- fixed held-out latest season for acceptance

Decision rules live in `docs/evaluation_policy.md`.
Use `docs/model_acceptance_scorecard_template.md` when deciding whether a challenger should replace the current champion.

This is implemented in:
- `training/train_main_model.py`

### Main model config
The frozen source-controlled model config lives in:
- `training/configs/main_models/result.json`

Generated runtime artifacts are written to `artifacts/models/` and are not meant to be committed.

The canonical trainer supports optional training-recipe fields:
- optimizer: `optimizer_name`, `beta1`, `beta2`, `optimizer_eps`
- scheduler: `scheduler_name`, `scheduler_warmup_epochs`, `scheduler_warmup_start_factor`, `scheduler_min_lr_ratio`, `scheduler_plateau_*`, `onecycle_*`
- final retrain rule: `final_epoch_mode`, `final_epoch_buffer`
- stability: `max_grad_norm`

## Commands

### Data refresh
- `uv run python data_collection/collect_understat.py`
- `uv run python data_collection/collect_full_schedule.py`
- `uv run python data_collection/collect_match_history.py`
- `uv run python data_collection/collect_elo.py`
- `uv run python preprocessing/build_understat_features.py`

### Train canonical model
- `uv run python training/train_main_model.py`

## Production

`prod_run/pipeline.py` builds production features, fetches match-result odds, loads the canonical model, and writes predictions to `data/predictions/`.

The output includes:
- result probabilities
- model pick
- positive-EV result side diagnostics

## Repo hygiene

Tracked source assets:
- code
- mappings under `artifacts/mappings/`
- frozen source config under `training/configs/main_models/`

Ignored or generated outputs:
- `artifacts/models/`
- `mlruns/`
- `downloaded_files/`
- `data/`
