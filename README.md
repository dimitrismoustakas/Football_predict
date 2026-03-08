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
- rolling expanding-window CV mean `log_loss` as the single experiment objective
- fixed final validation season for epoch selection
- fixed held-out latest season for acceptance

Decision rules live in `docs/evaluation_policy.md`.

This is implemented in:
- `training/train_main_model.py`

### Main model config
The frozen source-controlled model config lives in:
- `training/configs/main_models/result.json`
- `training/configs/main_models/result_features.json`

Generated runtime artifacts are written to `artifacts/models/` and are not meant to be committed.

The trainer also appends one row per canonical run to `artifacts/experiment_metrics/result_main_runs.tsv`.

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
- `downloaded_files/`
- `data/`
