# Football Predict

PyTorch football match-result forecasting pipeline for `Home / Draw / Away`.

The project covers data collection, feature building, rolling-window training, production prediction, positive-EV diagnostics, Kelly stake sizing, and a static HTML report.

Live report: https://dimitrismoustakas.github.io/Football_predict/

## What is included

- Data collectors for match history, schedules, Understat features, ClubElo ratings, and market odds.
- A reproducible feature pipeline with source-controlled team-name mappings.
- A PyTorch result model trained with a fixed expanding-window evaluation protocol.
- Production code that writes match probabilities, model picks, value diagnostics, stake suggestions, and an HTML report.
- Tests covering odds parsing, portfolio logic, production outputs, and the training loop.

## Repo map

- `data_collection/`: raw data refresh scripts.
- `preprocessing/`: feature engineering and canonical mapping utilities.
- `training/`: model definitions, training loop, inference helpers, and evaluation code.
- `training/configs/main_models/`: source-controlled model, feature, and evaluation configs.
- `prod_run/`: production feature build, odds fetch, prediction pipeline, and report generation.
- `site/`: static report output for GitHub Pages or another static host.
- `tests/`: regression tests for core training and production behavior.
- `artifacts/mappings/`: tracked canonical-name mappings.

Generated datasets and runtime model files are intentionally not committed.

## Workflow

Install dependencies:

```bash
uv sync
```

Refresh data:

```bash
uv run python data_collection/collect_understat.py
uv run python data_collection/collect_full_schedule.py
uv run python data_collection/collect_match_history.py
uv run python data_collection/collect_elo.py
uv run python preprocessing/build_understat_features.py
```

Train the canonical result model:

```bash
uv run python training/train_main_model.py
```

Run the production prediction pipeline:

```bash
uv run python prod_run/pipeline.py
```

Run tests:

```bash
uv run pytest
```

## Evaluation protocol

The main training loop uses the protocol in `training/configs/main_models/evaluation.json`.

Current policy:

- compare candidate models by rolling expanding-window CV mean `log_loss`;
- reserve the last pre-test season for epoch selection;
- keep the configured test season held out from branch acceptance decisions;
- record dataset fingerprints and per-season row counts in generated run outputs, so comparisons across data refreshes are explicit.

The implementation entry point is `training/train_main_model.py`.

## Production output

`prod_run/pipeline.py` builds production features, fetches match-result odds, loads the canonical model, and writes predictions under `data/predictions/`.

The output includes:

- result probabilities and model pick;
- market-implied probability and positive-EV diagnostics;
- Kelly-style stake fields for recommended positive-EV picks;
- an interactive HTML report at `data/predictions/upcoming_predictions.html`;
- an optional hosted copy at `site/index.html`.

Useful production settings:

- `FIXED_BUDGET`
- `KELLY_FRACTION`
- `MIN_BET_AMOUNT`
- `PREDICTION_WINDOW_DAYS`
- `PUBLISH_STATIC_REPORT`
- `STATIC_REPORT_PATH`
- `REPORT_PUBLIC_URL`

## Generated files

Ignored or generated paths include:

- `data/`
- `artifacts/models/`
- `downloaded_files/`
