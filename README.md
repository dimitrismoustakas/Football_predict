# Football Predict

Torch-based football prediction pipeline for one core task: match result (`Home/Draw/Away`).

## Canonical workflow

The repo keeps one stable training and evaluation loop for the production model.

### Training data
1. Collect or refresh raw data.
2. Build training features into `data/training/understat_df.parquet`.
3. Train the canonical result model with the fixed evaluation protocol.

### Evaluation protocol
The main training loop is frozen and should be used for branch comparisons.
The source-controlled protocol lives in:
- `training/configs/main_models/evaluation.json`

Current rules:
- rolling expanding-window CV mean `log_loss` is the single decision metric
- the last pre-test season is reserved for epoch selection
- the fixed test season is watch-only and not used for branch acceptance
- compare against the latest comparable ledger row and keep iterating when CV `log_loss` improves
- every run records a dataset fingerprint and per-season row counts so comparisons across data refreshes are explicit
- local prescreens should mirror the canonical split as closely as practical: use the epoch-selection season to choose training length, then score the objective folds without appending to the ledger
- do not trust the epoch-selection season alone for close experiment decisions

Decision rules are described in `docs/evaluation_policy.md`.

This is implemented in:
- `training/train_main_model.py`

### Main model config
The frozen source-controlled model config lives in:
- `training/configs/main_models/result.json`
- `training/configs/main_models/result_features.json`
- `training/configs/main_models/evaluation.json`

Generated runtime artifacts are written to `artifacts/models/` and are not meant to be committed.

The trainer appends one row per canonical run to `artifacts/experiment_metrics/result_main_runs.tsv`.
That TSV is the single experiment ledger.

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
- bankroll stake fields for recommended positive-EV picks (`Result_Value_Prob`, `Result_Value_Implied`, `Result_Edge`, `Result_EV`, `Result_Budget_Share`, `Result_Budget_Amount`)
- an interactive HTML report at `data/predictions/upcoming_predictions.html`
- an optional hosted copy at `site/index.html` for static-site deployment

Offline smoke-test command:
- `.venv\Scripts\python.exe prod_run\smoke_test.py`

Bankroll Kelly knobs for production:
- `FIXED_BUDGET` default `100`
- `KELLY_FRACTION` default `0.5`
- `MIN_BET_AMOUNT` default `0.1`
- `PREDICTION_WINDOW_DAYS` default `7`

Static report hosting knobs:
- `PUBLISH_STATIC_REPORT` default `true`
- `STATIC_REPORT_PATH` default `site/index.html`
- `REPORT_PUBLIC_URL` optional public URL used in the email body; when set, the email sends the link instead of attaching the HTML file

Fastest hosted flow:
1. Run `uv run python prod_run/pipeline.py`.
2. Commit the updated `site/index.html`.
3. Push the branch and deploy `site/` with a static host such as GitHub Pages or Cloudflare Pages.
4. Set `REPORT_PUBLIC_URL` so recipients get a URL instead of an attachment.

## Repo hygiene

Tracked source assets:
- code
- mappings under `artifacts/mappings/`
- frozen source config under `training/configs/main_models/`
- the canonical experiment ledger `artifacts/experiment_metrics/result_main_runs.tsv`

Ignored or generated outputs:
- `artifacts/models/`
- `downloaded_files/`
- `data/`
