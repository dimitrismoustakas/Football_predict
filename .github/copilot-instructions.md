# Football Prediction Pipeline - AI Agent Instructions

## Project Overview
This is a machine learning system for predicting football match outcomes (Win/Draw/Loss) and Over/Under 2.5 goals. The pipeline scrapes historical match data, engineers rolling features, trains sklearn models, and generates predictions for upcoming fixtures.

## Architecture & Data Flow

### 1. Data Collection (`data_collection/`)
- **Understat scraping** (`collect_understat.py`): Downloads historical match data (stats and schedule) into `data/understat/<league>/<season>/matches.parquet`. Uses `soccerdata` to fetch data.
- **Elo scraping** (`scrapp_elo.py`): Fetches ClubElo ratings at periodic snapshots, writes to `data/eloscores/`.
- **Salary data** (`Capology_payroll_scrapping.py`): Scrapes team payroll data (stored but not yet integrated into features).

### 2. Feature Engineering (`preprocessing/`)
- **`feature_engineering.py`**: Shared library containing core feature engineering logic (Polars-based).
  - `build_long`: Transforms wide match data → long format.
  - `compute_rolling_features`: Computes rolling means/sums (windows: 3, 5, 10).
  - `build_match_level`: Joins features back to match level.
- **`build_understat_features.py`**: Training feature pipeline.
  - Reads all historical `data/understat/*/*/matches.parquet`.
  - Uses `feature_engineering` module to generate features.
  - Output: `data/training/understat_df.parquet`.

### 3. Model Training (`training/`)
- **`models_training.py`**:
  - Loads `data/training/understat_df.parquet`.
  - Filters: both teams must have ≥5 prior games.
  - Feature selection: `ovr__*__r5__h` and `ovr__*__r5__a`.
  - Trains RandomForest (Result) and LogisticRegression (Over/Under).
  - Outputs models to `data/models/`.

### 4. Production Inference (`prod_run/`)
- **Workflow**:
  1. **`pipeline.py`**: The main entry point.
     - Calls `build_prod_features.py` to fetch data and generate features.
     - Loads trained models (`result_model.joblib`, `over_model.joblib`).
     - Generates predictions and probabilities for upcoming games (Today + 3 days).
     - Calculates implied odds (1/probability).
     - Saves results to `data/predictions/upcoming_predictions.csv`.
     - Sends email with CSV attachment if configured in `.env`.
  2. **`build_prod_features.py`**: 
     - Fetches current season data (completed matches + upcoming schedule) using `soccerdata`.
     - Computes rolling features using `feature_engineering` logic.
     - Output: `data/prod/features_season.parquet`.

## Critical Conventions

### Polars-First Design
- **All feature engineering uses Polars** for efficiency.
- Shared logic in `preprocessing/feature_engineering.py`.

### Date Handling
- Dates stored as **Datetime**.
- Production filters for `date >= today`.

### Rolling Features Naming Convention
```
<scope>__<stat>__r<window>__<side>
ovr__xg_for__r5__h
```
- **Scopes**: `ovr`, `home`, `away`
- **Windows**: r3, r5, r10
- **Sides**: `__h` (home), `__a` (away)

### Data Leakage Prevention
- **Always use `.shift(1)`** when computing rolling features.
- Window calculations grouped by `["league_id", "season", "team"]`.

## Developer Workflows

### Training New Models
```powershell
# 1. Scrape historical data
python data_collection/collect_understat.py

# 2. Build training features
python preprocessing/build_understat_features.py

# 3. Train models
python training/models_training.py --parquet data/training/understat_df.parquet
```

### Production Predictions
```powershell
# Run the end-to-end pipeline (fetch data -> features -> predict -> email)
uv run python prod_run/pipeline.py
```
*Requires `.env` file with `EMAIL_USER`, `EMAIL_PASS`, `EMAIL_RECIPIENTS`.*

### Debugging Feature Mismatches
- If prod features don't match training: compare `features_lib.py` vs `build_understat_features.py`
- Check `data/models/features_and_meta.json` for expected feature list
- Use `inspect_features.py` to verify null counts and column names

## Key Files Reference
- **Production Pipeline**: `prod_run/pipeline.py` (orchestrator), `prod_run/build_prod_features.py` (features)
- **Feature engineering logic**: `preprocessing/build_understat_features.py` (training), `prod_run/features_lib.py` (prod)
- **Scraper configs**: Leagues in `LEAGUES_DEFAULT` (hardcoded in scrapers)
- **Model artifacts**: `data/models/features_and_meta.json` (source of truth for feature list)
- **Dependencies**: `pyproject.toml` (polars>=1.34.0, scikit-learn>=1.7.2, scraperfc>=3.4.0, soccerdata>=1.8.2)

## Known Gaps
- `current.py` uses FotMob adapter (different from training's Understat data)
- `test_scripts/` contains exploratory notebooks but no unit tests
- Salary data collected but not integrated into features


## Coding Standards
- Avoid needless defensive coding; it is prefered to let errors raise naturally for visibility.
- Do not create needless abstractions; keep code simple and straightforward.
- Avoid wrapper functions that do not add value.
- Always keep in mind that we use uv instead of pip and the corresponding venv.
- We are in development mode, we don't need about backwards compatibility, we can break stuff as needed.