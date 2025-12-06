#!/usr/bin/env python3
"""
Train two simple models on Understat-derived rolling features produced by build_understat_features.py.

- Features: ONLY overall ("ovr__*") rolling-mean features for window r5, for both home and away sides.
  (Exclude any "__sum__" engineered columns and exclude the games counters from the feature set.)
- Keep only matches where BOTH teams have at least 5 prior games (i.e., ovr__games__r5__h >= 5 and ...__a >= 5).
- Exclude columns that leak the outcome (goals, xg of the match, targets, etc.).
- Seasons: use all seasons except the most recent (current) and the immediately previous for training.
  Evaluate on the previous season only; exclude the current season entirely.
- Models:
    1) Multiclass match result (H/D/A) → RandomForestClassifier
    2) Over/Under (>2.5) → LogisticRegression (binary)
- Outputs:
    * Metrics printed to stdout and also written under data/models/metrics_*.csv
    * Pickled sklearn models saved under data/models/
    * Feature list saved to JSON for reproducibility

Requirements:
    polars>=1.21.0, numpy, pandas, scikit-learn, joblib

Usage:
    python train_understat_models.py [--parquet data/training/understat_df.parquet]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
from joblib import dump


DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_frame(parquet_path: Path) -> pl.DataFrame:
    lf = pl.scan_parquet(str(parquet_path))
    # Select only needed columns lazily to reduce memory
    # We'll materialize after deriving targets and filters.
    df = lf.collect()
    return df


def select_feature_columns(df: pl.DataFrame) -> list[str]:
    """Return ONLY overall rolling-mean r5 features for both sides.
    Includes columns like: ovr__xg_for__r5__h, ovr__shots_against__r5__a, etc.
    Excludes any column containing "__sum__" and the games counters.
    """
    cols = df.columns
    feat_cols = [
        c
        for c in cols
        if c.startswith("ovr__")
        and "__r5" in c
        and not "__sum__" in c
        and not c.startswith("ovr__games__")
        and (c.endswith("__h") or c.endswith("__a"))
    ]
    return sorted(feat_cols)


def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
    # Ensure both teams have >=5 prior games
    need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for history filter: {missing}")
    return df.filter((pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5))


def train_test_season_splits(df: pl.DataFrame) -> tuple[list[str], str, str]:
    """Determine train seasons (all except last two), previous season (eval), and current season (excluded)."""
    seasons = (
        df.select(pl.col("season").cast(pl.Utf8)).unique().sort(by="season").to_series().to_list()
        if "season" in df.columns
        else []
    )
    if len(seasons) < 3:
        raise ValueError("Need at least 3 seasons to create the requested splits.")
    current = seasons[-1]
    previous = seasons[-2]
    train = seasons[:-2]
    return train, previous, current


def make_targets(df: pl.DataFrame) -> pl.DataFrame:
    """Add targets: Over (binary) is already present; add match_result as 'H'/'D'/'A'."""
    need = ["home_goals", "away_goals"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Required column missing for targets: {c}")
    df = df.with_columns(
        pl.when(pl.col("home_goals") > pl.col("away_goals"))
        .then(pl.lit("H"))
        .when(pl.col("home_goals") < pl.col("away_goals"))
        .then(pl.lit("A"))
        .otherwise(pl.lit("D"))
        .alias("match_result")
    )
    if "Over" not in df.columns:
        raise ValueError("'Over' column not found; ensure you used build_match_level().")
    return df


def prepare_matrices(df: pl.DataFrame, feature_cols: list[str], season_list: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    part = df.filter(pl.col("season").cast(pl.Utf8).is_in(season_list))
    # Drop rows with any nulls in features
    part = part.drop_nulls(subset=feature_cols)
    # Build pandas matrices
    X = part.select(feature_cols).to_pandas()
    y_res = part.select("match_result").to_pandas().iloc[:, 0]
    y_over = part.select("Over").to_pandas().iloc[:, 0].astype(int)
    return X, y_res, y_over


def main(parquet_path: Path):
    print(f"Loading: {parquet_path}")
    df = load_frame(parquet_path)

    # Filter to rows with sufficient history and create targets
    df = filter_min_history(df)
    df = make_targets(df)

    # Decide seasons
    train_seasons, prev_season, curr_season = train_test_season_splits(df)
    print(f"Train seasons: {train_seasons}")
    print(f"Validation (previous) season: {prev_season}")
    print(f"Excluded current season: {curr_season}")

    # Feature selection
    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No feature columns selected. Check that r5 overall features exist.")

    # Prepare matrices
    X_train, y_res_train, y_over_train = prepare_matrices(df, feature_cols, train_seasons)
    X_val, y_res_val, y_over_val = prepare_matrices(df, feature_cols, [prev_season])

    # -------- Model 1: RandomForest for H/D/A --------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    rf.fit(X_train, y_res_train)
    res_val_pred = rf.predict(X_val)
    res_val_proba = rf.predict_proba(X_val)

    res_acc = accuracy_score(y_res_val, res_val_pred)
    res_f1m = f1_score(y_res_val, res_val_pred, average="macro")

    print("\n[Result RF] Previous season metrics:")
    print(f"Accuracy: {res_acc:.4f}")
    print(f"Macro F1: {res_f1m:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_res_val, res_val_pred, labels=rf.classes_))

    # -------- Model 2: Logistic Regression for Over ---------
    lr = LogisticRegression(max_iter=2000, solver="lbfgs")
    lr.fit(X_train, y_over_train)
    over_val_proba = lr.predict_proba(X_val)[:, 1]
    over_val_pred = (over_val_proba >= 0.5).astype(int)

    over_acc = accuracy_score(y_over_val, over_val_pred)
    over_brier = brier_score_loss(y_over_val, over_val_proba)
    try:
        over_logloss = log_loss(y_over_val, np.c_[1 - over_val_proba, over_val_proba])
    except ValueError:
        over_logloss = float("nan")

    frac_pos, mean_pred = calibration_curve(y_over_val, over_val_proba, n_bins=10, strategy="quantile")

    print("\n[Over LR] Previous season metrics:")
    print(f"Accuracy: {over_acc:.4f}")
    print(f"Brier score: {over_brier:.4f}")
    print(f"Log loss: {over_logloss:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_over_val, over_val_pred))

    # -------- Persist artifacts --------
    # Save models
    rf_path = MODELS_DIR / "rf_result_model.joblib"
    lr_path = MODELS_DIR / "lr_over_model.joblib"
    dump(rf, rf_path)
    dump(lr, lr_path)

    # Save feature list and metadata
    meta = {
        "parquet": str(parquet_path),
        "feature_cols": feature_cols,
        "train_seasons": train_seasons,
        "validation_season": prev_season,
        "excluded_current_season": curr_season,
        "filters": {
            "min_history_games": 5,
            "features": "ovr__*__r5 for both sides; exclude __sum__ and games",
        },
        "models": {
            "result_model": rf_path.name,
            "over_model": lr_path.name,
        },
    }
    (MODELS_DIR / "features_and_meta.json").write_text(json.dumps(meta, indent=2))

    # Save metrics CSVs
    metrics_res = pd.DataFrame(
        {
            "metric": ["accuracy", "macro_f1"],
            "value": [res_acc, res_f1m],
        }
    )
    metrics_res.to_csv(MODELS_DIR / "metrics_result_prev_season.csv", index=False)

    # For Over: include calibration table
    metrics_over = pd.DataFrame(
        {
            "metric": ["accuracy", "brier", "logloss"],
            "value": [over_acc, over_brier, over_logloss],
        }
    )
    metrics_over.to_csv(MODELS_DIR / "metrics_over_prev_season.csv", index=False)

    calib_df = pd.DataFrame({
        "bin_fraction_positive": frac_pos,
        "bin_mean_predicted": mean_pred,
    })
    calib_df.to_csv(MODELS_DIR / "calibration_over_prev_season.csv", index=False)

    # Also persist the raw class probabilities for inspection
    proba_res_df = pd.DataFrame(res_val_proba, columns=rf.classes_)
    proba_res_df.to_csv(MODELS_DIR / "result_rf_probabilities_prev_season.csv", index=False)

    over_proba_df = pd.DataFrame({"proba_over": over_val_proba})
    over_proba_df.to_csv(MODELS_DIR / "over_lr_probabilities_prev_season.csv", index=False)

    print("\nArtifacts saved:")
    print(f" - {rf_path}")
    print(f" - {lr_path}")
    print(f" - {MODELS_DIR / 'features_and_meta.json'}")
    print(f" - {MODELS_DIR / 'metrics_result_prev_season.csv'}")
    print(f" - {MODELS_DIR / 'metrics_over_prev_season.csv'}")
    print(f" - {MODELS_DIR / 'calibration_over_prev_season.csv'}")
    print(f" - {MODELS_DIR / 'result_rf_probabilities_prev_season.csv'}")
    print(f" - {MODELS_DIR / 'over_lr_probabilities_prev_season.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    args = parser.parse_args()
    main(args.parquet)
