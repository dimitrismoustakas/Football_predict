# preprocess_prod.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import polars as pl

from features_lib import (
    rename_and_cast, build_long, compute_historical_rolling_features, build_match_level_with_rolling_features
)

# Inputs
HIST_GLOB = "data/understat/*/*/matches.parquet"
SEASON_GAMES_PARQUET = Path("data/prod/season_games.parquet")   # produced by main
OUTPUT_DIR = Path("data/prod")
OUTPUT_PARQUET = OUTPUT_DIR / "features_season.parquet"


def _ensure_league_id(df: pl.LazyFrame) -> pl.LazyFrame:
    # Some Understat exports have 'league_id'; else derive a stable id from 'league'
    cols = df.collect_schema().names()
    if "league_id" in cols:
        return df
    if "league" in cols:
        return df.with_columns(
            pl.col("league").str.replace_all(r"\s+", "_").str.to_lowercase().alias("league_id")
        )
    return df.with_columns(pl.lit(None).alias("league_id"))


def _prepare_history(now_utc: Optional[datetime] = None) -> pl.LazyFrame:
    """
    Read historical finished matches and return a normalized LazyFrame limited to matches in the past.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    lf = pl.scan_parquet(HIST_GLOB)
    lf = rename_and_cast(lf)
    lf = _ensure_league_id(lf)

    # Limit to matches strictly before 'now' to prevent leakage
    if "date" in lf.collect_schema().names():
        lf = lf.filter(pl.col("date") < pl.lit(now_utc))

    return lf


def _season_games_to_match_base(season_games_df: pd.DataFrame) -> pl.LazyFrame:
    """
    Convert the season_games parquet to a 'base_matches' frame compatible with build_match_level.
    """
    df = season_games_df.copy()

    # derive league_id if needed
    if "league_id" not in df.columns:
        df["league_id"] = df["league"].astype(str).str.lower().str.replace(r"\s+", "_", regex=True)

    # canonical "date" for joins & ordering: use UTC (no leakage)
    df["date"] = pd.to_datetime(df["kickoff"], utc=True)

    # Ensure all required columns are present, filling with nulls if necessary
    required_cols = {
        "match_id", "league_id", "league", "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "home_xg", "away_xg", "home_shots", "away_shots",
        "home_sot", "away_sot", "home_deep", "away_deep", "home_ppda", "away_ppda"
    }
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    base = pl.from_pandas(df)
    return base.lazy()


def build_prod_features(season_games_df: pd.DataFrame) -> pl.DataFrame:
    hist = _prepare_history()
    base_matches = _season_games_to_match_base(season_games_df)

    # Define a common schema
    common_schema = {
        "match_id": pl.Utf8,
        "league_id": pl.Categorical,
        "league": pl.Categorical,
        "season": pl.Categorical,
        "date": pl.Datetime("us", "UTC"),
        "home_team": pl.Categorical,
        "away_team": pl.Categorical,
        "home_goals": pl.Int32,
        "away_goals": pl.Int32,
        "home_xg": pl.Float64,
        "away_xg": pl.Float64,
        "home_shots": pl.Float64,
        "away_shots": pl.Float64,
        "home_sot": pl.Float64,
        "away_sot": pl.Float64,
        "home_deep": pl.Float64,
        "away_deep": pl.Float64,
        "home_ppda": pl.Float64,
        "away_ppda": pl.Float64,
    }

    # Select and cast columns
    hist = hist.select(list(common_schema.keys())).cast(common_schema)
    base_matches = base_matches.select(list(common_schema.keys())).cast(common_schema)

    # Combine historical and current season games to compute rolling features correctly
    all_games = pl.concat([
        hist,
        base_matches
    ])

    # Build long/historical features per team within league+season
    long_df = build_long(all_games)

    # Calculate historical rolling features
    rolling_feats = compute_historical_rolling_features(long_df)

    # Join rolling features to the base matches
    final = build_match_level_with_rolling_features(base_matches, rolling_feats)

    return final.collect()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SEASON_GAMES_PARQUET.exists():
        raise FileNotFoundError(f"Expected season games parquet at: {SEASON_GAMES_PARQUET}")

    season_games_df = pd.read_parquet(SEASON_GAMES_PARQUET)
    out = build_prod_features(season_games_df)

    out.write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}  (rows={out.height})")


if __name__ == "__main__":
    main()