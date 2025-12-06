# build_understat_features.py
# Requires: polars>=1.21.0
from pathlib import Path
import sys
import os

# Add project root to path to allow imports
sys.path.append(os.getcwd())

import polars as pl
from preprocessing.feature_engineering import (
    rename_and_cast,
    build_long,
    compute_rolling_features,
    build_match_level,
)

# ---------- Config ----------
INPUT_GLOB = "data/understat/*/*/matches.parquet"
OUTPUT_DIR = Path("data/training")
OUTPUT_PARQUET = OUTPUT_DIR / "understat_df.parquet"


def main():
    pl.enable_string_cache()

    # Scan & normalize
    lf = pl.scan_parquet(INPUT_GLOB)
    lf = rename_and_cast(lf)

    # Base match columns we need
    base_needed = [
        "match_id",
        "league_id",
        "league",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_npxg",
        "away_npxg",
        # "home_shots",
        # "away_shots",
        # "home_sot",
        # "away_sot",
        "home_deep",
        "away_deep",
        "home_ppda",
        "away_ppda",
    ]
    
    schema = lf.collect_schema()
    have = set(schema.names())
    base_cols = [c for c in base_needed if c in have]
    base_matches = lf.select(base_cols)

    # Long spine
    long_df = build_long(base_matches)

    # Rolling features (within league+season; shift(1) prevents leakage)
    long_feats = compute_rolling_features(long_df)

    # Rejoin to match level and write
    final_df = build_match_level(base_matches, long_feats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.collect().write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
