from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

import polars as pl
import json
from preprocessing.feature_engineering import (
    rename_and_cast,
    build_long,
    compute_rolling_features,
    build_match_level,
)
from preprocessing.odds_integration import load_match_history_and_map, join_odds
from preprocessing.elo_integration import merge_elo_features

# ---------- Config ----------
INPUT_GLOB = "data/understat/*/*/matches.parquet"
OUTPUT_DIR = Path("data/training")
OUTPUT_PARQUET = OUTPUT_DIR / "understat_df.parquet"
UNDERSTAT_MAPPING_PATH = Path("data/mappings/understat_to_canonical.json")

def main():
    pl.enable_string_cache()

    # Scan & normalize
    lf = pl.scan_parquet(INPUT_GLOB)
    lf = rename_and_cast(lf)
    
    # Apply Canonical Mapping to Understat Data
    if UNDERSTAT_MAPPING_PATH.exists():
        with open(UNDERSTAT_MAPPING_PATH, "r") as f:
            u_mapping = json.load(f)
        
        print("Applying canonical team mapping to Understat data...")
        lf = lf.with_columns([
            pl.col("home_team").replace(u_mapping).alias("home_team"),
            pl.col("away_team").replace(u_mapping).alias("away_team")
        ])
    
    # Load Match History
    mh = load_match_history_and_map()
    
    if mh is not None:
        print("Joining Match History data...")
        lf = join_odds(lf, mh)
    
    # Load Match History
    mh = load_match_history_and_map()
    
    if mh is not None:
        print("Joining Match History data...")
        lf = join_odds(lf, mh)

    # Merge Elo ratings
    print("Merging Elo ratings...")
    # merge_elo_features expects DataFrame (eager) and returns DataFrame (eager)
    df_temp = lf.collect()
    df_temp = merge_elo_features(df_temp)
    lf = df_temp.lazy()

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
        "home_shots",
        "away_shots",
        "home_sot",
        "away_sot",
        "home_deep",
        "away_deep",
        "home_ppda",
        "away_ppda",
        "odds_h",
        "odds_d",
        "odds_a",
        "odds_over",
        "odds_under",
        "home_elo",
        "away_elo",
        "elo_diff",
        "elo_sum",
        "elo_mean"
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
