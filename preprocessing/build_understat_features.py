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
    compute_opponent_baselines,
    join_opponent_baselines,
    compute_adjusted_stats,
    compute_adjusted_rolling_features,
    build_match_level,
    merge_european_schedule,
    compute_schedule_features,
    add_categorical_features,
    load_promoted_teams,
    build_promoted_teams_set,
)
from preprocessing.odds_integration import load_match_history_and_map, join_odds
from preprocessing.elo_integration import merge_elo_features
from preprocessing.player_feature_engineering import (
    load_all_player_data,
    build_player_team_features,
)

# ---------- Config ----------
INPUT_GLOB = "data/understat/*/*/matches.parquet"
OUTPUT_DIR = Path("data/training")
OUTPUT_PARQUET = OUTPUT_DIR / "understat_df.parquet"
UNDERSTAT_MAPPING_PATH = Path("data/mappings/understat_to_canonical.json")
EUROPEAN_SCHEDULE_PATH = Path("data/full_schedule/european_all.csv")

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
        "home_team_id",
        "away_team_id",
        "game_id",
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
        "elo_mean",
    ]
    
    schema = lf.collect_schema()
    have = set(schema.names())
    base_cols = [c for c in base_needed if c in have]
    base_matches = lf.select(base_cols)

    # Long spine
    long_df = build_long(base_matches)

    # Rolling features (within league+season; shift(1) prevents leakage)
    long_feats = compute_rolling_features(long_df)
    
    # Opponent-adjusted features
    # 1. Compute each team's baseline stats (what they typically produce/concede)
    long_feats = compute_opponent_baselines(long_feats)
    # 2. Join opponent's baselines to each row
    long_feats = join_opponent_baselines(long_feats)
    # 3. Calculate adjusted stats (actual / expected based on opponent)
    long_feats = compute_adjusted_stats(long_feats)
    # 4. Compute rolling averages of adjusted stats
    long_feats = compute_adjusted_rolling_features(long_feats)
    
    # Merge European schedule for fixture congestion features
    if EUROPEAN_SCHEDULE_PATH.exists():
        print("Merging European schedule for fixture congestion features...")
        combined_long = merge_european_schedule(long_feats, EUROPEAN_SCHEDULE_PATH)
        
        # Compute schedule features (days_since_last_match, games_last_15_days)
        print("Computing schedule features...")
        combined_df = compute_schedule_features(combined_long)
        
        # Filter back to domestic games only and join schedule features to long_feats
        domestic_with_schedule = combined_df.filter(pl.col("is_european") == False)
        schedule_cols = ["match_id", "team", "days_since_last_match", "games_last_15_days"]
        schedule_feats = domestic_with_schedule.select(schedule_cols)
        
        long_feats = long_feats.collect().join(
            schedule_feats,
            on=["match_id", "team"],
            how="left"
        ).lazy()
    else:
        print("No European schedule found, skipping fixture congestion features")
        long_feats = long_feats.with_columns([
            pl.lit(None).cast(pl.Float64).alias("days_since_last_match"),
            pl.lit(None).cast(pl.Int64).alias("games_last_15_days"),
        ])

    # Rejoin to match level
    final_df = build_match_level(base_matches, long_feats)
    
    # Add player-derived team features
    print("Building player-derived team features...")
    player_df = load_all_player_data()
    player_team_features = build_player_team_features(player_df)
    
    # Join player features for home team
    player_feature_cols = [c for c in player_team_features.columns if "_r15" in c or "_r5_sum" in c]
    home_player_feats = player_team_features.select(
        ["league", "team_id", "game_id"] + player_feature_cols
    ).rename({"team_id": "home_team_id"})
    home_player_feats = home_player_feats.rename({col: f"home_{col}" for col in player_feature_cols})
    
    # Join player features for away team  
    away_player_feats = player_team_features.select(
        ["league", "team_id", "game_id"] + player_feature_cols
    ).rename({"team_id": "away_team_id"})
    away_player_feats = away_player_feats.rename({col: f"away_{col}" for col in player_feature_cols})
    
    # Collect final_df for joining
    final_df_collected = final_df.collect()
    
    # Check if we have the necessary columns for joining
    if "home_team_id" in final_df_collected.columns and "away_team_id" in final_df_collected.columns:
        print(f"Joining {len(player_feature_cols)} player features for home and away teams...")
        
        # Cast league to string to match player features (which uses str)
        final_df_collected = final_df_collected.with_columns(
            pl.col("league").cast(pl.Utf8)
        )
        
        final_df_collected = final_df_collected.join(
            home_player_feats,
            left_on=["league", "home_team_id", "game_id"],
            right_on=["league", "home_team_id", "game_id"],
            how="left"
        ).join(
            away_player_feats,
            left_on=["league", "away_team_id", "game_id"],
            right_on=["league", "away_team_id", "game_id"],
            how="left"
        )
        print(f"Added player features: home columns = {len([c for c in final_df_collected.columns if c.startswith('home_') and '_r15' in c])}")
    else:
        print("Warning: Missing team_id columns, skipping player feature join")

    # Add categorical features (league_idx, round_number, season_progress, promoted flags)
    print("Adding categorical features...")
    promoted_data = load_promoted_teams()
    promoted_lookup = build_promoted_teams_set(promoted_data)
    final_df_collected = add_categorical_features(final_df_collected.lazy(), promoted_lookup).collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df_collected.write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
