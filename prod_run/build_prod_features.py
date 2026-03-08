import soccerdata as sd
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

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
from prod_run.elo_scrap import build_prod_elo
from utils.paths import MAPPINGS_DIR

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
OUTPUT_DIR = Path("data/prod")
OUTPUT_PARQUET = OUTPUT_DIR / "features_season.parquet"
EUROPEAN_SCHEDULE_PATH = Path("data/full_schedule/european_all.csv")

def get_current_season_str():
    now = datetime.now()
    if now.month > 6:
        return f"{now.year}/{now.year + 1}"
    else:
        return f"{now.year - 1}/{now.year}"

def fetch_current_data():
    season_str = get_current_season_str()
    print(f"Fetching data for season {season_str}...")
    
    all_matches = []
    
    for league in LEAGUES:
        print(f"  Processing {league}...")
        try:
            reader = sd.Understat(leagues=league, seasons=season_str)
            
            # 1. Completed matches stats
            stats = reader.read_team_match_stats()
            if not stats.empty:
                stats = stats.reset_index()
                # Rename columns
                stats.columns = [c.lower() for c in stats.columns]
                rename_map = {
                    "home_shot": "home_shots",
                    "away_shot": "away_shots",
                    "home_shotontarget": "home_sot",
                    "away_shotontarget": "away_sot",
                    "game": "match_id"
                }
                stats = stats.rename(columns=rename_map)
                
                # Ensure date is datetime
                if 'date' in stats.columns:
                    stats['date'] = pd.to_datetime(stats['date'])
                
                all_matches.append(stats)
                
            # 2. Schedule (upcoming)
            schedule = reader.read_schedule(include_matches_without_data=True)
            if not schedule.empty:
                schedule = schedule.reset_index()
                schedule.columns = [c.lower() for c in schedule.columns]
                if 'game' in schedule.columns:
                    schedule = schedule.rename(columns={'game': 'match_id'})
                
                # Ensure date is datetime
                if 'date' in schedule.columns:
                    schedule['date'] = pd.to_datetime(schedule['date'])
                
                all_matches.append(schedule)
                
        except Exception as e:
            print(f"    Error fetching {league}: {e}")
            
    if not all_matches:
        return pl.DataFrame()
        
    # Combine all
    full_df = pd.concat(all_matches, ignore_index=True)
    
    # Deduplicate: if a match is in both stats and schedule, keep stats (more cols)
    full_df['non_null_count'] = full_df.notnull().sum(axis=1)
    full_df = full_df.sort_values('non_null_count', ascending=False)
    # match_id is unique per league usually, but let's be safe with league+match_id
    full_df = full_df.drop_duplicates(subset=['league', 'match_id'])
    full_df = full_df.drop(columns=['non_null_count'])
    
    return pl.from_pandas(full_df)

def main():
    pl.enable_string_cache()
    
    # 1. Fetch data
    df = fetch_current_data()
    if df.is_empty():
        print("No data fetched.")
        return

    lf = df.lazy()
    lf = rename_and_cast(lf)
    
    # Apply Canonical Mapping to Production Data
    UNDERSTAT_MAPPING_PATH = MAPPINGS_DIR / "understat_to_canonical.json"
    if UNDERSTAT_MAPPING_PATH.exists():
        with open(UNDERSTAT_MAPPING_PATH, "r") as f:
            u_mapping = json.load(f)
        
        lf = lf.with_columns([
            pl.col("home_team").replace(u_mapping).alias("home_team"),
            pl.col("away_team").replace(u_mapping).alias("away_team")
        ])

    # 2. Join Odds (Match History)
    mh = load_match_history_and_map()
    if mh is not None:
        print("Joining Match History data (Odds)...")
        lf = join_odds(lf, mh)

    # --- Elo Integration ---
    print("Merging Elo ratings...")
    df_temp = lf.collect()
    df_temp = merge_elo_features(df_temp)
    
    # Fill missing Elo with current Elo (for upcoming games)
    missing_mask = df_temp["home_elo"].is_null() | df_temp["away_elo"].is_null()
    if missing_mask.any():
        print("Fetching current Elo for missing values...")
        try:
            elo_paths = build_prod_elo(write_histories=False)
            elo_asof_path = elo_paths["elo_asof"]
            elo_current = pl.read_parquet(elo_asof_path)
            
            with open(MAPPINGS_DIR / "clubelo_to_canonical.json", "r") as f:
                mapping = json.load(f)
            
            mapping_df = pl.DataFrame([{"team_clubelo": k, "team_canonical": v} for k, v in mapping.items()])
            
            elo_mapped = elo_current.join(
                mapping_df,
                left_on="team_clubelo",
                right_on="team_clubelo",
                how="inner"
            ).select([pl.col("team_canonical"), pl.col("elo")])
            
            # Join and fill
            df_temp = df_temp.join(
                elo_mapped.rename({"team_canonical": "home_team", "elo": "home_elo_curr"}),
                on="home_team",
                how="left"
            ).join(
                elo_mapped.rename({"team_canonical": "away_team", "elo": "away_elo_curr"}),
                on="away_team",
                how="left"
            ).with_columns([
                pl.col("home_elo").fill_null(pl.col("home_elo_curr")),
                pl.col("away_elo").fill_null(pl.col("away_elo_curr"))
            ]).drop(["home_elo_curr", "away_elo_curr"])
            
            # Recompute features
            df_temp = df_temp.with_columns([
                (pl.col("home_elo") - pl.col("away_elo")).alias("elo_diff"),
                (pl.col("home_elo") + pl.col("away_elo")).alias("elo_sum"),
                ((pl.col("home_elo") + pl.col("away_elo")) / 2).alias("elo_mean")
            ])
        except Exception as e:
            print(f"Error filling current Elo: {e}")

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
    long_feats = compute_opponent_baselines(long_feats)
    long_feats = join_opponent_baselines(long_feats)
    long_feats = compute_adjusted_stats(long_feats)
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

    # Add player-derived team features using the latest available rolling window per team
    print("Building player-derived team features...")
    player_df = load_all_player_data()
    player_team_features = build_player_team_features(player_df)

    player_feature_cols = [c for c in player_team_features.columns if "_r15" in c or "_r5_sum" in c]
    final_df_collected = final_df.collect()

    if "home_team_id" in final_df_collected.columns and "away_team_id" in final_df_collected.columns:
        print(f"Joining {len(player_feature_cols)} player features for home and away teams...")

        final_df_collected = final_df_collected.with_columns([
            pl.col("league").cast(pl.Utf8),
            pl.col("date").cast(pl.Datetime),
        ])

        home_player_feats = player_team_features.select(
            ["league", "team_id", "date"] + player_feature_cols
        ).rename({"team_id": "home_team_id"})
        home_player_feats = home_player_feats.rename({col: f"home_{col}" for col in player_feature_cols})

        away_player_feats = player_team_features.select(
            ["league", "team_id", "date"] + player_feature_cols
        ).rename({"team_id": "away_team_id"})
        away_player_feats = away_player_feats.rename({col: f"away_{col}" for col in player_feature_cols})

        final_df_collected = final_df_collected.sort(["league", "home_team_id", "date"]).join_asof(
            home_player_feats.sort(["league", "home_team_id", "date"]),
            on="date",
            by=["league", "home_team_id"],
            strategy="backward",
        )

        final_df_collected = final_df_collected.sort(["league", "away_team_id", "date"]).join_asof(
            away_player_feats.sort(["league", "away_team_id", "date"]),
            on="date",
            by=["league", "away_team_id"],
            strategy="backward",
        )
    else:
        print("Warning: Missing team_id columns, skipping player feature join")

    # Add categorical features (league_idx, round_number, season_progress, promoted flags)
    print("Adding categorical features...")
    promoted_data = load_promoted_teams()
    promoted_lookup = build_promoted_teams_set(promoted_data)
    final_df = add_categorical_features(final_df_collected.lazy(), promoted_lookup)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.collect().write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
