import soccerdata as sd
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from preprocessing.feature_engineering import (
    rename_and_cast,
    build_long,
    compute_rolling_features,
    build_match_level,
)

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
OUTPUT_DIR = Path("data/prod")
OUTPUT_PARQUET = OUTPUT_DIR / "features_season.parquet"

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
    # lf = fetch_current_data().lazy() # Removed double call
    
    df = fetch_current_data()
    if df.is_empty():
        print("No data found.")
        return
    lf = df.lazy()

    # 2. Preprocess
    lf = rename_and_cast(lf)
    
    # Base match columns needed
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
    
    # 3. Feature Engineering
    long_df = build_long(base_matches)
    long_feats = compute_rolling_features(long_df)
    final_df = build_match_level(base_matches, long_feats)
    
    # 4. Filter for upcoming matches
    # We want matches in the future.
    # We can filter by date >= today.
    
    now = datetime.now()
    # We might want to include today's matches even if time passed?
    # Let's say date >= today at 00:00
    today = datetime(now.year, now.month, now.day)
    
    final_df = final_df.filter(
        pl.col("date") >= today
    )
    
    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.collect().write_parquet(OUTPUT_PARQUET)
    print(f"Wrote: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
