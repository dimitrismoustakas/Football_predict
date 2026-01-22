import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]

START_YEAR = 2014

START_YEAR = 2014
OUTPUT_ROOT = Path("data/understat")

def get_season_str(start_year):
    return f"{start_year}/{start_year + 1}"

def get_folder_season_str(start_year):
    return f"{start_year}-{start_year + 1}"

def sanitize_league_name(league):
    return league.replace(" ", "_")

def collect_history():
    current_year = datetime.now().year
    # If today is Dec 2025, current season is 2025/2026.
    if datetime.now().month > 6:
        end_year = current_year
    else:
        end_year = current_year - 1

    for league in LEAGUES:
        print(f"Processing {league}...")
        sanitized_league = sanitize_league_name(league)
        
        for year in range(START_YEAR, end_year + 1):
            season_str = get_season_str(year)
            folder_season_str = get_folder_season_str(year)
            
            print(f"  Fetching {season_str}...")
            
            try:
                reader = sd.Understat(leagues=league, seasons=season_str)
                df = reader.read_team_match_stats()
                
                if df.empty:
                    print(f"    No data for {season_str}. Skipping.")
                    continue
                
                # Reset index to get columns
                df = df.reset_index()
                
                # We need to normalize column names to lowercase/snake_case if they aren't.
                df.columns = [c.lower() for c in df.columns]
                
                # Fix specific renames if needed
                rename_map = {
                    "home_shot": "home_shots",
                    "away_shot": "away_shots",
                    "home_shotontarget": "home_sot",
                    "away_shotontarget": "away_sot",
                }
                df = df.rename(columns=rename_map)
                
                if 'game' in df.columns:
                    df = df.rename(columns={'game': 'match_id'})
                
                # Ensure date is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Save
                output_dir = OUTPUT_ROOT / sanitized_league / folder_season_str
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "matches.parquet"
                
                df.to_parquet(output_path)
                print(f"    Saved to {output_path}")
                
            except Exception as e:
                print(f"    Error fetching {season_str}: {e}")

if __name__ == "__main__":
    collect_history()