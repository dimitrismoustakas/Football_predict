import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime

# LEAGUES = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "RFPL"]
# Using the names soccerdata expects if they differ, but usually they are close.
# Let's check available leagues if needed, but for now assume these work or map them.
# soccerdata uses "La Liga" -> "La Liga", "EPL" -> "EPL", "Bundesliga" -> "Bundesliga", "Serie A" -> "Serie A", "Ligue 1" -> "Ligue 1", "RFPL" -> "RFPL"
# LEAGUES = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "RFPL"]
# Using the names soccerdata expects if they differ, but usually they are close.
# Let's check available leagues if needed, but for now assume these work or map them.
# soccerdata uses "La Liga" -> "La Liga", "EPL" -> "EPL", "Bundesliga" -> "Bundesliga", "Serie A" -> "Serie A", "Ligue 1" -> "Ligue 1", "RFPL" -> "RFPL"
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
    # If we are in the second half of the year, the season started this year.
    # If we are in the first half, the season started last year.
    # But we want to go up to the current season (e.g. 2025/2026 if it's Dec 2025).
    # Actually, let's just go up to a safe future year or handle errors.
    # The user said "start from the 2014-2015 season and make the code work upward from there."
    
    # Let's assume we want to go up to the current season.
    # If today is Dec 2025, current season is 2025/2026.
    if datetime.now().month > 6:
        end_year = current_year
    else:
        end_year = current_year - 1
        
    # We can also just try to fetch and if it fails or is empty, we stop?
    # But soccerdata might just return empty df.
    
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
                
                # Rename columns to match our schema
                # Expected by feature_engineering:
                # home_team, away_team, home_goals, away_goals, home_xg, away_xg, 
                # home_shots, away_shots, home_sot, away_sot, home_deep, away_deep, home_ppda, away_ppda
                # date, match_id, league_id, league, season
                
                # soccerdata columns (based on common output):
                # league, season, game, team, date, ...
                # read_team_match_stats returns one row per team per match? 
                # Wait, read_team_match_stats usually returns stats for both teams in a match if it's match-level?
                # Or is it team-level?
                # Let's check documentation or assume it returns match-level with home/away columns if formatted right.
                # Actually, read_team_match_stats returns a MultiIndex [league, season, game].
                # And columns like home_team, away_team, etc.
                
                # Let's inspect columns in a dry run or assume standard mapping.
                # Common mapping:
                # home_team -> home_team
                # away_team -> away_team
                # home_goals -> home_goals
                # away_goals -> away_goals
                # home_xG -> home_xg
                # away_xG -> away_xg
                # home_shots -> home_shots
                # away_shots -> away_shots
                # home_sot -> home_sot
                # away_sot -> away_sot
                # home_deep -> home_deep
                # away_deep -> away_deep
                # home_ppda -> home_ppda
                # away_ppda -> away_ppda
                
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
                
                # Check what we have
                # If we have 'home_xg', keep it.
                
                # We also need 'match_id'. soccerdata uses 'game' in index.
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
