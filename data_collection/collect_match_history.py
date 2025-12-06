import soccerdata as sd
import pandas as pd
from pathlib import Path
from datetime import datetime

# Leagues configuration
# Using the same league names as in collect_understat.py
LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
START_YEAR = 2014
OUTPUT_DIR = Path("data/match_history")
OUTPUT_FILE = OUTPUT_DIR / "matches.parquet"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Determine end season
    current_year = datetime.now().year
    # If we are in the second half of the year, the season started this year.
    if datetime.now().month > 6:
        end_year = current_year
    else:
        end_year = current_year - 1
        
    seasons = []
    for year in range(START_YEAR, end_year + 1):
        # soccerdata MatchHistory expects '14-15', '15-16' format
        short_start = str(year)[-2:]
        short_end = str(year + 1)[-2:]
        seasons.append(f"{short_start}-{short_end}")
        
    print(f"Fetching MatchHistory for leagues: {LEAGUES}")
    print(f"Seasons: {seasons}")
    
    # Initialize the scraper
    mh = sd.MatchHistory(leagues=LEAGUES, seasons=seasons)
    
    # Read games
    print("Downloading data...")
    df = mh.read_games()
    
    if df.empty:
        print("No data found.")
        return

    print(f"Collected {len(df)} matches.")
    
    # Reset index to ensure all key columns (league, season, game_id/date) are available as columns
    df = df.reset_index()
    
    # Save to parquet
    df.to_parquet(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
