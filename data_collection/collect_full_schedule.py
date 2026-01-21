"""
Collect full match schedules including European competitions (Champions League, Europa League, etc.)
from FBRef using the soccerdata library.

This script fetches comprehensive match schedules that include:
- Domestic league games (Big 5 leagues)
- Champions League
- Europa League  
- Conference League

The soccerdata library handles rate limiting and caching automatically.

SETUP REQUIRED:
Before running this script, ensure the custom league config exists at:
C:\\Users\\<username>\\soccerdata\\config\\league_dict.json

With content:
{
  "EUR-Champions League": {"FBref": "UEFA Champions League", "season_start": "Sep", "season_end": "May"},
  "EUR-Europa League": {"FBref": "UEFA Europa League", "season_start": "Sep", "season_end": "May"},
  "EUR-Conference League": {"FBref": "UEFA Conference League", "season_start": "Sep", "season_end": "May"}
}
"""

import pandas as pd
import soccerdata as sd
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "full_schedule"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Leagues to fetch
DOMESTIC_LEAGUES = [
	"ENG-Premier League",
	"ESP-La Liga", 
	"GER-Bundesliga",
	"ITA-Serie A",
	"FRA-Ligue 1",
]

EUROPEAN_LEAGUES = [
	"EUR-Champions League",
	"EUR-Europa League",
	"EUR-Conference League",
]


def ensure_league_config():
	"""Ensure the soccerdata custom league config exists with European competitions."""
	import os
	config_dir = Path(os.path.expanduser("~")) / "soccerdata" / "config"
	config_dir.mkdir(parents=True, exist_ok=True)
	
	league_dict_path = config_dir / "league_dict.json"
	
	league_dict = {
		"EUR-Champions League": {
			"FBref": "UEFA Champions League",
			"season_start": "Sep",
			"season_end": "May"
		},
		"EUR-Europa League": {
			"FBref": "UEFA Europa League", 
			"season_start": "Sep",
			"season_end": "May"
		},
		"EUR-Conference League": {
			"FBref": "UEFA Conference League",
			"season_start": "Sep", 
			"season_end": "May"
		}
	}
	
	# Only write if file doesn't exist or is empty
	if not league_dict_path.exists() or league_dict_path.stat().st_size == 0:
		with open(league_dict_path, 'w') as f:
			json.dump(league_dict, f, indent=2)
		print(f"Created league config at: {league_dict_path}")
	else:
		# Check if our leagues are in the config
		with open(league_dict_path) as f:
			existing = json.load(f)
		
		updated = False
		for league, config in league_dict.items():
			if league not in existing:
				existing[league] = config
				updated = True
		
		if updated:
			with open(league_dict_path, 'w') as f:
				json.dump(existing, f, indent=2)
			print(f"Updated league config at: {league_dict_path}")


def get_current_season() -> str:
	"""Determine the current football season based on today's date."""
	today = datetime.now()
	year = today.year
	month = today.month
	
	# Football season runs Aug-May
	# If we're in Jan-Jul, we're in the season that started last year
	if month <= 7:
		return f"{year-1}-{year}"
	else:
		return f"{year}-{year+1}"


def get_historical_seasons(start_year: int = 2014) -> list[str]:
	"""
	Generate list of seasons from start_year to current season.
	
	Args:
		start_year: First year to include (e.g., 2014 for 2014-2015 season)
	
	Returns:
		List of season strings like ["2014-2015", "2015-2016", ..., "2024-2025"]
	"""
	today = datetime.now()
	current_year = today.year
	current_month = today.month
	
	# Determine the last season to include
	if current_month <= 7:
		end_year = current_year - 1  # Still in previous season
	else:
		end_year = current_year  # New season has started
	
	seasons = []
	for year in range(start_year, end_year + 1):
		seasons.append(f"{year}-{year+1}")
	
	return seasons


def fetch_league_schedule(leagues: list[str], season: str) -> pd.DataFrame:
	"""
	Fetch schedule for the specified leagues and season.
	
	Args:
		leagues: List of league IDs (e.g., ["EUR-Champions League", "ENG-Premier League"])
		season: Season string (e.g., "2024-2025")
	
	Returns:
		DataFrame with all matches
	"""
	all_matches = []
	
	for league in leagues:
		print(f"Fetching {league} ({season})...")
		try:
			fbref = sd.FBref(leagues=league, seasons=season)
			df = fbref.read_schedule()
			df = df.reset_index()
			all_matches.append(df)
			print(f"  Found {len(df)} matches")
		except Exception as e:
			print(f"  Error: {e}")
	
	if not all_matches:
		return pd.DataFrame()
	
	return pd.concat(all_matches, ignore_index=True)


def filter_upcoming_matches(df: pd.DataFrame, days_ahead: int = 14) -> pd.DataFrame:
	"""
	Filter DataFrame to only include upcoming matches within the specified days.
	
	Args:
		df: DataFrame with schedule
		days_ahead: Number of days ahead to include
	
	Returns:
		Filtered DataFrame with upcoming matches only
	"""
	if df.empty:
		return df
	
	# Ensure date column is datetime
	df = df.copy()
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	
	# Filter for upcoming (unplayed) matches
	# Unplayed matches have NaN score or date >= today
	today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
	end_date = today + timedelta(days=days_ahead)
	
	# Score can be NaN or empty string for unplayed matches
	is_unplayed = df["score"].isna() | (df["score"] == "")
	is_in_range = (df["date"] >= today) & (df["date"] <= end_date)
	
	return df[is_unplayed & is_in_range]


def filter_completed_matches(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Filter DataFrame to only include completed matches.
	
	Args:
		df: DataFrame with schedule
	
	Returns:
		Filtered DataFrame with completed matches only
	"""
	if df.empty:
		return df
	
	# Completed matches have a score
	df = df.copy()
	has_score = df["score"].notna() & (df["score"] != "")
	return df[has_score]


def save_schedule(df: pd.DataFrame, filename: str, output_dir: Path = OUTPUT_DIR):
	"""Save schedule DataFrame to CSV and Parquet."""
	if df.empty:
		print(f"  No data to save for {filename}")
		return
	
	csv_path = output_dir / f"{filename}.csv"
	parquet_path = output_dir / f"{filename}.parquet"
	
	df.to_csv(csv_path, index=False)
	df.to_parquet(parquet_path, index=False)
	
	print(f"  Saved {len(df)} matches to {csv_path}")


def main():
	"""Main function to collect and save full schedules."""
	print("=" * 60)
	print("Collecting Full Match Schedules")
	print("=" * 60)
	
	# Ensure league config exists
	ensure_league_config()
	
	current_season = get_current_season()
	print(f"\nCurrent season: {current_season}")
	
	# Get all historical seasons (2014-2015 to current)
	all_seasons = get_historical_seasons(start_year=2014)
	print(f"Historical seasons to fetch: {all_seasons[0]} to {all_seasons[-1]} ({len(all_seasons)} seasons)")
	
	# Collect European competitions (all historical seasons)
	print("\n--- European Competitions (Historical) ---")
	european_dfs = []
	for season in all_seasons:
		season_df = fetch_league_schedule(EUROPEAN_LEAGUES, season)
		if not season_df.empty:
			european_dfs.append(season_df)
	
	european_df = pd.concat(european_dfs, ignore_index=True) if european_dfs else pd.DataFrame()
	
	if not european_df.empty:
		# All matches (history + upcoming)
		save_schedule(european_df, "european_all")
		
		# Upcoming matches only
		european_upcoming = filter_upcoming_matches(european_df, days_ahead=14)
		save_schedule(european_upcoming, "european_upcoming")
		
		print(f"\n  Total European matches: {len(european_df)}")
		print(f"  Upcoming European matches (14 days): {len(european_upcoming)}")
	
	# Collect domestic leagues (all historical seasons)
	print("\n--- Domestic Leagues (Historical) ---")
	domestic_dfs = []
	for season in all_seasons:
		season_df = fetch_league_schedule(DOMESTIC_LEAGUES, season)
		if not season_df.empty:
			domestic_dfs.append(season_df)
	
	domestic_df = pd.concat(domestic_dfs, ignore_index=True) if domestic_dfs else pd.DataFrame()
	
	if not domestic_df.empty:
		# All matches
		save_schedule(domestic_df, "domestic_all")
		
		# Upcoming matches only
		domestic_upcoming = filter_upcoming_matches(domestic_df, days_ahead=14)
		save_schedule(domestic_upcoming, "domestic_upcoming")
		
		print(f"\n  Total domestic matches: {len(domestic_df)}")
		print(f"  Upcoming domestic matches (14 days): {len(domestic_upcoming)}")
	
	# Combined schedule (European + Domestic)
	print("\n--- Combined Schedule ---")
	combined_df = pd.DataFrame()
	if not european_df.empty or not domestic_df.empty:
		dfs_to_combine = []
		if not european_df.empty:
			dfs_to_combine.append(european_df)
		if not domestic_df.empty:
			dfs_to_combine.append(domestic_df)
		combined_df = pd.concat(dfs_to_combine, ignore_index=True)
		save_schedule(combined_df, "all_competitions")
		
		combined_upcoming = filter_upcoming_matches(combined_df, days_ahead=14)
		save_schedule(combined_upcoming, "all_upcoming")
		
		print(f"\n  Total matches (all competitions): {len(combined_df)}")
		print(f"  Upcoming matches (14 days): {len(combined_upcoming)}")
	
	# Print summary of upcoming matches
	if not combined_df.empty:
		upcoming = filter_upcoming_matches(combined_df, days_ahead=14)
		if not upcoming.empty:
			print("\n--- Upcoming Matches Preview (first 20) ---")
			cols_to_show = ["league", "date", "home_team", "away_team", "venue"]
			cols_available = [c for c in cols_to_show if c in upcoming.columns]
			# Sort by date
			upcoming = upcoming.sort_values("date")
			preview = upcoming[cols_available].head(20)
			# Print without unicode issues
			for _, row in preview.iterrows():
				date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "TBD"
				league = str(row.get("league", ""))[:25]
				home = str(row.get("home_team", ""))[:20]
				away = str(row.get("away_team", ""))[:20]
				print(f"  {date_str} | {league:<25} | {home:<20} vs {away:<20}")
	
	print("\n" + "=" * 60)
	print("Collection complete!")
	print(f"Output directory: {OUTPUT_DIR}")
	print("=" * 60)


if __name__ == "__main__":
	main()
