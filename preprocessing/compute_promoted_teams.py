"""
Compute promoted teams from historical data.

For seasons after 2014/2015, promoted teams are inferred as teams present
in the current season but not in the previous season.

The 2014/2015 season is manually seeded in data/promoted_teams.json.

This script reads all historical data, computes promoted teams, and updates
the promoted_teams.json file.
"""

import polars as pl
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import DATA_DIR

UNDERSTAT_GLOB = DATA_DIR / "understat" / "*" / "*" / "matches.parquet"
PROMOTED_TEAMS_PATH = DATA_DIR / "promoted_teams.json"


def get_teams_per_league_season(lf: pl.LazyFrame) -> dict[str, dict[str, set[str]]]:
	"""
	Extract unique teams per league per season from match data.
	Returns: {league: {season: {team1, team2, ...}}}
	"""
	# Get unique teams (both home and away)
	home_teams = lf.select(
		pl.col("league").cast(pl.Utf8),
		pl.col("season").cast(pl.Utf8),
		pl.col("home_team").cast(pl.Utf8).alias("team"),
	)
	away_teams = lf.select(
		pl.col("league").cast(pl.Utf8),
		pl.col("season").cast(pl.Utf8),
		pl.col("away_team").cast(pl.Utf8).alias("team"),
	)
	
	all_teams = pl.concat([home_teams, away_teams]).unique().collect()
	
	result = {}
	for row in all_teams.iter_rows(named=True):
		league = row["league"]
		season = row["season"]
		team = row["team"]
		
		if league not in result:
			result[league] = {}
		if season not in result[league]:
			result[league][season] = set()
		result[league][season].add(team)
	
	return result


def compute_promoted_teams(teams_by_league_season: dict) -> dict[str, dict[str, list[str]]]:
	"""
	Compute promoted teams by comparing consecutive seasons.
	A team is promoted if it's in the current season but wasn't in the previous season.
	"""
	promoted = {}
	
	for league, seasons_teams in teams_by_league_season.items():
		promoted[league] = {}
		
		# Sort seasons chronologically (format: "1415", "1516", etc.)
		sorted_seasons = sorted(seasons_teams.keys())
		
		for i, season in enumerate(sorted_seasons):
			if i == 0:
				# First season - can't compute (need seed data)
				continue
			
			prev_season = sorted_seasons[i - 1]
			current_teams = seasons_teams[season]
			prev_teams = seasons_teams[prev_season]
			
			# Teams in current but not in previous = promoted
			new_teams = current_teams - prev_teams
			if new_teams:
				promoted[league][season] = sorted(list(new_teams))
	
	return promoted


def load_existing_promoted_teams() -> dict:
	"""Load existing promoted teams JSON (contains seed data)."""
	if PROMOTED_TEAMS_PATH.exists():
		with open(PROMOTED_TEAMS_PATH, "r", encoding="utf-8") as f:
			return json.load(f)
	return {}


def merge_promoted_teams(existing: dict, computed: dict) -> dict:
	"""
	Merge computed promoted teams with existing (seed) data.
	Existing seed data takes precedence for seasons that are manually entered.
	"""
	# Start with metadata from existing
	result = {k: v for k, v in existing.items() if k.startswith("_")}
	
	# Get all leagues
	all_leagues = set(existing.keys()) | set(computed.keys())
	all_leagues = {l for l in all_leagues if not l.startswith("_")}
	
	for league in all_leagues:
		result[league] = {}
		
		existing_seasons = existing.get(league, {})
		computed_seasons = computed.get(league, {})
		
		all_seasons = set(existing_seasons.keys()) | set(computed_seasons.keys())
		
		for season in sorted(all_seasons):
			# Existing (seed) data takes precedence
			if season in existing_seasons:
				result[league][season] = existing_seasons[season]
			elif season in computed_seasons:
				result[league][season] = computed_seasons[season]
	
	return result


def save_promoted_teams(data: dict):
	"""Save promoted teams to JSON."""
	with open(PROMOTED_TEAMS_PATH, "w", encoding="utf-8") as f:
		json.dump(data, f, indent="\t", ensure_ascii=False)
	print(f"Saved promoted teams to {PROMOTED_TEAMS_PATH}")


def main():
	print("Loading match data...")
	lf = pl.scan_parquet(str(UNDERSTAT_GLOB))
	
	print("Extracting teams per league per season...")
	teams_by_league_season = get_teams_per_league_season(lf)
	
	# Print summary
	for league, seasons in sorted(teams_by_league_season.items()):
		print(f"\n{league}:")
		for season in sorted(seasons.keys()):
			print(f"  {season}: {len(seasons[season])} teams")
	
	print("\nComputing promoted teams...")
	computed = compute_promoted_teams(teams_by_league_season)
	
	print("\nLoading existing seed data...")
	existing = load_existing_promoted_teams()
	
	print("Merging computed with seed data...")
	merged = merge_promoted_teams(existing, computed)
	
	# Print results
	print("\n" + "=" * 60)
	print("PROMOTED TEAMS BY LEAGUE AND SEASON")
	print("=" * 60)
	for league, seasons in sorted(merged.items()):
		if league.startswith("_"):
			continue
		print(f"\n{league}:")
		for season, teams in sorted(seasons.items()):
			print(f"  {season}: {teams}")
	
	save_promoted_teams(merged)


if __name__ == "__main__":
	main()
