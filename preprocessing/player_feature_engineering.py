"""
Player-level feature engineering for team ability estimation.

Features are computed using player performance data aggregated at the team level.
Rolling features require two observations and use up to 15 completed games.

Feature Categories:
1. Expected Abilities (xG90, xA90) - minute-weighted team averages
2. Concentration Metrics (HHI for xG, xA) - how concentrated is output among players
3. Squad Depth (unique players, rotation index)
4. Availability (total minutes available)
"""

import polars as pl
from datetime import datetime
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import DATA_DIR


# Configuration
MAX_WINDOW = 15  # Maximum games to look back
MIN_GAMES = 2    # Minimum games required to compute features (matches base features)


def load_all_player_data(data_root: Path = DATA_DIR / "understat") -> pl.DataFrame:
	"""Load all player match stats from parquet files."""
	player_files = list(data_root.rglob("player_match_stats.parquet"))
	
	if not player_files:
		raise FileNotFoundError(f"No player_match_stats.parquet files found in {data_root}")
	
	dfs = [pl.read_parquet(f) for f in player_files]
	df = pl.concat(dfs, how="diagonal")
	
	return df


def prepare_player_data(df: pl.DataFrame) -> pl.DataFrame:
	"""
	Prepare player data for feature engineering.
	
	- Extract date from match_id
	- Ensure proper types
	- Sort by date
	"""
	df = df.with_columns([
		# Extract date from match_id (format: "2024-08-16 Team1-Team2")
		pl.col("match_id").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
	])
	
	# Ensure numeric columns
	numeric_cols = ["minutes", "goals", "xg", "xa", "assists", "shots", "key_passes", "xg_chain", "xg_buildup"]
	for col in numeric_cols:
		if col in df.columns:
			df = df.with_columns(pl.col(col).cast(pl.Float64))
	
	# Sort by date
	df = df.sort(["league", "team_id", "date"])
	
	return df


def compute_team_match_aggregates(df: pl.DataFrame) -> pl.DataFrame:
	"""
	Aggregate player-level stats to team-match level.
	
	For each (team, match) compute:
	- Total minutes played by all players
	- Minute-weighted xG (sum of player xG)
	- Minute-weighted xA (sum of player xA)
	- HHI for xG concentration (Herfindahl-Hirschman Index)
	- HHI for xA concentration
	- Number of unique players used
	"""
	
	# First, compute per-match totals for the team
	team_match = df.group_by(["league", "season", "team_id", "team", "game_id", "date"]).agg([
		# Total stats (already at player level, so sum gives team totals)
		pl.col("minutes").sum().alias("team_total_minutes"),
		pl.col("xg").sum().alias("team_total_xg"),
		pl.col("xa").sum().alias("team_total_xa"),
		pl.col("goals").sum().alias("team_total_goals"),
		pl.col("assists").sum().alias("team_total_assists"),
		pl.col("shots").sum().alias("team_total_shots"),
		pl.col("key_passes").sum().alias("team_total_key_passes"),
		
		# Unique players (depth indicator)
		pl.col("player_id").n_unique().alias("unique_players"),
		
		# For HHI calculation, we need the sum of squared shares
		# HHI = sum((player_xg / team_xg)^2) for all players with xg > 0
		# We'll compute this as sum(xg^2) / sum(xg)^2
		(pl.col("xg").pow(2).sum()).alias("sum_xg_squared"),
		(pl.col("xa").pow(2).sum()).alias("sum_xa_squared"),
		(pl.col("minutes").pow(2).sum()).alias("sum_minutes_squared"),
	])
	
	# Compute HHI (with safe division)
	team_match = team_match.with_columns([
		# HHI for xG: measures concentration of xG production
		# Higher = fewer players producing xG (more concentrated)
		pl.when(pl.col("team_total_xg") > 0)
		.then(pl.col("sum_xg_squared") / pl.col("team_total_xg").pow(2))
		.otherwise(pl.lit(None))
		.alias("xg_hhi"),
		
		# HHI for xA: measures concentration of xA production
		pl.when(pl.col("team_total_xa") > 0)
		.then(pl.col("sum_xa_squared") / pl.col("team_total_xa").pow(2))
		.otherwise(pl.lit(None))
		.alias("xa_hhi"),
		
		# HHI for minutes: measures rotation
		# Higher = fewer players getting minutes (less rotation)
		pl.when(pl.col("team_total_minutes") > 0)
		.then(pl.col("sum_minutes_squared") / pl.col("team_total_minutes").pow(2))
		.otherwise(pl.lit(None))
		.alias("minutes_hhi"),
	])
	
	# Compute per-90 stats (assuming ~90 minutes per match)
	team_match = team_match.with_columns([
		# xG per 90 - already at team level, this is just the team's xG
		pl.col("team_total_xg").alias("team_xg"),
		# xA per 90 - already at team level
		pl.col("team_total_xa").alias("team_xa"),
	])
	
	return team_match.drop(["sum_xg_squared", "sum_xa_squared", "sum_minutes_squared"])


def compute_rolling_features(team_match: pl.DataFrame) -> pl.DataFrame:
	"""Compute pre-match rolling features, excluding the row's own match."""
	
	# Sort by date within each team
	team_match = team_match.sort(["league", "team_id", "date"])
	
	# Define the stats we want rolling features for
	stats_to_roll = [
		"team_xg", "team_xa", "team_total_goals", "team_total_assists",
		"team_total_shots", "team_total_key_passes", "team_total_minutes",
		"xg_hhi", "xa_hhi", "minutes_hhi", "unique_players"
	]
	
	rolling_exprs = [
		pl.col(stat)
		.shift(1)
		.rolling_mean(window_size=MAX_WINDOW, min_samples=MIN_GAMES)
		.over(["league", "team_id"])
		.alias(f"{stat}_r{MAX_WINDOW}")
		for stat in stats_to_roll
	]
	rolling_exprs.append(
		pl.col("unique_players")
		.shift(1)
		.rolling_sum(window_size=5, min_samples=MIN_GAMES)
		.over(["league", "team_id"])
		.alias("unique_players_r5_sum")
	)
	return team_match.with_columns(rolling_exprs)


def build_player_team_features(
	df: pl.DataFrame,
	*,
	prediction_time: datetime | None = None,
) -> pl.DataFrame:
	"""Build historical pre-match features and, optionally, a current state for upcoming fixtures."""
	team_match = compute_team_match_aggregates(prepare_player_data(df))
	if prediction_time is not None:
		# One next-match row per team includes every completed match in the shifted window.
		# It becomes available now, never on a completed match's pre-match row.
		next_matches = team_match.select("league", "team_id").unique().with_columns(
			pl.lit(prediction_time).alias("date")
		)
		team_match = pl.concat([team_match, next_matches], how="diagonal")
	return compute_rolling_features(team_match)


def main():
	"""Test the player feature engineering pipeline."""
	print("Loading player data...")
	df = load_all_player_data()
	print(f"Loaded {len(df)} player-match records")
	
	print("\nBuilding player-derived team features...")
	team_match = build_player_team_features(df)
	print(f"Created {len(team_match)} team-match records")
	
	# Show sample of features
	feature_cols = [c for c in team_match.columns if c.endswith(f"_r{MAX_WINDOW}") or c.endswith("_r5_sum")]
	print(f"\nRolling features created: {feature_cols}")
	
	# Show sample
	print("\nSample data (EPL 2024-2025):")
	sample = team_match.filter(
		(pl.col("league") == "ENG-Premier League") & 
		(pl.col("season") == "2425")
	).sort("date").head(10)
	
	print(sample.select(["team", "date", "team_xg", "team_xa", "unique_players"] + feature_cols[:3]))
	
	# Show null counts for feature columns
	print("\nNull counts for rolling features:")
	for col in feature_cols[:5]:
		null_count = team_match.select(pl.col(col).is_null().sum()).item()
		total = len(team_match)
		print(f"  {col}: {null_count}/{total} ({100*null_count/total:.1f}% null)")
	
	return team_match


if __name__ == "__main__":
	main()
