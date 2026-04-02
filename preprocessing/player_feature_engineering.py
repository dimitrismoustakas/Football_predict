"""
Player-level feature engineering for team ability estimation.

Features are computed using player performance data aggregated at the team level.
All features use a 15-game lookback window with a minimum of 5 games to compute.

Feature Categories:
1. Expected Abilities (xG90, xA90) - minute-weighted team averages
2. Concentration Metrics (HHI for xG, xA) - how concentrated is output among players
3. Squad Depth (unique players, rotation index)
4. Availability (total minutes available)
"""

import polars as pl
from pathlib import Path


# Configuration
MAX_WINDOW = 15  # Maximum games to look back
MIN_GAMES = 2    # Minimum games required to compute features (matches base features)


def load_all_player_data(data_root: Path = Path("data/understat")) -> pl.DataFrame:
	"""Load all player match stats from parquet files."""
	player_files = sorted(data_root.rglob("player_match_stats.parquet"), key=lambda path: str(path))
	
	if not player_files:
		raise FileNotFoundError(f"No player_match_stats.parquet files found in {data_root}")
	
	dfs = [pl.read_parquet(f) for f in player_files]
	df = pl.concat(dfs, how="diagonal")
	
	return df


def extract_match_date(match_id: str) -> str:
	"""Extract date from match_id format: '2024-08-16 Manchester United-Fulham'."""
	# The match_id format is: "YYYY-MM-DD TeamA-TeamB"
	return match_id[:10]


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
	"""
	Compute rolling features with 15-game window and 5-game minimum.
	
	Uses an "expanding then capped" approach:
	- Games 1-4: null (insufficient data)
	- Games 5-14: use all available history (expanding window)
	- Games 15+: use last 15 games (rolling window)
	
	All features are shifted by 1 to prevent data leakage.
	"""
	
	# Sort by date within each team
	team_match = team_match.sort(["league", "team_id", "date"])
	
	# Define the stats we want rolling features for
	stats_to_roll = [
		"team_xg", "team_xa", "team_total_goals", "team_total_assists",
		"team_total_shots", "team_total_key_passes", "team_total_minutes",
		"xg_hhi", "xa_hhi", "minutes_hhi", "unique_players"
	]
	
	# Create rolling features using Polars rolling functions
	# We use rolling_mean with window size of MAX_WINDOW and min_periods of MIN_GAMES
	rolling_exprs = []
	
	for stat in stats_to_roll:
		if stat in team_match.columns:
			# Shift by 1 first, then compute rolling mean
			rolling_exprs.append(
				pl.col(stat)
				.shift(1)
				.rolling_mean(window_size=MAX_WINDOW, min_samples=MIN_GAMES)
				.over(["league", "team_id"])
				.alias(f"{stat}_r{MAX_WINDOW}")
			)
			
			# Also compute rolling sum for some stats (like unique_players as cumulative)
			if stat in ["unique_players"]:
				rolling_exprs.append(
					pl.col(stat)
					.shift(1)
					.rolling_sum(window_size=5, min_samples=MIN_GAMES)
					.over(["league", "team_id"])
					.alias(f"{stat}_r5_sum")
				)
	
	# Apply rolling computations
	team_match = team_match.with_columns(rolling_exprs)
	
	return team_match


def build_player_team_features(df: pl.DataFrame) -> pl.DataFrame:
	"""
	Main function to build player-derived team features.
	
	Returns a DataFrame with one row per (team, match) with rolling features.
	"""
	# Prepare data
	df = prepare_player_data(df)
	
	# Aggregate to team-match level
	team_match = compute_team_match_aggregates(df)
	
	# Compute rolling features
	team_match = compute_rolling_features(team_match)
	
	return team_match


def merge_with_match_data(
	player_features: pl.DataFrame,
	match_data: pl.DataFrame
) -> pl.DataFrame:
	"""
	Merge player-derived team features with match-level data.
	
	This joins the rolling player features for both home and away teams
	to each match row.
	
	Args:
		player_features: Output from build_player_team_features()
		match_data: Match-level DataFrame with home_team_id and away_team_id
	
	Returns:
		Match data with player features for both teams
	"""
	# Select only the rolling feature columns for merging
	feature_cols = [c for c in player_features.columns if c.endswith(f"_r{MAX_WINDOW}") or c.endswith("_r5_sum")]
	
	# Create home team features
	home_features = player_features.select(
		["league", "team_id", "game_id"] + feature_cols
	).rename({col: f"home_{col}" for col in feature_cols})
	home_features = home_features.rename({"team_id": "home_team_id"})
	
	# Create away team features
	away_features = player_features.select(
		["league", "team_id", "game_id"] + feature_cols
	).rename({col: f"away_{col}" for col in feature_cols})
	away_features = away_features.rename({"team_id": "away_team_id"})
	
	# Merge with match data
	result = match_data.join(
		home_features,
		on=["league", "home_team_id", "game_id"],
		how="left"
	).join(
		away_features,
		on=["league", "away_team_id", "game_id"],
		how="left"
	)
	
	return result


# ---------------------------------------------------------------------------
# Per-player rolling features (for Set Transformer / player-level modelling)
# ---------------------------------------------------------------------------

PLAYER_WINDOW = 10   # Rolling window over player's personal appearances
PLAYER_MIN_GAMES = 3 # Minimum appearances to compute rolling stats
RECENT_MINUTES_WINDOW = 3
START_RATE_WINDOW = 5
CONSECUTIVE_MISSED_FEATURE_COL = "consecutive_team_matches_missed"
ABSENCE_STREAK_SCORE_COL = "absence_streak_score"

# Stats to normalise per 90 minutes
_PER90_STATS = ["xg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup",
                "yellow_cards", "red_cards"]

# All positions from Understat (excluding "Sub" which we handle separately)
POSITION_VOCABULARY = [
	"GK", "DC", "DL", "DR", "DML", "DMR", "DMC",
	"MC", "ML", "MR", "AMC", "AML", "AMR",
	"FW", "FWL", "FWR", "Sub",
]
# Mapping: position string -> integer index (0 = padding, 1-17 = positions)
POSITION_TO_IDX = {pos: idx + 1 for idx, pos in enumerate(POSITION_VOCABULARY)}


def build_team_match_sequence_lookup(df: pl.DataFrame) -> pl.DataFrame:
	"""Enumerate each team's matches within a season in chronological order."""

	prepared = prepare_player_data(df)
	return (
		prepared
		.select(["league", "season", "team_id", "game_id", "date"])
		.unique()
		.sort(["league", "season", "team_id", "date", "game_id"])
		.with_columns(
			pl.col("game_id")
			.cum_count()
			.over(["league", "season", "team_id"])
			.alias("team_match_sequence")
		)
	)


def compute_player_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
	"""
	Compute per-player rolling features over their last N personal appearances.

	For each player, across all their matches (regardless of team), compute:
	- Per-90 stats: xg, xa, shots, key_passes, xg_chain, xg_buildup, yellow/red cards
	- avg_minutes: rolling mean of minutes played
	- appearances: count of games in the rolling window
	- most_common_position: mode of position over the rolling window

	All features are shifted by 1 to prevent data leakage.

	Returns a DataFrame with one row per (player_id, game_id) containing
	the player's pre-match rolling features.
	"""
	df = prepare_player_data(df)

	# Ensure position column exists
	if "position" not in df.columns:
		raise ValueError("Player data must include a 'position' column")

	# Ensure card columns exist (fill with 0 if missing)
	for col in ["yellow_cards", "red_cards"]:
		if col not in df.columns:
			df = df.with_columns(pl.lit(0).cast(pl.Float64).alias(col))
		else:
			df = df.with_columns(pl.col(col).cast(pl.Float64))

	# Sort by player, then date (for rolling over personal history)
	df = df.sort(["player_id", "date"])

	# Add a constant column for counting appearances in rolling window
	df = df.with_columns([
		pl.lit(1.0).alias("_one"),
		pl.when(pl.col("position") != "Sub").then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("_starter_flag"),
	])

	# --- Per-90 stats: shift first, then rolling sum of raw + rolling sum of minutes ---
	# per90 = (sum of stat over window) / (sum of minutes over window) * 90
	rolling_exprs = []

	for stat in _PER90_STATS:
		if stat not in df.columns:
			continue
		# Numerator: rolling sum of raw stat (shifted by 1)
		rolling_exprs.append(
			pl.col(stat)
			.shift(1)
			.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
			.over("player_id")
			.alias(f"_sum_{stat}")
		)

	# Denominator: rolling sum of minutes (shifted by 1)
	rolling_exprs.append(
		pl.col("minutes")
		.shift(1)
		.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias("_sum_minutes")
	)

	# Average minutes per appearance
	rolling_exprs.append(
		pl.col("minutes")
		.shift(1)
		.rolling_mean(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias(f"avg_minutes_r{PLAYER_WINDOW}")
	)

	# Appearance count in window
	rolling_exprs.append(
		pl.col("_one")
		.shift(1)
		.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias(f"appearances_r{PLAYER_WINDOW}")
	)
	rolling_exprs.append(
		pl.col("minutes")
		.shift(1)
		.rolling_mean(window_size=RECENT_MINUTES_WINDOW, min_samples=1)
		.over("player_id")
		.alias(f"avg_minutes_r{RECENT_MINUTES_WINDOW}")
	)
	rolling_exprs.append(
		pl.col("minutes")
		.shift(1)
		.over("player_id")
		.alias("minutes_last_match")
	)
	rolling_exprs.append(
		pl.col("_starter_flag")
		.shift(1)
		.rolling_mean(window_size=START_RATE_WINDOW, min_samples=1)
		.over("player_id")
		.alias(f"start_rate_r{START_RATE_WINDOW}")
	)

	df = df.with_columns(rolling_exprs)

	# Compute per-90 rates with safe division (minimum 45 total minutes)
	per90_exprs = []
	for stat in _PER90_STATS:
		if f"_sum_{stat}" not in df.columns:
			continue
		per90_exprs.append(
			pl.when(pl.col("_sum_minutes") >= 45)
			.then(pl.col(f"_sum_{stat}") / pl.col("_sum_minutes") * 90.0)
			.otherwise(pl.lit(None))
			.alias(f"{stat}_per90_r{PLAYER_WINDOW}")
		)
	df = df.with_columns(per90_exprs)

	# --- Discipline features ---
	# Red card in previous game (binary, shifted by 1)
	df = df.with_columns(
		pl.when(pl.col("red_cards").shift(1).over("player_id") > 0)
		.then(pl.lit(1.0))
		.otherwise(pl.lit(0.0))
		.alias("red_card_prev_game")
	)

	# Cumulative yellow cards this season (shifted by 1 to prevent leakage)
	df = df.with_columns(
		pl.col("yellow_cards")
		.shift(1)
		.cum_sum()
		.over(["player_id", "season"])
		.fill_null(0.0)
		.alias("season_yellow_cards")
	)

	# --- Most common position over rolling window ---
	# We compute this by finding the mode of position in the last N appearances.
	# Polars doesn't have a rolling mode, so we use a struct-based approach:
	# shift positions, then for each row take the last PLAYER_WINDOW values.
	# For simplicity, we use the most recent non-Sub position as a proxy.
	# If all recent positions are Sub, fall back to Sub.
	df = df.with_columns(
		pl.when(pl.col("position") != "Sub")
		.then(pl.col("position"))
		.otherwise(pl.lit(None))
		.alias("_starter_position")
	)
	# Forward-fill the last starter position within each player
	df = df.with_columns(
		pl.col("_starter_position")
		.shift(1)
		.forward_fill()
		.over("player_id")
		.alias(f"most_common_position_r{PLAYER_WINDOW}")
	)
	# Fall back to "Sub" if no starter position found
	df = df.with_columns(
		pl.col(f"most_common_position_r{PLAYER_WINDOW}")
		.fill_null("Sub")
	)

	# Team-specific tenure / usage signal.
	# Keep this numerically compact because player-set models do not standardize inputs.
	df = df.sort(["league", "team_id", "player_id", "date"])
	df = df.with_columns(
		pl.col("minutes")
		.shift(1)
		.cum_sum()
		.over(["league", "team_id", "player_id"])
		.fill_null(0.0)
		.log1p()
		.alias("log_team_cumulative_minutes")
	)
	team_match_sequence = build_team_match_sequence_lookup(df)
	df = df.join(
		team_match_sequence,
		on=["league", "season", "team_id", "game_id", "date"],
		how="left",
	)
	df = df.sort(["league", "season", "team_id", "player_id", "date", "game_id"])
	df = df.with_columns(
		pl.col("team_match_sequence")
		.shift(1)
		.over(["league", "season", "team_id", "player_id"])
		.alias("_previous_team_match_sequence")
	)
	df = df.with_columns(
		pl.when(pl.col("_previous_team_match_sequence").is_not_null())
		.then(
			(
				pl.col("team_match_sequence").cast(pl.Int64)
				- pl.col("_previous_team_match_sequence").cast(pl.Int64)
				- 1
			)
			.clip(lower_bound=0)
			.cast(pl.Float64)
		)
		.otherwise(pl.lit(0.0))
		.alias(CONSECUTIVE_MISSED_FEATURE_COL)
	)
	df = df.with_columns(
		(
			pl.col(CONSECUTIVE_MISSED_FEATURE_COL).log1p()
			* pl.col(f"start_rate_r{START_RATE_WINDOW}").fill_null(0.0)
		)
		.cast(pl.Float64)
		.alias(ABSENCE_STREAK_SCORE_COL)
	)

	# Select output columns
	id_cols = ["player_id", "league", "season", "team_id", "team", "game_id", "date"]
	feature_cols = (
		[f"{stat}_per90_r{PLAYER_WINDOW}" for stat in _PER90_STATS if f"_sum_{stat}" in df.columns]
		+ [f"avg_minutes_r{PLAYER_WINDOW}", f"appearances_r{PLAYER_WINDOW}",
		   f"avg_minutes_r{RECENT_MINUTES_WINDOW}", "minutes_last_match", f"start_rate_r{START_RATE_WINDOW}",
		   CONSECUTIVE_MISSED_FEATURE_COL, ABSENCE_STREAK_SCORE_COL, "red_card_prev_game", "season_yellow_cards",
		   "log_team_cumulative_minutes", f"most_common_position_r{PLAYER_WINDOW}"]
	)

	return df.select(id_cols + feature_cols)


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
