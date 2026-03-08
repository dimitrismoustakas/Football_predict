"""
Shared orchestration for building match-level feature tables.
"""

import json
from pathlib import Path

import polars as pl

from preprocessing.feature_engineering import (
	add_categorical_features,
	build_long,
	build_match_level,
	build_promoted_teams_set,
	compute_adjusted_rolling_features,
	compute_adjusted_stats,
	compute_opponent_baselines,
	compute_rolling_features,
	compute_schedule_features,
	join_opponent_baselines,
	load_promoted_teams,
	merge_european_schedule,
)
from preprocessing.player_feature_engineering import build_player_team_features, load_all_player_data

BASE_MATCH_COLUMNS = [
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


def apply_team_name_mapping(
	lf: pl.LazyFrame,
	mapping_path: Path,
	label: str,
) -> pl.LazyFrame:
	"""Apply canonical team-name mapping if the mapping file exists."""

	if not mapping_path.exists():
		return lf

	with open(mapping_path, "r", encoding="utf-8") as file:
		mapping = json.load(file)

	print(f"Applying canonical team mapping to {label}...")
	return lf.with_columns([
		pl.col("home_team").replace(mapping).alias("home_team"),
		pl.col("away_team").replace(mapping).alias("away_team"),
	])


def select_base_matches(lf: pl.LazyFrame) -> pl.LazyFrame:
	"""Select the match-level columns needed by the shared feature pipeline."""

	schema = lf.collect_schema()
	have = set(schema.names())
	base_cols = [col for col in BASE_MATCH_COLUMNS if col in have]
	return lf.select(base_cols)


def add_schedule_features(
	long_feats: pl.LazyFrame,
	european_schedule_path: Path,
) -> pl.LazyFrame:
	"""Add fixture congestion features, or null placeholders if schedule data is unavailable."""

	if european_schedule_path.exists():
		print("Merging European schedule for fixture congestion features...")
		combined_long = merge_european_schedule(long_feats, european_schedule_path)
		print("Computing schedule features...")
		combined_df = compute_schedule_features(combined_long)
		domestic_with_schedule = combined_df.filter(~pl.col("is_european"))
		schedule_feats = domestic_with_schedule.select([
			"match_id",
			"team",
			"days_since_last_match",
			"games_last_15_days",
		])
		return long_feats.collect().join(schedule_feats, on=["match_id", "team"], how="left").lazy()

	print("No European schedule found, skipping fixture congestion features")
	return long_feats.with_columns([
		pl.lit(None).cast(pl.Float64).alias("days_since_last_match"),
		pl.lit(None).cast(pl.Int64).alias("games_last_15_days"),
	])


def build_match_features_from_lf(
	lf: pl.LazyFrame,
	european_schedule_path: Path,
) -> pl.LazyFrame:
	"""Build shared long-format and match-level engineered features from a normalized match frame."""

	base_matches = select_base_matches(lf)
	long_feats = build_long(base_matches)
	long_feats = compute_rolling_features(long_feats)
	long_feats = compute_opponent_baselines(long_feats)
	long_feats = join_opponent_baselines(long_feats)
	long_feats = compute_adjusted_stats(long_feats)
	long_feats = compute_adjusted_rolling_features(long_feats)
	long_feats = add_schedule_features(long_feats, european_schedule_path)
	return build_match_level(base_matches, long_feats)


def get_player_feature_columns(player_team_features: pl.DataFrame) -> list[str]:
	"""Return the canonical subset of player-derived feature columns."""

	return [col for col in player_team_features.columns if "_r15" in col or "_r5_sum" in col]


def join_player_features_by_game_id(
	match_df: pl.DataFrame,
	player_team_features: pl.DataFrame,
) -> pl.DataFrame:
	"""Join player features using exact `(league, team_id, game_id)` keys."""

	player_feature_cols = get_player_feature_columns(player_team_features)
	if "home_team_id" not in match_df.columns or "away_team_id" not in match_df.columns:
		print("Warning: Missing team_id columns, skipping player feature join")
		return match_df

	print(f"Joining {len(player_feature_cols)} player features for home and away teams...")
	match_df = match_df.with_columns(pl.col("league").cast(pl.Utf8))

	home_player_feats = player_team_features.select(
		["league", "team_id", "game_id"] + player_feature_cols
	).rename({"team_id": "home_team_id"})
	home_player_feats = home_player_feats.rename({col: f"home_{col}" for col in player_feature_cols})

	away_player_feats = player_team_features.select(
		["league", "team_id", "game_id"] + player_feature_cols
	).rename({"team_id": "away_team_id"})
	away_player_feats = away_player_feats.rename({col: f"away_{col}" for col in player_feature_cols})

	match_df = match_df.join(
		home_player_feats,
		left_on=["league", "home_team_id", "game_id"],
		right_on=["league", "home_team_id", "game_id"],
		how="left",
	).join(
		away_player_feats,
		left_on=["league", "away_team_id", "game_id"],
		right_on=["league", "away_team_id", "game_id"],
		how="left",
	)
	print(
		f"Added player features: home columns = {len([col for col in match_df.columns if col.startswith('home_') and '_r15' in col])}"
	)
	return match_df


def join_player_features_asof(
	match_df: pl.DataFrame,
	player_team_features: pl.DataFrame,
) -> pl.DataFrame:
	"""Join latest available player features using `(league, team_id, date)` as-of keys."""

	player_feature_cols = get_player_feature_columns(player_team_features)
	if "home_team_id" not in match_df.columns or "away_team_id" not in match_df.columns:
		print("Warning: Missing team_id columns, skipping player feature join")
		return match_df

	print(f"Joining {len(player_feature_cols)} player features for home and away teams...")
	match_df = match_df.with_columns([
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

	match_df = match_df.sort(["league", "home_team_id", "date"]).join_asof(
		home_player_feats.sort(["league", "home_team_id", "date"]),
		on="date",
		by=["league", "home_team_id"],
		strategy="backward",
	)
	match_df = match_df.sort(["league", "away_team_id", "date"]).join_asof(
		away_player_feats.sort(["league", "away_team_id", "date"]),
		on="date",
		by=["league", "away_team_id"],
		strategy="backward",
	)
	return match_df


def add_match_categorical_features(match_df: pl.DataFrame) -> pl.DataFrame:
	"""Add categorical features to the engineered match frame."""

	print("Adding categorical features...")
	promoted_lookup = build_promoted_teams_set(load_promoted_teams())
	return add_categorical_features(match_df.lazy(), promoted_lookup).collect()


def load_player_features() -> pl.DataFrame:
	"""Build the shared player-derived team feature table."""

	print("Building player-derived team features...")
	return build_player_team_features(load_all_player_data())
