"""
Projected squad assembly for player-level modelling.

Builds pre-match "expected squads" for each team by selecting the top-N players
by cumulative minutes up to (but not including) the match date.  This avoids
leaking actual lineup information into the model.

The module produces padded tensors suitable for set-based neural networks
(Deep Sets, Set Transformer).
"""

import numpy as np
import polars as pl

from preprocessing.player_feature_engineering import (
	ABSENCE_STREAK_SCORE_COL,
	CONSECUTIVE_MISSED_FEATURE_COL,
	PLAYER_MIN_GAMES,
	PLAYER_WINDOW,
	POSITION_TO_IDX,
	POSITION_VOCABULARY,
	RECENT_MINUTES_WINDOW,
	START_RATE_WINDOW,
	build_team_match_sequence_lookup,
	compute_player_rolling_features,
	load_all_player_data,
	prepare_player_data,
)

# Default continuous features fed into the player-set models.
BASE_PLAYER_FEATURE_COLS = [
	f"xg_per90_r{PLAYER_WINDOW}",
	f"xa_per90_r{PLAYER_WINDOW}",
	f"shots_per90_r{PLAYER_WINDOW}",
	f"key_passes_per90_r{PLAYER_WINDOW}",
	f"xg_chain_per90_r{PLAYER_WINDOW}",
	f"xg_buildup_per90_r{PLAYER_WINDOW}",
	f"yellow_cards_per90_r{PLAYER_WINDOW}",
	f"red_cards_per90_r{PLAYER_WINDOW}",
	f"avg_minutes_r{PLAYER_WINDOW}",
	f"appearances_r{PLAYER_WINDOW}",
	ABSENCE_STREAK_SCORE_COL,
	"red_card_prev_game",
	"season_yellow_cards",
]
PLAYER_FEATURE_COLS = list(BASE_PLAYER_FEATURE_COLS)
PLAYER_DYNAMIC_FEATURE_COLS = [CONSECUTIVE_MISSED_FEATURE_COL, ABSENCE_STREAK_SCORE_COL]
PLAYER_STATE_FEATURE_COLS = [col for col in PLAYER_FEATURE_COLS if col not in PLAYER_DYNAMIC_FEATURE_COLS]
PLAYER_AUX_STATE_COLS = [
	f"avg_minutes_r{RECENT_MINUTES_WINDOW}",
	"minutes_last_match",
	f"start_rate_r{START_RATE_WINDOW}",
	CONSECUTIVE_MISSED_FEATURE_COL,
	"log_team_cumulative_minutes",
]
PLAYER_DYNAMIC_AUX_STATE_COLS = [CONSECUTIVE_MISSED_FEATURE_COL]
PLAYER_STATE_AUX_COLS = [col for col in PLAYER_AUX_STATE_COLS if col not in PLAYER_DYNAMIC_AUX_STATE_COLS]

POSITION_COL = f"most_common_position_r{PLAYER_WINDOW}"

NUM_POSITIONS = len(POSITION_VOCABULARY)  # 17
NUM_FEATURES = len(PLAYER_FEATURE_COLS)

_PER90_STATS = [
	"xg",
	"xa",
	"shots",
	"key_passes",
	"xg_chain",
	"xg_buildup",
	"yellow_cards",
	"red_cards",
]


def _normalize_date_column(df: pl.DataFrame) -> pl.DataFrame:
	if "date" not in df.columns:
		return df
	return df.with_columns(pl.col("date").cast(pl.Datetime("us")))


def _compute_cumulative_minutes(raw_player_data: pl.DataFrame) -> pl.DataFrame:
	"""
	Compute cumulative minutes for each (team_id, player_id) up to each match.

	Returns a DataFrame with columns:
	  league, team_id, player_id, game_id, date, cumulative_minutes

	cumulative_minutes is shifted by 1 (excludes the current match) to prevent
	leakage — it represents total minutes played *before* this game.
	"""
	df = prepare_player_data(raw_player_data)
	df = _normalize_date_column(df)
	df = df.sort(["league", "team_id", "player_id", "date"])

	df = df.with_columns(
		pl.col("minutes")
		.shift(1)
		.cum_sum()
		.over(["league", "team_id", "player_id"])
		.alias("cumulative_minutes")
	)
	# First appearance for a player on a team gets 0 cumulative minutes
	df = df.with_columns(pl.col("cumulative_minutes").fill_null(0.0))

	return df.select([
		"league", "team_id", "player_id", "game_id", "date", "cumulative_minutes",
	])


def _compute_cumulative_minutes_history(raw_player_data: pl.DataFrame) -> pl.DataFrame:
	"""Compute cumulative minutes after each appearance for leak-free as-of joins."""

	df = prepare_player_data(raw_player_data)
	df = _normalize_date_column(df)
	df = df.sort(["league", "team_id", "player_id", "date"])
	df = df.with_columns(
		pl.col("minutes")
		.cum_sum()
		.over(["league", "team_id", "player_id"])
		.alias("cumulative_minutes")
	)
	return df.select(["league", "team_id", "player_id", "date", "cumulative_minutes"])


def _compute_player_state_history(raw_player_data: pl.DataFrame) -> pl.DataFrame:
	"""Compute end-of-appearance player state for leak-free as-of joins."""

	df = prepare_player_data(raw_player_data)
	df = _normalize_date_column(df)
	if "position" not in df.columns:
		raise ValueError("Player data must include a 'position' column")
	for col in ["yellow_cards", "red_cards"]:
		if col not in df.columns:
			df = df.with_columns(pl.lit(0.0).alias(col))
		else:
			df = df.with_columns(pl.col(col).cast(pl.Float64))
	df = df.sort(["player_id", "date"])
	df = df.with_columns([
		pl.lit(1.0).alias("_one"),
		pl.when(pl.col("position") != "Sub").then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("_starter_flag"),
	])

	rolling_exprs = []
	for stat in _PER90_STATS:
		if stat not in df.columns:
			continue
		rolling_exprs.append(
			pl.col(stat)
			.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
			.over("player_id")
			.alias(f"_sum_{stat}")
		)
	rolling_exprs.append(
		pl.col("minutes")
		.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias("_sum_minutes")
	)
	rolling_exprs.append(
		pl.col("minutes")
		.rolling_mean(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias(f"avg_minutes_r{PLAYER_WINDOW}")
	)
	rolling_exprs.append(
		pl.col("_one")
		.rolling_sum(window_size=PLAYER_WINDOW, min_samples=PLAYER_MIN_GAMES)
		.over("player_id")
		.alias(f"appearances_r{PLAYER_WINDOW}")
	)
	rolling_exprs.append(
		pl.col("minutes")
		.rolling_mean(window_size=RECENT_MINUTES_WINDOW, min_samples=1)
		.over("player_id")
		.alias(f"avg_minutes_r{RECENT_MINUTES_WINDOW}")
	)
	rolling_exprs.append(
		pl.col("minutes")
		.alias("minutes_last_match")
	)
	rolling_exprs.append(
		pl.col("_starter_flag")
		.rolling_mean(window_size=START_RATE_WINDOW, min_samples=1)
		.over("player_id")
		.alias(f"start_rate_r{START_RATE_WINDOW}")
	)
	df = df.with_columns(rolling_exprs)

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

	df = df.with_columns(
		pl.when(pl.col("red_cards") > 0)
		.then(pl.lit(1.0))
		.otherwise(pl.lit(0.0))
		.alias("red_card_prev_game")
	)
	df = df.with_columns(
		pl.col("yellow_cards")
		.cum_sum()
		.over(["player_id", "season"])
		.fill_null(0.0)
		.alias("season_yellow_cards")
	)
	df = df.with_columns(
		pl.when(pl.col("position") != "Sub")
		.then(pl.col("position"))
		.otherwise(pl.lit(None))
		.alias("_starter_position")
	)
	df = df.with_columns(
		pl.col("_starter_position")
		.forward_fill()
		.over("player_id")
		.alias(POSITION_COL)
	)
	df = df.with_columns(pl.col(POSITION_COL).fill_null("Sub"))
	df = df.sort(["league", "team_id", "player_id", "date"])
	df = df.with_columns(
		pl.col("minutes")
		.cum_sum()
		.over(["league", "team_id", "player_id"])
		.log1p()
		.alias("log_team_cumulative_minutes")
	)
	team_match_sequence = build_team_match_sequence_lookup(raw_player_data)
	df = df.join(
		team_match_sequence,
		on=["league", "season", "team_id", "game_id", "date"],
		how="left",
	)
	df = df.with_columns([
		pl.col("team_match_sequence").alias("last_appearance_team_match_sequence"),
		pl.col("season").alias("last_appearance_season"),
	])

	return df.select([
		"league",
		"team_id",
		"player_id",
		"date",
		*PLAYER_STATE_FEATURE_COLS,
		*PLAYER_STATE_AUX_COLS,
		POSITION_COL,
		"last_appearance_team_match_sequence",
		"last_appearance_season",
	])


def _build_team_match_keys(raw_player_data: pl.DataFrame, match_df: pl.DataFrame | None) -> pl.DataFrame:
	prepared = prepare_player_data(raw_player_data)
	prepared = _normalize_date_column(prepared)
	if match_df is None:
		return prepared.select(["league", "season", "game_id", "date", "team_id"]).unique()

	date_lookup = prepared.select(["game_id", "date"]).unique()
	season_lookup = prepared.select(["game_id", "season"]).unique()
	match_keys = _normalize_date_column(match_df)
	if "date" not in match_keys.columns:
		match_keys = match_keys.join(date_lookup, on="game_id", how="left")
	if "season" not in match_keys.columns:
		match_keys = match_keys.join(season_lookup, on="game_id", how="left")
	home_keys = match_keys.select(["league", "season", "game_id", "date", "home_team_id"]).rename({"home_team_id": "team_id"})
	away_keys = match_keys.select(["league", "season", "game_id", "date", "away_team_id"]).rename({"away_team_id": "team_id"})
	return pl.concat([home_keys, away_keys], how="vertical_relaxed").unique()


def build_projected_squads(
	raw_player_data: pl.DataFrame,
	player_rolling: pl.DataFrame,
	match_df: pl.DataFrame | None = None,
	top_n: int = 16,
) -> pl.DataFrame:
	"""
	Build projected squads: top-N players per team by pre-match cumulative minutes.

	Args:
		raw_player_data: Raw player-match data (from load_all_player_data)
		player_rolling: Per-player rolling features (from compute_player_rolling_features)
		top_n: Number of players to select per team per match

	Returns:
		DataFrame with one row per (game_id, team_id, player_id), containing:
		- Identification columns: league, season, team_id, game_id, date, player_id
		- Rolling feature columns (continuous)
		- most_common_position_r10 (categorical)
		- cumulative_minutes (for sorting/ranking)
		- squad_rank (1 = most minutes, top_n = least among selected)
	"""
	_ = player_rolling
	team_match_keys = _build_team_match_keys(raw_player_data, match_df)
	team_match_sequence = build_team_match_sequence_lookup(raw_player_data)
	team_match_keys = team_match_keys.join(
		team_match_sequence,
		on=["league", "season", "team_id", "game_id", "date"],
		how="left",
	)
	team_player_pool = prepare_player_data(raw_player_data).select(["league", "team_id", "player_id"]).unique()
	candidate_rows = team_match_keys.join(team_player_pool, on=["league", "team_id"], how="inner")
	candidate_rows = candidate_rows.sort(["league", "team_id", "player_id", "date"])

	cum_minutes_history = _compute_cumulative_minutes_history(raw_player_data).sort(["league", "team_id", "player_id", "date"])
	player_state_history = _compute_player_state_history(raw_player_data).sort(["league", "team_id", "player_id", "date"])

	squads = candidate_rows.join_asof(
		cum_minutes_history,
		on="date",
		by=["league", "team_id", "player_id"],
		strategy="backward",
		allow_exact_matches=False,
	)
	squads = squads.join_asof(
		player_state_history,
		on="date",
		by=["league", "team_id", "player_id"],
		strategy="backward",
		allow_exact_matches=False,
	)
	squads = squads.filter(pl.col("cumulative_minutes").is_not_null())
	squads = squads.with_columns(
		pl.when(
			pl.col("last_appearance_team_match_sequence").is_not_null()
			& (pl.col("last_appearance_season") == pl.col("season"))
		)
		.then(
			(
				pl.col("team_match_sequence").cast(pl.Int64)
				- pl.col("last_appearance_team_match_sequence").cast(pl.Int64)
				- 1
			)
			.clip(lower_bound=0)
			.cast(pl.Float64)
		)
		.otherwise(pl.lit(0.0))
		.alias(CONSECUTIVE_MISSED_FEATURE_COL)
	)
	squads = squads.with_columns(
		(
			pl.col(CONSECUTIVE_MISSED_FEATURE_COL).log1p()
			* pl.col(f"start_rate_r{START_RATE_WINDOW}").fill_null(0.0)
		)
		.cast(pl.Float64)
		.alias(ABSENCE_STREAK_SCORE_COL)
	)

	# Step 3: Rank players within each (team, match) by cumulative minutes
	squads = squads.sort(["league", "team_id", "game_id", "cumulative_minutes", "player_id"], descending=[False, False, False, True, False])
	squads = squads.with_columns(
		pl.col("cumulative_minutes")
		.rank(method="ordinal", descending=True)
		.over(["league", "team_id", "game_id"])
		.alias("squad_rank")
	)

	# Step 4: Keep only top-N players per team per match
	squads = squads.filter(pl.col("squad_rank") <= top_n)
	squads = squads.drop(["team_match_sequence", "last_appearance_team_match_sequence", "last_appearance_season"])

	return squads


def _position_to_index(position_series: pl.Series) -> np.ndarray:
	"""Convert position strings to integer indices (0 = padding)."""
	return np.array([
		POSITION_TO_IDX.get(p, 0) for p in position_series.to_list()
	], dtype=np.int64)


def assemble_squad_tensors(
	squads_df: pl.DataFrame,
	match_df: pl.DataFrame,
	max_players: int = 16,
) -> dict:
	"""
	Assemble padded tensors for all matches from projected squad data.

	Args:
		squads_df: Output from build_projected_squads()
		match_df: Match-level DataFrame with game_id, league, home_team_id, away_team_id
		max_players: Maximum players per team (tensors are padded to this size)

	Returns:
		Dictionary with:
		- home_players: (N, max_players, D) float32
		- away_players: (N, max_players, D) float32
		- home_positions: (N, max_players) int64
		- away_positions: (N, max_players) int64
		- home_mask: (N, max_players) bool
		- away_mask: (N, max_players) bool
		- game_ids: list of game_id values (for alignment with match data)
	"""
	# Get ordered list of matches
	matches = match_df.select(["game_id", "league", "home_team_id", "away_team_id"]).unique()
	game_ids = matches["game_id"].to_list()
	n_matches = len(game_ids)

	# Pre-allocate output arrays
	home_players = np.zeros((n_matches, max_players, NUM_FEATURES), dtype=np.float32)
	away_players = np.zeros((n_matches, max_players, NUM_FEATURES), dtype=np.float32)
	home_positions = np.zeros((n_matches, max_players), dtype=np.int64)
	away_positions = np.zeros((n_matches, max_players), dtype=np.int64)
	home_mask = np.zeros((n_matches, max_players), dtype=bool)
	away_mask = np.zeros((n_matches, max_players), dtype=bool)

	# Build lookup: (game_id, team_id) -> squad rows
	# Group squads by game_id and team_id for fast lookup
	squad_groups = {}
	for row in squads_df.sort("squad_rank").iter_rows(named=True):
		key = (row["game_id"], row["team_id"])
		if key not in squad_groups:
			squad_groups[key] = []
		squad_groups[key].append(row)

	# Fill tensors
	for i, match_row in enumerate(matches.iter_rows(named=True)):
		gid = match_row["game_id"]
		home_tid = match_row["home_team_id"]
		away_tid = match_row["away_team_id"]

		for side, tid, feat_arr, pos_arr, mask_arr in [
			("home", home_tid, home_players, home_positions, home_mask),
			("away", away_tid, away_players, away_positions, away_mask),
		]:
			squad = squad_groups.get((gid, tid), [])
			n_players = min(len(squad), max_players)

			for j in range(n_players):
				player = squad[j]
				# Fill continuous features
				for k, col in enumerate(PLAYER_FEATURE_COLS):
					val = player.get(col)
					feat_arr[i, j, k] = 0.0 if val is None else float(val)
				# Fill position
				pos_arr[i, j] = POSITION_TO_IDX.get(player.get(POSITION_COL, "Sub"), 0)
				mask_arr[i, j] = True

	return {
		"home_players": home_players,
		"away_players": away_players,
		"home_positions": home_positions,
		"away_positions": away_positions,
		"home_mask": home_mask,
		"away_mask": away_mask,
		"game_ids": game_ids,
	}


def load_and_build_squad_tensors(
	match_df: pl.DataFrame,
	top_n: int = 16,
) -> dict:
	"""
	End-to-end: load player data, compute features, build projected squad tensors.

	Args:
		match_df: Match-level DataFrame (must have game_id, league, home_team_id, away_team_id)
		top_n: Number of players per team

	Returns:
		Tensor dictionary from assemble_squad_tensors()
	"""
	raw = load_all_player_data()
	rolling = compute_player_rolling_features(raw)
	squads = build_projected_squads(raw, rolling, match_df=match_df, top_n=top_n)
	return assemble_squad_tensors(squads, match_df, max_players=top_n)
