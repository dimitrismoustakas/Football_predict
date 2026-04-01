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
	PLAYER_WINDOW,
	POSITION_TO_IDX,
	POSITION_VOCABULARY,
	compute_player_rolling_features,
	load_all_player_data,
	prepare_player_data,
)

# Feature columns produced by compute_player_rolling_features (continuous only)
PLAYER_FEATURE_COLS = [
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
	"red_card_prev_game",
	"season_yellow_cards",
]

POSITION_COL = f"most_common_position_r{PLAYER_WINDOW}"

NUM_POSITIONS = len(POSITION_VOCABULARY)  # 17
NUM_FEATURES = len(PLAYER_FEATURE_COLS)   # 10


def _compute_cumulative_minutes(raw_player_data: pl.DataFrame) -> pl.DataFrame:
	"""
	Compute cumulative minutes for each (team_id, player_id) up to each match.

	Returns a DataFrame with columns:
	  league, team_id, player_id, game_id, date, cumulative_minutes

	cumulative_minutes is shifted by 1 (excludes the current match) to prevent
	leakage — it represents total minutes played *before* this game.
	"""
	df = prepare_player_data(raw_player_data)
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


def build_projected_squads(
	raw_player_data: pl.DataFrame,
	player_rolling: pl.DataFrame,
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
	# Step 1: Cumulative minutes per (team, player) up to each match
	cum_minutes = _compute_cumulative_minutes(raw_player_data)

	# Step 2: Join rolling features onto cumulative minutes
	# player_rolling has one row per (player_id, game_id) with pre-match features
	join_cols = ["league", "team_id", "player_id", "game_id"]
	squads = cum_minutes.join(
		player_rolling,
		on=join_cols,
		how="inner",  # only keep players with rolling features available
	)

	# Step 3: Rank players within each (team, match) by cumulative minutes
	squads = squads.sort(["league", "team_id", "game_id", "cumulative_minutes"], descending=[False, False, False, True])
	squads = squads.with_columns(
		pl.col("cumulative_minutes")
		.rank(method="ordinal", descending=True)
		.over(["league", "team_id", "game_id"])
		.alias("squad_rank")
	)

	# Step 4: Keep only top-N players per team per match
	squads = squads.filter(pl.col("squad_rank") <= top_n)

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
	squads = build_projected_squads(raw, rolling, top_n=top_n)
	return assemble_squad_tensors(squads, match_df, max_players=top_n)
