"""Tests for player rolling features, projected squad assembly, and tensor building."""

import unittest

import numpy as np
import polars as pl

from preprocessing.player_feature_engineering import (
	ABSENCE_STREAK_SCORE_COL,
	CONSECUTIVE_MISSED_FEATURE_COL,
	PLAYER_WINDOW,
	POSITION_TO_IDX,
	RECENT_MINUTES_WINDOW,
	START_RATE_WINDOW,
	compute_player_rolling_features,
)
from preprocessing.lineup_builder import (
	NUM_FEATURES,
	PLAYER_FEATURE_COLS,
	_compute_cumulative_minutes,
	assemble_squad_tensors,
	build_projected_squads,
)


def _make_player_data(n_matches_per_player: int = 12, n_players: int = 20) -> pl.DataFrame:
	"""Create synthetic player-match data for testing."""
	rows = []
	for pid in range(1, n_players + 1):
		for m in range(n_matches_per_player):
			day = m + 1
			rows.append({
				"player_id": pid,
				"player": f"Player_{pid}",
				"team_id": 1 if pid <= 14 else 2,
				"team": "Team_A" if pid <= 14 else "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": m + 1,
				"match_id": f"2024-09-{day:02d} Team_A-Team_B",
				"minutes": float(60 + (pid % 5) * 5),
				"goals": float(pid % 3),
				"xg": float(pid % 3) * 0.3 + 0.1,
				"xa": float(pid % 4) * 0.15,
				"assists": float(pid % 4),
				"shots": float(2 + pid % 3),
				"key_passes": float(1 + pid % 2),
				"xg_chain": 0.5,
				"xg_buildup": 0.3,
				"yellow_cards": float(1 if m % 5 == 0 else 0),
				"red_cards": 0.0,
				"position": ["GK", "DC", "DC", "DL", "DR", "MC", "MC", "AMC",
							"FW", "FW", "Sub", "AML", "AMR", "DMC",
							"GK", "DC", "MC", "FW", "DL", "Sub"][pid - 1],
				"position_id": pid,
			})
	return pl.DataFrame(rows)


class TestPlayerRollingFeatures(unittest.TestCase):
	def setUp(self):
		self.raw = _make_player_data(n_matches_per_player=12, n_players=20)

	def test_output_has_expected_columns(self):
		rolling = compute_player_rolling_features(self.raw)
		for col in PLAYER_FEATURE_COLS:
			self.assertIn(col, rolling.columns, f"Missing column: {col}")
		self.assertIn(f"most_common_position_r{PLAYER_WINDOW}", rolling.columns)

	def test_shift_prevents_leakage(self):
		"""First PLAYER_MIN_GAMES rows per player should have null rolling features."""
		rolling = compute_player_rolling_features(self.raw)
		# Player 1 first 3 appearances should be null (min_samples=3, shift=1)
		p1 = rolling.filter(pl.col("player_id") == 1).sort("date")
		first_vals = p1[f"xg_per90_r{PLAYER_WINDOW}"].to_list()[:3]
		self.assertTrue(all(v is None for v in first_vals),
						f"Expected null for first 3 rows, got {first_vals}")

	def test_rolling_features_not_all_null(self):
		"""After enough appearances, rolling features should be populated."""
		rolling = compute_player_rolling_features(self.raw)
		p1 = rolling.filter(pl.col("player_id") == 1).sort("date")
		late_vals = p1[f"xg_per90_r{PLAYER_WINDOW}"].to_list()[-3:]
		self.assertTrue(all(v is not None for v in late_vals),
						f"Expected non-null for late rows, got {late_vals}")

	def test_cards_features_present(self):
		rolling = compute_player_rolling_features(self.raw)
		self.assertIn(f"yellow_cards_per90_r{PLAYER_WINDOW}", rolling.columns)
		self.assertIn(f"red_cards_per90_r{PLAYER_WINDOW}", rolling.columns)

	def test_position_column_populated(self):
		rolling = compute_player_rolling_features(self.raw)
		pos_col = f"most_common_position_r{PLAYER_WINDOW}"
		positions = rolling[pos_col].unique().to_list()
		# Should contain real positions (not all null)
		self.assertTrue(any(p is not None for p in positions))

	def test_recent_usage_features_are_shifted(self):
		raw = pl.DataFrame([
			{
				"player_id": 1,
				"player": "Player_1",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": idx + 1,
				"match_id": f"2024-09-0{idx + 1} Team_A-Team_B",
				"minutes": float(minutes),
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.0,
				"xg_buildup": 0.0,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": position,
				"position_id": 1,
			}
			for idx, (minutes, position) in enumerate([(10, "Sub"), (20, "Sub"), (30, "FW"), (40, "FW")])
		])

		rolling = compute_player_rolling_features(raw).sort("date")
		row = rolling.row(3, named=True)
		self.assertEqual(row["minutes_last_match"], 30.0)
		self.assertAlmostEqual(row[f"avg_minutes_r{RECENT_MINUTES_WINDOW}"], 20.0, places=6)
		self.assertAlmostEqual(row[f"start_rate_r{START_RATE_WINDOW}"], 1.0 / 3.0, places=6)
		self.assertAlmostEqual(row["log_team_cumulative_minutes"], np.log1p(60.0), places=6)

	def test_consecutive_team_matches_missed_tracks_intervening_team_games(self):
		raw = pl.DataFrame([
			{
				"player_id": 1,
				"player": "Player_1",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.3,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.2,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 99,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 99,
			},
			{
				"player_id": 99,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 2,
				"match_id": "2024-09-08 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 99,
			},
			{
				"player_id": 99,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 3,
				"match_id": "2024-09-15 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 99,
			},
			{
				"player_id": 1,
				"player": "Player_1",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 4,
				"match_id": "2024-09-22 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.3,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.2,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 99,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 4,
				"match_id": "2024-09-22 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 99,
			},
			{
				"player_id": 1,
				"player": "Player_1",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2526",
				"game_id": 5,
				"match_id": "2025-08-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.3,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.2,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 99,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2526",
				"game_id": 5,
				"match_id": "2025-08-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 99,
			},
			{
				"player_id": 2,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 2,
			},
			{
				"player_id": 2,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 2,
				"match_id": "2024-09-08 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 2,
			},
			{
				"player_id": 2,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 3,
				"match_id": "2024-09-15 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 2,
			},
			{
				"player_id": 2,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 4,
				"match_id": "2024-09-22 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 2,
			},
			{
				"player_id": 2,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2526",
				"game_id": 5,
				"match_id": "2025-08-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.0,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 2,
			},
		])

		rolling = compute_player_rolling_features(raw).sort(["season", "date"])
		player_rows = rolling.filter(pl.col("player_id") == 1)
		self.assertEqual(player_rows[CONSECUTIVE_MISSED_FEATURE_COL].to_list(), [0.0, 2.0, 0.0])
		self.assertAlmostEqual(player_rows[ABSENCE_STREAK_SCORE_COL][1], np.log1p(2.0), places=6)


class TestCumulativeMinutes(unittest.TestCase):
	def test_cumulative_excludes_current_match(self):
		"""Cumulative minutes should not include the current match (shifted)."""
		raw = _make_player_data(n_matches_per_player=5, n_players=2)
		cum = _compute_cumulative_minutes(raw)
		p1 = cum.filter(pl.col("player_id") == 1).sort("date")
		# First match should have 0 cumulative minutes
		self.assertEqual(p1["cumulative_minutes"][0], 0.0)
		# Second match should have minutes from first match only
		self.assertGreater(p1["cumulative_minutes"][1], 0.0)


class TestBuildProjectedSquads(unittest.TestCase):
	def setUp(self):
		self.raw = _make_player_data(n_matches_per_player=12, n_players=20)
		self.rolling = compute_player_rolling_features(self.raw)
		self.match_df = pl.DataFrame({
			"game_id": list(range(1, 13)),
			"league": ["ENG-Premier League"] * 12,
			"season": ["2425"] * 12,
			"date": [f"2024-09-{day:02d}" for day in range(1, 13)],
			"home_team_id": [1] * 12,
			"away_team_id": [2] * 12,
		}).with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

	def test_max_players_per_team_per_match(self):
		top_n = 10
		squads = build_projected_squads(self.raw, self.rolling, match_df=self.match_df, top_n=top_n)
		# No team-match combo should have more than top_n players
		counts = squads.group_by(["game_id", "team_id"]).len()
		self.assertTrue((counts["len"] <= top_n).all(),
						f"Some teams have more than {top_n} players")

	def test_squad_rank_ordering(self):
		squads = build_projected_squads(self.raw, self.rolling, match_df=self.match_df, top_n=16)
		# squad_rank should be 1..N for each team-match
		for group in squads.group_by(["game_id", "team_id"]):
			rows = group[1] if isinstance(group, tuple) else group
			ranks = sorted(rows["squad_rank"].to_list())
			expected = list(range(1, len(ranks) + 1))
			self.assertEqual(ranks, expected)

	def test_projected_squads_can_include_player_without_current_match_row(self):
		rows = [
			{
				"player_id": 1,
				"player": "Known_Starter",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.2,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.3,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 1,
				"player": "Known_Starter",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 2,
				"match_id": "2024-09-02 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.2,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.3,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 1,
				"player": "Known_Starter",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 3,
				"match_id": "2024-09-03 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.2,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.4,
				"xg_buildup": 0.3,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			},
			{
				"player_id": 2,
				"player": "Bench_Player",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 30.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.2,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 2,
			},
			{
				"player_id": 2,
				"player": "Bench_Player",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 2,
				"match_id": "2024-09-02 Team_A-Team_B",
				"minutes": 30.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.2,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 2,
			},
			{
				"player_id": 3,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 1,
				"match_id": "2024-09-01 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 1.0,
				"xg_chain": 0.2,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 3,
			},
			{
				"player_id": 3,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 2,
				"match_id": "2024-09-02 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 1.0,
				"xg_chain": 0.2,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 3,
			},
			{
				"player_id": 3,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": 3,
				"match_id": "2024-09-03 Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 1.0,
				"xg_chain": 0.2,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 3,
			},
		]
		raw = pl.DataFrame(rows)
		rolling = compute_player_rolling_features(raw)
		match_df = pl.DataFrame({
			"game_id": [1, 2, 3],
			"league": ["ENG-Premier League"] * 3,
			"season": ["2425"] * 3,
			"date": ["2024-09-01", "2024-09-02", "2024-09-03"],
			"home_team_id": [1] * 3,
			"away_team_id": [2] * 3,
		}).with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

		squads = build_projected_squads(raw, rolling, match_df=match_df, top_n=1)
		team_a_game_3 = squads.filter((pl.col("game_id") == 3) & (pl.col("team_id") == 1))
		self.assertEqual(team_a_game_3["player_id"].to_list(), [1])

	def test_projected_squad_features_exclude_current_match_stats(self):
		rows = []
		for game_id, date, player_id, team_id, team, position, xg in [
			(1, "2024-09-01", 1, 1, "Team_A", "FW", 0.1),
			(2, "2024-09-02", 1, 1, "Team_A", "FW", 0.2),
			(3, "2024-09-03", 1, 1, "Team_A", "FW", 0.3),
			(4, "2024-09-04", 1, 1, "Team_A", "FW", 10.0),
			(1, "2024-09-01", 2, 2, "Team_B", "DC", 0.1),
			(2, "2024-09-02", 2, 2, "Team_B", "DC", 0.1),
			(3, "2024-09-03", 2, 2, "Team_B", "DC", 0.1),
			(4, "2024-09-04", 2, 2, "Team_B", "DC", 0.1),
		]:
			rows.append({
				"player_id": player_id,
				"player": f"Player_{player_id}",
				"team_id": team_id,
				"team": team,
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": game_id,
				"match_id": f"{date} Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": xg,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.0,
				"xg_buildup": 0.0,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": position,
				"position_id": player_id,
			})

		raw = pl.DataFrame(rows)
		rolling = compute_player_rolling_features(raw)
		match_df = pl.DataFrame({
			"game_id": [1, 2, 3, 4],
			"league": ["ENG-Premier League"] * 4,
			"season": ["2425"] * 4,
			"date": ["2024-09-01", "2024-09-02", "2024-09-03", "2024-09-04"],
			"home_team_id": [1] * 4,
			"away_team_id": [2] * 4,
		}).with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

		squads = build_projected_squads(raw, rolling, match_df=match_df, top_n=1)
		team_a_game_4 = squads.filter((pl.col("game_id") == 4) & (pl.col("team_id") == 1))

		self.assertEqual(team_a_game_4.height, 1)
		expected_xg_per90 = (0.1 + 0.2 + 0.3) / (90.0 * 3) * 90.0
		self.assertAlmostEqual(team_a_game_4["xg_per90_r10"][0], expected_xg_per90, places=6)
		self.assertLess(team_a_game_4["xg_per90_r10"][0], 1.0)
		self.assertEqual(team_a_game_4["minutes_last_match"][0], 90.0)
		self.assertAlmostEqual(team_a_game_4[f"avg_minutes_r{RECENT_MINUTES_WINDOW}"][0], 90.0, places=6)
		self.assertAlmostEqual(team_a_game_4[f"start_rate_r{START_RATE_WINDOW}"][0], 1.0, places=6)
		self.assertAlmostEqual(team_a_game_4["log_team_cumulative_minutes"][0], np.log1p(270.0), places=6)

	def test_projected_squad_absence_feature_uses_pre_match_team_gaps(self):
		raw = pl.DataFrame([
			{
				"player_id": 1,
				"player": "Known_Starter",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": game_id,
				"match_id": f"{date} Team_A-Team_B",
				"minutes": minutes,
				"goals": 0.0,
				"xg": 0.2,
				"xa": 0.1,
				"assists": 0.0,
				"shots": 2.0,
				"key_passes": 1.0,
				"xg_chain": 0.3,
				"xg_buildup": 0.2,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "FW",
				"position_id": 1,
			}
			for game_id, date, minutes in [(1, "2024-09-01", 90.0), (4, "2024-09-22", 90.0)]
		] + [
			{
				"player_id": 2,
				"player": "Teammate",
				"team_id": 1,
				"team": "Team_A",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": game_id,
				"match_id": f"{date} Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "MC",
				"position_id": 2,
			}
			for game_id, date in [
				(1, "2024-09-01"),
				(2, "2024-09-08"),
				(3, "2024-09-15"),
				(4, "2024-09-22"),
			]
		] + [
			{
				"player_id": 3,
				"player": "Opponent",
				"team_id": 2,
				"team": "Team_B",
				"league": "ENG-Premier League",
				"season": "2425",
				"game_id": game_id,
				"match_id": f"{date} Team_A-Team_B",
				"minutes": 90.0,
				"goals": 0.0,
				"xg": 0.1,
				"xa": 0.0,
				"assists": 0.0,
				"shots": 1.0,
				"key_passes": 0.0,
				"xg_chain": 0.1,
				"xg_buildup": 0.1,
				"yellow_cards": 0.0,
				"red_cards": 0.0,
				"position": "DC",
				"position_id": 3,
			}
			for game_id, date in [
				(1, "2024-09-01"),
				(2, "2024-09-08"),
				(3, "2024-09-15"),
				(4, "2024-09-22"),
			]
		])
		rolling = compute_player_rolling_features(raw)
		match_df = pl.DataFrame({
			"game_id": [1, 2, 3, 4],
			"league": ["ENG-Premier League"] * 4,
			"season": ["2425"] * 4,
			"date": ["2024-09-01", "2024-09-08", "2024-09-15", "2024-09-22"],
			"home_team_id": [1] * 4,
			"away_team_id": [2] * 4,
		}).with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

		squads = build_projected_squads(raw, rolling, match_df=match_df, top_n=2)
		player_game_4 = squads.filter(
			(pl.col("game_id") == 4)
			& (pl.col("team_id") == 1)
			& (pl.col("player_id") == 1)
		)

		self.assertEqual(player_game_4.height, 1)
		self.assertEqual(player_game_4[CONSECUTIVE_MISSED_FEATURE_COL][0], 2.0)
		self.assertAlmostEqual(player_game_4[ABSENCE_STREAK_SCORE_COL][0], np.log1p(2.0), places=6)


class TestAssembleSquadTensors(unittest.TestCase):
	def setUp(self):
		self.raw = _make_player_data(n_matches_per_player=12, n_players=20)
		self.rolling = compute_player_rolling_features(self.raw)
		self.match_df = pl.DataFrame({
			"game_id": list(range(1, 13)),
			"league": ["ENG-Premier League"] * 12,
			"season": ["2425"] * 12,
			"date": [f"2024-09-{day:02d}" for day in range(1, 13)],
			"home_team_id": [1] * 12,
			"away_team_id": [2] * 12,
		}).with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))
		self.squads = build_projected_squads(self.raw, self.rolling, match_df=self.match_df, top_n=16)
		# Create a match_df
		self.match_df = self.match_df.select(["game_id", "league", "home_team_id", "away_team_id"])

	def test_tensor_shapes(self):
		max_p = 16
		tensors = assemble_squad_tensors(self.squads, self.match_df, max_players=max_p)
		n = len(tensors["game_ids"])
		self.assertEqual(tensors["home_players"].shape, (n, max_p, NUM_FEATURES))
		self.assertEqual(tensors["away_players"].shape, (n, max_p, NUM_FEATURES))
		self.assertEqual(tensors["home_positions"].shape, (n, max_p))
		self.assertEqual(tensors["away_positions"].shape, (n, max_p))
		self.assertEqual(tensors["home_mask"].shape, (n, max_p))
		self.assertEqual(tensors["away_mask"].shape, (n, max_p))

	def test_match_order_is_stable(self):
		"""Tensor assembly should not depend on the incoming match-row order."""

		reference = assemble_squad_tensors(self.squads, self.match_df, max_players=16)
		reversed_match_df = self.match_df.sort("game_id", descending=True)
		reordered = assemble_squad_tensors(self.squads, reversed_match_df, max_players=16)

		self.assertEqual(reference["game_ids"], reordered["game_ids"])
		for key in ["home_players", "away_players", "home_positions", "away_positions", "home_mask", "away_mask"]:
			np.testing.assert_array_equal(reference[key], reordered[key])

	def test_mask_consistency(self):
		"""Where mask is False, features and positions should be zero (padding)."""
		tensors = assemble_squad_tensors(self.squads, self.match_df, max_players=16)
		for side in ["home", "away"]:
			mask = tensors[f"{side}_mask"]
			feats = tensors[f"{side}_players"]
			pos = tensors[f"{side}_positions"]
			# Padded positions should be 0
			padded_pos = pos[~mask]
			if len(padded_pos) > 0:
				self.assertTrue((padded_pos == 0).all(),
								f"Padded {side} positions should be 0")

	def test_position_indices_valid(self):
		"""All position indices should be in valid range."""
		tensors = assemble_squad_tensors(self.squads, self.match_df, max_players=16)
		for side in ["home", "away"]:
			pos = tensors[f"{side}_positions"]
			self.assertTrue((pos >= 0).all())
			self.assertTrue((pos <= len(POSITION_TO_IDX)).all())


class TestPlayerSetModel(unittest.TestCase):
	"""Smoke tests for the model forward pass."""

	def test_set_transformer_forward(self):
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES,
			hidden_dim=32,
			team_output_dim=16,
			num_heads=2,
			num_sab_layers=1,
			dropout=0.0,
		)
		model.eval()

		batch = 4
		max_p = 16
		home_feat = torch.randn(batch, max_p, NUM_FEATURES)
		home_pos = torch.randint(0, 18, (batch, max_p))
		home_mask = torch.ones(batch, max_p, dtype=torch.bool)
		away_feat = torch.randn(batch, max_p, NUM_FEATURES)
		away_pos = torch.randint(0, 18, (batch, max_p))
		away_mask = torch.ones(batch, max_p, dtype=torch.bool)
		implied = torch.rand(batch, 3)

		with torch.no_grad():
			logits = model(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied)
		self.assertEqual(logits.shape, (batch, 3))

	def test_set_transformer_cross_team_forward(self):
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES,
			hidden_dim=32,
			team_output_dim=16,
			num_heads=2,
			num_sab_layers=1,
			dropout=0.0,
			num_cross_team_layers=1,
		)
		model.eval()

		batch = 4
		max_p = 16
		home_feat = torch.randn(batch, max_p, NUM_FEATURES)
		home_pos = torch.randint(0, 18, (batch, max_p))
		home_mask = torch.ones(batch, max_p, dtype=torch.bool)
		away_feat = torch.randn(batch, max_p, NUM_FEATURES)
		away_pos = torch.randint(0, 18, (batch, max_p))
		away_mask = torch.ones(batch, max_p, dtype=torch.bool)
		implied = torch.rand(batch, 3)
		implied = implied / implied.sum(dim=1, keepdim=True)

		with torch.no_grad():
			logits = model(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied)
		self.assertEqual(logits.shape, (batch, 3))

	def test_masking_changes_output(self):
		"""Masking out players should change the model output."""
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES,
			hidden_dim=32,
			team_output_dim=16,
			num_heads=2,
			num_sab_layers=1,
			dropout=0.0,
		)
		model.eval()

		batch = 2
		max_p = 16
		home_feat = torch.randn(batch, max_p, NUM_FEATURES)
		home_pos = torch.randint(1, 18, (batch, max_p))
		away_feat = torch.randn(batch, max_p, NUM_FEATURES)
		away_pos = torch.randint(1, 18, (batch, max_p))
		implied = torch.rand(batch, 3)

		full_mask = torch.ones(batch, max_p, dtype=torch.bool)
		partial_mask = torch.ones(batch, max_p, dtype=torch.bool)
		partial_mask[:, 8:] = False  # mask out half the players

		with torch.no_grad():
			out_full = model(home_feat, home_pos, full_mask, away_feat, away_pos, full_mask, implied)
			out_partial = model(home_feat, home_pos, partial_mask, away_feat, away_pos, partial_mask, implied)

		# Outputs should differ
		self.assertFalse(torch.allclose(out_full, out_partial),
						"Masking should change model output")


if __name__ == "__main__":
	unittest.main()
