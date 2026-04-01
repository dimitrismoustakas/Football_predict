"""Tests for player rolling features, projected squad assembly, and tensor building."""

import unittest

import numpy as np
import polars as pl

from preprocessing.player_feature_engineering import (
	PLAYER_WINDOW,
	POSITION_TO_IDX,
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

	def test_max_players_per_team_per_match(self):
		top_n = 10
		squads = build_projected_squads(self.raw, self.rolling, top_n=top_n)
		# No team-match combo should have more than top_n players
		counts = squads.group_by(["game_id", "team_id"]).len()
		self.assertTrue((counts["len"] <= top_n).all(),
						f"Some teams have more than {top_n} players")

	def test_squad_rank_ordering(self):
		squads = build_projected_squads(self.raw, self.rolling, top_n=16)
		# squad_rank should be 1..N for each team-match
		for group in squads.group_by(["game_id", "team_id"]):
			rows = group[1] if isinstance(group, tuple) else group
			ranks = sorted(rows["squad_rank"].to_list())
			expected = list(range(1, len(ranks) + 1))
			self.assertEqual(ranks, expected)


class TestAssembleSquadTensors(unittest.TestCase):
	def setUp(self):
		self.raw = _make_player_data(n_matches_per_player=12, n_players=20)
		self.rolling = compute_player_rolling_features(self.raw)
		self.squads = build_projected_squads(self.raw, self.rolling, top_n=16)
		# Create a match_df
		self.match_df = pl.DataFrame({
			"game_id": list(range(1, 13)),
			"league": ["ENG-Premier League"] * 12,
			"home_team_id": [1] * 12,
			"away_team_id": [2] * 12,
		})

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


class TestSetTransformerModel(unittest.TestCase):
	"""Smoke tests for the model forward pass."""

	def test_deep_sets_forward(self):
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
			hidden_dim=32, team_output_dim=16, dropout=0.0)
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

	def test_set_transformer_forward(self):
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES, team_encoder_type="set_transformer",
			hidden_dim=32, team_output_dim=16, num_heads=2, num_sab_layers=1, dropout=0.0)
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

	def test_masking_changes_output(self):
		"""Masking out players should change the model output."""
		import torch
		from training.models.set_transformer import PlayerMatchModel

		model = PlayerMatchModel(
			input_dim=NUM_FEATURES, team_encoder_type="deep_sets",
			hidden_dim=32, team_output_dim=16, dropout=0.0)
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
