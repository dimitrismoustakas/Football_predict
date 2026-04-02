import unittest

import numpy as np
import torch

from preprocessing.lineup_builder import NUM_FEATURES
from training.models.set_transformer import SetTransformerTeamEncoder
from training.train_player_model import DEFAULT_CONFIG, build_player_model, shuffle_squad_features


class PlayerModelConfigTests(unittest.TestCase):
	def test_build_player_model_uses_set_transformer_encoder_and_implied_head(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "set_transformer"
		config["use_implied"] = True
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3 + 3)

	def test_build_player_model_without_implied_uses_plain_context_head(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "set_transformer"
		config["use_implied"] = False
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3)

	def test_build_player_model_supports_gated_residual_head(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "set_transformer"
		config["head_type"] = "gated_residual"
		config["use_implied"] = True
		config["market_feature_stats"] = 4
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertIsNone(model.head)
		self.assertIsNotNone(model.residual_head)
		self.assertIsNotNone(model.gate_head)
		self.assertEqual(model.gate_head[0].in_features, config["team_output_dim"] * 3 + 7)

	def test_gated_residual_forward_pass_runs(self):
		config = dict(DEFAULT_CONFIG)
		config["head_type"] = "gated_residual"
		config["market_feature_stats"] = 4
		model = build_player_model(config)
		model.eval()

		batch = 3
		max_p = 6
		home_feat = torch.randn(batch, max_p, NUM_FEATURES)
		away_feat = torch.randn(batch, max_p, NUM_FEATURES)
		home_pos = torch.randint(1, 18, (batch, max_p))
		away_pos = torch.randint(1, 18, (batch, max_p))
		mask = torch.ones(batch, max_p, dtype=torch.bool)
		implied = torch.rand(batch, 3)
		implied = implied / implied.sum(dim=1, keepdim=True)
		raw_margin = torch.rand(batch)

		with torch.no_grad():
			logits, components = model(
				home_feat,
				home_pos,
				mask,
				away_feat,
				away_pos,
				mask,
				implied,
				raw_margin,
				return_components=True,
			)

		self.assertEqual(logits.shape, (batch, 3))
		self.assertEqual(components["gate"].shape, (batch, 3))

	def test_invalid_encoder_raises(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "bad_encoder"
		with self.assertRaises(ValueError):
			build_player_model(config)

	def test_shuffle_squad_features_preserves_mask_and_feature_multiset(self):
		squad_tensors = {
			"home_players": np.array([
				[[1.0, 10.0], [2.0, 20.0], [0.0, 0.0]],
				[[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
			], dtype=np.float32),
			"away_players": np.array([
				[[6.0, 60.0], [7.0, 70.0], [0.0, 0.0]],
				[[8.0, 80.0], [9.0, 90.0], [10.0, 100.0]],
			], dtype=np.float32),
			"home_mask": np.array([[True, True, False], [True, True, True]]),
			"away_mask": np.array([[True, True, False], [True, True, True]]),
			"home_positions": np.zeros((2, 3), dtype=np.int64),
			"away_positions": np.zeros((2, 3), dtype=np.int64),
			"game_ids": [1, 2],
		}

		shuffled = shuffle_squad_features(squad_tensors, seed=7)
		np.testing.assert_array_equal(shuffled["home_mask"], squad_tensors["home_mask"])
		np.testing.assert_array_equal(shuffled["away_mask"], squad_tensors["away_mask"])
		for side in ["home", "away"]:
			mask = squad_tensors[f"{side}_mask"]
			original = squad_tensors[f"{side}_players"]
			candidate = shuffled[f"{side}_players"]
			for feature_idx in range(original.shape[2]):
				original_values = np.sort(original[:, :, feature_idx][mask])
				candidate_values = np.sort(candidate[:, :, feature_idx][mask])
				np.testing.assert_allclose(original_values, candidate_values)


if __name__ == "__main__":
	unittest.main()
