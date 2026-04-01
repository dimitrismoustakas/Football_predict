import unittest

import numpy as np

from training.models.set_transformer import DeepSetsTeamEncoder, SetTransformerTeamEncoder
from training.train_player_model import DEFAULT_CONFIG, build_player_model, shuffle_squad_features


class PlayerModelConfigTests(unittest.TestCase):
	def test_build_player_model_uses_deep_sets_encoder_and_implied_head(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "deep_sets"
		config["use_implied"] = True
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, DeepSetsTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3 + 3)

	def test_build_player_model_uses_set_transformer_without_implied_head(self):
		config = dict(DEFAULT_CONFIG)
		config["encoder_type"] = "set_transformer"
		config["use_implied"] = False
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3)

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