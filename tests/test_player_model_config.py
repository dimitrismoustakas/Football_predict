import unittest

import torch

from preprocessing.lineup_builder import NUM_FEATURES
from training.models.set_transformer import PlayerMatchModel, SetTransformerTeamEncoder
from training.train_player_model import build_player_model, load_training_config


class PlayerModelConfigTests(unittest.TestCase):
	def make_config(self):
		return load_training_config()

	def test_build_player_model_uses_kept_head_dims(self):
		config = self.make_config()
		model = build_player_model(config)

		self.assertIsInstance(model, PlayerMatchModel)
		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3 + 3)
		self.assertEqual(len(model.cross_team_layers), config["num_cross_team_layers"])

	def test_build_player_model_forward_pass_runs(self):
		config = self.make_config()
		model = build_player_model(config)
		model.eval()

		batch = 2
		max_p = config["top_n_players"]
		home_feat = torch.randn(batch, max_p, NUM_FEATURES)
		away_feat = torch.randn(batch, max_p, NUM_FEATURES)
		home_pos = torch.randint(1, 18, (batch, max_p))
		away_pos = torch.randint(1, 18, (batch, max_p))
		home_mask = torch.ones(batch, max_p, dtype=torch.bool)
		away_mask = torch.ones(batch, max_p, dtype=torch.bool)
		home_mask[0, -2:] = False
		away_mask[1, -3:] = False
		home_pos[0, -2:] = 0
		away_pos[1, -3:] = 0
		implied = torch.rand(batch, 3)
		implied = implied / implied.sum(dim=1, keepdim=True)

		with torch.no_grad():
			logits = model(
				home_feat,
				home_pos,
				home_mask,
				away_feat,
				away_pos,
				away_mask,
				implied,
			)

		self.assertEqual(logits.shape, (batch, 3))
		self.assertTrue(torch.isfinite(logits).all())

	def test_negative_cross_team_layers_raise(self):
		with self.assertRaises(ValueError):
			PlayerMatchModel(input_dim=NUM_FEATURES, num_cross_team_layers=-1)


if __name__ == "__main__":
	unittest.main()
