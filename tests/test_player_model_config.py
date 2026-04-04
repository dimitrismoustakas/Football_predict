import unittest

import torch

from preprocessing.lineup_builder import NUM_FEATURES
from training.models.set_transformer import SetTransformerTeamEncoder
from training.train_player_model import build_player_model, load_training_config


class PlayerModelConfigTests(unittest.TestCase):
	def make_config(self):
		return load_training_config()

	def test_build_player_model_uses_set_transformer_encoder_and_implied_head(self):
		config = self.make_config()
		config["encoder_type"] = "set_transformer"
		config["use_implied"] = True
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3 + 3)

	def test_build_player_model_without_implied_uses_plain_context_head(self):
		config = self.make_config()
		config["encoder_type"] = "set_transformer"
		config["use_implied"] = False
		model = build_player_model(config)

		self.assertIsInstance(model.team_encoder, SetTransformerTeamEncoder)
		self.assertEqual(model.head[0].in_features, config["team_output_dim"] * 3)

	def test_build_player_model_supports_gated_residual_head(self):
		config = self.make_config()
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
		config = self.make_config()
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
		config = self.make_config()
		config["encoder_type"] = "bad_encoder"
		with self.assertRaises(ValueError):
			build_player_model(config)


if __name__ == "__main__":
	unittest.main()
