import unittest

import torch

from training.models import FeatureBackbone, GatedResidualModel
from training.models.neural_net import _apply_true_class_surprise_scaling


class ResultModelingTests(unittest.TestCase):
	def test_gated_residual_model_uses_feature_backbone(self):
		model = GatedResidualModel(
			input_dim=6,
			hidden_layers=[8, 8],
			gate_hidden_dim=4,
			cross_layers=2,
		)
		x = torch.randn(4, 6)
		implied_probs = torch.softmax(torch.randn(4, 3), dim=-1)
		raw_margin = torch.rand(4, 1)

		logits = model(x, implied_probs=implied_probs, raw_margin=raw_margin)

		self.assertIsInstance(model.backbone, FeatureBackbone)
		self.assertEqual(tuple(logits.shape), (4, 3))
		self.assertEqual(model.backbone.hidden_dim, 14)

	def test_feature_backbone_requires_positive_cross_layers(self):
		with self.assertRaisesRegex(ValueError, "cross_layers must be positive for feature backbone"):
			FeatureBackbone(
				input_dim=6,
				hidden_layers=[8],
				cross_layers=0,
			)

	def test_gated_residual_model_no_longer_accepts_backbone_selector(self):
		with self.assertRaises(TypeError):
			GatedResidualModel(
				input_dim=6,
				hidden_layers=[8, 8],
				backbone_type="cross_resnet",
			)

	def test_true_class_surprise_scaling_is_noop_at_zero_scale(self):
		base_mix = torch.full((2, 1), 0.05)
		implied_probs = torch.tensor([[0.70, 0.20, 0.10], [0.20, 0.60, 0.20]], dtype=torch.float32)
		target = torch.tensor([0, 1])

		scaled = _apply_true_class_surprise_scaling(base_mix, implied_probs, target, scale=0.0)

		self.assertTrue(torch.equal(scaled, base_mix))

	def test_true_class_surprise_scaling_boosts_market_upsets_more(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.70, 0.20, 0.10],
			[0.20, 0.60, 0.20],
			[0.10, 0.20, 0.70],
		], dtype=torch.float32)
		target = torch.tensor([0, 1, 0])

		scaled = _apply_true_class_surprise_scaling(base_mix, implied_probs, target, scale=1.0)

		self.assertAlmostEqual(float(scaled[0, 0]), 0.13, places=6)
		self.assertAlmostEqual(float(scaled[1, 0]), 0.14, places=6)
		self.assertAlmostEqual(float(scaled[2, 0]), 0.19, places=6)


if __name__ == "__main__":
	unittest.main()
