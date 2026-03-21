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

	def test_true_class_surprise_scaling_power_focuses_on_larger_upsets(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.70, 0.20, 0.10],
			[0.20, 0.60, 0.20],
			[0.10, 0.20, 0.70],
		], dtype=torch.float32)
		target = torch.tensor([0, 1, 0])

		scaled = _apply_true_class_surprise_scaling(base_mix, implied_probs, target, scale=1.0, power=2.0)

		self.assertAlmostEqual(float(scaled[0, 0]), 0.109, places=6)
		self.assertAlmostEqual(float(scaled[1, 0]), 0.116, places=6)
		self.assertAlmostEqual(float(scaled[2, 0]), 0.181, places=6)

	def test_true_class_surprise_scaling_floor_limits_boost_to_large_upsets(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.70, 0.20, 0.10],
			[0.20, 0.60, 0.20],
			[0.10, 0.20, 0.70],
		], dtype=torch.float32)
		target = torch.tensor([0, 1, 0])

		scaled = _apply_true_class_surprise_scaling(base_mix, implied_probs, target, scale=1.0, floor=0.5)

		self.assertAlmostEqual(float(scaled[0, 0]), 0.10, places=6)
		self.assertAlmostEqual(float(scaled[1, 0]), 0.10, places=6)
		self.assertAlmostEqual(float(scaled[2, 0]), 0.18, places=6)

	def test_true_class_surprise_scaling_can_override_draw_and_away_scale(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.70, 0.20, 0.10],
			[0.20, 0.60, 0.20],
			[0.20, 0.20, 0.60],
		], dtype=torch.float32)
		target = torch.tensor([0, 1, 2])

		scaled = _apply_true_class_surprise_scaling(
			base_mix,
			implied_probs,
			target,
			scale=1.0,
			draw_scale=0.5,
			away_scale=2.0,
		)

		self.assertAlmostEqual(float(scaled[0, 0]), 0.13, places=6)
		self.assertAlmostEqual(float(scaled[1, 0]), 0.12, places=6)
		self.assertAlmostEqual(float(scaled[2, 0]), 0.18, places=6)

	def test_true_class_surprise_scaling_can_override_draw_and_away_floor(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.70, 0.20, 0.10],
			[0.20, 0.60, 0.20],
			[0.20, 0.20, 0.60],
		], dtype=torch.float32)
		target = torch.tensor([0, 1, 2])

		scaled = _apply_true_class_surprise_scaling(
			base_mix,
			implied_probs,
			target,
			scale=1.0,
			draw_floor=0.30,
			away_floor=0.60,
		)

		self.assertAlmostEqual(float(scaled[0, 0]), 0.13, places=6)
		self.assertAlmostEqual(float(scaled[1, 0]), 0.1142857, places=6)
		self.assertAlmostEqual(float(scaled[2, 0]), 0.10, places=6)

	def test_true_class_surprise_scaling_logistic_saturates(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.80, 0.10, 0.10],
			[0.50, 0.25, 0.25],
			[0.20, 0.10, 0.70],
		], dtype=torch.float32)
		target = torch.tensor([0, 0, 0])

		scaled = _apply_true_class_surprise_scaling(
			base_mix,
			implied_probs,
			target,
			scale=1.0,
			mode="logistic",
			center=0.5,
			slope=12.0,
		)

		self.assertLess(float(scaled[0, 0]), float(scaled[1, 0]))
		self.assertLess(float(scaled[1, 0]), float(scaled[2, 0]))
		self.assertLessEqual(float(scaled[2, 0]), 0.20)

	def test_true_class_surprise_scaling_band_focuses_middle_surprises(self):
		base_mix = torch.full((3, 1), 0.10)
		implied_probs = torch.tensor([
			[0.80, 0.10, 0.10],
			[0.50, 0.25, 0.25],
			[0.20, 0.10, 0.70],
		], dtype=torch.float32)
		target = torch.tensor([0, 0, 0])

		scaled = _apply_true_class_surprise_scaling(
			base_mix,
			implied_probs,
			target,
			scale=1.0,
			mode="band",
			center=0.5,
			width=0.25,
			slope=30.0,
		)

		self.assertLess(float(scaled[0, 0]), float(scaled[1, 0]))
		self.assertLess(float(scaled[2, 0]), float(scaled[1, 0]))
		self.assertGreater(float(scaled[1, 0]), 0.18)


if __name__ == "__main__":
	unittest.main()
