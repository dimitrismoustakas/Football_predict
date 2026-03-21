import unittest

import torch

from training.models import FeatureBackbone, GatedResidualModel


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


if __name__ == "__main__":
	unittest.main()
