import unittest

import torch
import torch.nn.functional as F

from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.models import FeatureBackbone, GatedResidualModel, _log_softmax_from_implied
from training.models.neural_net import (
	_apply_true_class_surprise_scaling,
	_anchor_regret_penalty,
	_bi_tempered_logistic_loss,
	_bi_tempered_logistic_loss_autograd,
	_reverse_cross_entropy,
	_tempered_softmax,
	gated_loss,
)


class ResultModelingTests(unittest.TestCase):
	def test_gated_residual_model_uses_feature_backbone(self):
		model = GatedResidualModel(
			input_dim=6,
			hidden_layers=[8, 8],
			num_leagues=5,
			cross_layers=2,
		)
		x = torch.randn(4, 6)
		cat_features = torch.zeros(4, 3, dtype=torch.long)
		implied_probs = torch.softmax(torch.randn(4, 3), dim=-1)
		raw_margin = torch.rand(4, 1)

		logits = model(x, cat_features, implied_probs, raw_margin)

		self.assertIsInstance(model.backbone, FeatureBackbone)
		self.assertEqual(tuple(logits.shape), (4, 3))
		self.assertEqual(model.backbone.hidden_dim, 14)

	def test_anchor_regret_penalty_only_counts_harm_vs_anchor(self):
		final_log_probs = torch.log(torch.tensor([
			[0.70, 0.20, 0.10],
			[0.40, 0.30, 0.30],
		], dtype=torch.float32))
		anchor_logits = torch.log(torch.tensor([
			[0.60, 0.25, 0.15],
			[0.20, 0.30, 0.50],
		], dtype=torch.float32))
		target = torch.tensor([0, 2], dtype=torch.long)

		penalty = _anchor_regret_penalty(final_log_probs, anchor_logits, target)

		self.assertAlmostEqual(float(penalty[0].item()), 0.0, places=6)
		expected = float(-torch.log(torch.tensor(0.30 / 0.50)).item())
		self.assertAlmostEqual(float(penalty[1].item()), expected, places=6)

	def test_anchor_regret_penalty_respects_margin_and_power(self):
		final_log_probs = torch.log(torch.tensor([[0.45, 0.25, 0.30]], dtype=torch.float32))
		anchor_logits = torch.log(torch.tensor([[0.40, 0.20, 0.40]], dtype=torch.float32))
		target = torch.tensor([2], dtype=torch.long)

		penalty = _anchor_regret_penalty(final_log_probs, anchor_logits, target, margin=0.20, power=2.0)

		excess = max(0.0, float(-torch.log(torch.tensor(0.30 / 0.40)).item()) - 0.20)
		self.assertAlmostEqual(float(penalty.item()), excess**2, places=6)

	def test_gated_loss_anchor_regret_penalizes_harmful_deviations_more(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=1,
		)
		x = torch.zeros(1, 2)
		cat_features = torch.zeros(1, 3, dtype=torch.long)
		implied_probs = torch.tensor([[0.60, 0.25, 0.15]], dtype=torch.float32)
		raw_margin = torch.tensor([[1.0]], dtype=torch.float32)
		target = torch.tensor([2], dtype=torch.long)
		with torch.no_grad():
			model.backbone.final_layer.weight.zero_()
			model.gate_head.weight.zero_()
			model.gate_head.bias.zero_()
			model.gate_bias.fill_(10.0)
			model.backbone.final_layer.bias.copy_(torch.tensor([-2.0, 0.0, 1.5], dtype=torch.float32))

		helpful_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			anchor_regret_weight=0.5,
		)
		with torch.no_grad():
			model.backbone.final_layer.bias.copy_(torch.tensor([1.5, 0.0, -2.0], dtype=torch.float32))
		harmful_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			anchor_regret_weight=0.5,
		)

		self.assertLess(float(helpful_loss.item()), float(harmful_loss.item()))

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

	def test_reverse_cross_entropy_matches_clipped_one_hot_form(self):
		pred_probs = torch.tensor([[0.70, 0.20, 0.10]], dtype=torch.float32)
		target_distribution = F.one_hot(torch.tensor([0]), num_classes=3).float()

		loss = _reverse_cross_entropy(pred_probs, target_distribution, label_floor=1e-4)

		expected = -(0.20 + 0.10) * torch.log(torch.tensor(1e-4))
		self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)

	def test_reverse_cross_entropy_prefers_predictions_aligned_with_soft_target(self):
		target_distribution = torch.tensor([[0.80, 0.15, 0.05]], dtype=torch.float32)
		aligned = torch.tensor([[0.78, 0.16, 0.06]], dtype=torch.float32)
		misaligned = torch.tensor([[0.45, 0.30, 0.25]], dtype=torch.float32)

		aligned_loss = _reverse_cross_entropy(aligned, target_distribution, label_floor=1e-4)
		misaligned_loss = _reverse_cross_entropy(misaligned, target_distribution, label_floor=1e-4)

		self.assertLess(float(aligned_loss.item()), float(misaligned_loss.item()))

	def test_tempered_softmax_matches_softmax_at_unit_temperature(self):
		logits = torch.tensor([[1.2, -0.4, 0.1]], dtype=torch.float32)

		tempered = _tempered_softmax(logits, t=1.0)
		expected = torch.softmax(logits, dim=-1)

		self.assertTrue(torch.allclose(tempered, expected, atol=1e-6))

	def test_bi_tempered_loss_matches_cross_entropy_at_unit_temperatures(self):
		logits = torch.tensor([[1.2, -0.4, 0.1]], dtype=torch.float32)
		target_distribution = F.one_hot(torch.tensor([0]), num_classes=3).float()

		loss = _bi_tempered_logistic_loss(logits, target_distribution, t1=1.0, t2=1.0)
		expected = F.cross_entropy(logits, torch.tensor([0]), reduction="none")

		self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

	def test_bi_tempered_loss_matches_reference_forward_on_soft_targets(self):
		logits = torch.tensor(
			[
				[1.2, -0.4, 0.1],
				[-0.7, 0.3, 1.1],
			],
			dtype=torch.float64,
		)
		target_distribution = torch.tensor(
			[
				[0.82, 0.13, 0.05],
				[0.10, 0.35, 0.55],
			],
			dtype=torch.float64,
		)

		loss = _bi_tempered_logistic_loss(logits, target_distribution, t1=0.82, t2=1.05, num_iters=5)
		expected = _bi_tempered_logistic_loss_autograd(
			logits,
			target_distribution,
			t1=0.82,
			t2=1.05,
			num_iters=5,
		)

		self.assertTrue(torch.allclose(loss, expected, atol=1e-10, rtol=1e-8))

	def test_bi_tempered_loss_custom_backward_matches_reference_gradient(self):
		torch.manual_seed(0)
		logits = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
		reference_logits = logits.detach().clone().requires_grad_(True)
		target_distribution = torch.tensor(
			[
				[0.82, 0.13, 0.05],
				[0.10, 0.75, 0.15],
				[0.20, 0.35, 0.45],
				[0.60, 0.15, 0.25],
			],
			dtype=torch.float64,
		)

		loss = _bi_tempered_logistic_loss(logits, target_distribution, t1=0.82, t2=1.05, num_iters=5).sum()
		reference_loss = _bi_tempered_logistic_loss_autograd(
			reference_logits,
			target_distribution,
			t1=0.82,
			t2=1.05,
			num_iters=5,
		).sum()
		loss.backward()
		reference_loss.backward()

		self.assertTrue(torch.allclose(logits.grad, reference_logits.grad, atol=1e-10, rtol=1e-8))

	def test_bi_tempered_aux_weight_penalizes_misaligned_logits_more(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=1,
		)
		x = torch.zeros(1, 2)
		cat_features = torch.zeros(1, 3, dtype=torch.long)
		implied_probs = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
		raw_margin = torch.tensor([[1.0]], dtype=torch.float32)
		target = torch.tensor([0], dtype=torch.long)
		with torch.no_grad():
			model.backbone.final_layer.weight.zero_()
			model.backbone.final_layer.bias.copy_(torch.tensor([2.0, -1.0, -1.0], dtype=torch.float32))
			model.gate_head.weight.zero_()
			model.gate_head.bias.zero_()
			model.gate_bias.fill_(10.0)

		aligned_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			bi_tempered_mix_weight=0.5,
			bi_tempered_t1=0.9,
			bi_tempered_t2=1.05,
		)
		with torch.no_grad():
			model.backbone.final_layer.bias.copy_(torch.tensor([-2.0, 1.0, 1.0], dtype=torch.float32))
		misaligned_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			bi_tempered_mix_weight=0.5,
			bi_tempered_t1=0.9,
			bi_tempered_t2=1.05,
		)

		self.assertLess(float(aligned_loss.item()), float(misaligned_loss.item()))

	def test_brier_aux_weight_penalizes_misaligned_logits_more(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=1,
		)
		x = torch.zeros(1, 2)
		cat_features = torch.zeros(1, 3, dtype=torch.long)
		implied_probs = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float32)
		raw_margin = torch.tensor([[1.0]], dtype=torch.float32)
		target = torch.tensor([0], dtype=torch.long)
		with torch.no_grad():
			model.backbone.final_layer.weight.zero_()
			model.backbone.final_layer.bias.copy_(torch.tensor([2.0, -1.0, -1.0], dtype=torch.float32))
			model.gate_head.weight.zero_()
			model.gate_head.bias.zero_()
			model.gate_bias.fill_(10.0)

		aligned_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			brier_aux_weight=0.5,
		)
		with torch.no_grad():
			model.backbone.final_layer.bias.copy_(torch.tensor([-2.0, 1.0, 1.0], dtype=torch.float32))
		misaligned_loss = gated_loss(
			model,
			x,
			cat_features,
			implied_probs,
			target,
			raw_margin,
			gate_mean_weight=0.0,
			gate_sat_weight=0.0,
			brier_aux_weight=0.5,
		)

		self.assertLess(float(aligned_loss.item()), float(misaligned_loss.item()))

	def test_league_market_bias_mask_disables_selected_leagues(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=2,
			league_market_bias_enabled_leagues=[1],
		)
		with torch.no_grad():
			model.league_market_bias.weight.copy_(torch.tensor([
				[0.4, -0.2, -0.2],
				[0.1, 0.2, -0.3],
			], dtype=torch.float32))

		implied_probs = torch.tensor([
			[0.5, 0.3, 0.2],
			[0.5, 0.3, 0.2],
		], dtype=torch.float32)
		cat_features = torch.tensor([
			[0, 0, 0],
			[1, 0, 0],
		], dtype=torch.long)

		logits = model._compute_implied_logits(implied_probs, cat_features)
		base_logits = _log_softmax_from_implied(implied_probs)

		self.assertTrue(torch.allclose(logits[0], base_logits[0], atol=1e-6))
		self.assertTrue(torch.allclose(logits[1], base_logits[1] + torch.tensor([0.1, 0.2, -0.3]), atol=1e-6))

	def test_league_market_scale_mask_disables_selected_leagues(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=2,
			league_market_scale_enabled_leagues=[1],
		)
		with torch.no_grad():
			model.league_market_scale.weight.copy_(torch.log(torch.tensor([[2.0], [0.5]], dtype=torch.float32)))

		implied_probs = torch.tensor([
			[0.5, 0.3, 0.2],
			[0.5, 0.3, 0.2],
		], dtype=torch.float32)
		cat_features = torch.tensor([
			[0, 0, 0],
			[1, 0, 0],
		], dtype=torch.long)

		logits = model._compute_implied_logits(implied_probs, cat_features)
		base_logits = _log_softmax_from_implied(implied_probs)

		self.assertTrue(torch.allclose(logits[0], base_logits[0], atol=1e-6))
		self.assertTrue(torch.allclose(logits[1], 0.5 * base_logits[1], atol=1e-6))

	def test_league_market_class_scale_mask_disables_selected_leagues(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=2,
			league_market_class_scale_enabled_leagues=[1],
		)
		with torch.no_grad():
			model.league_market_class_scale.weight.copy_(torch.log(torch.tensor([
				[2.0, 2.0, 2.0],
				[1.5, 0.5, 1.0],
			], dtype=torch.float32)))
		implied_probs = torch.tensor([
			[0.5, 0.3, 0.2],
			[0.5, 0.3, 0.2],
		], dtype=torch.float32)
		cat_features = torch.tensor([
			[0, 0, 0],
			[1, 0, 0],
		], dtype=torch.long)

		logits = model._compute_implied_logits(implied_probs, cat_features)
		base_logits = _log_softmax_from_implied(implied_probs)

		self.assertTrue(torch.allclose(logits[0], base_logits[0], atol=1e-6))
		self.assertTrue(torch.allclose(logits[1], base_logits[1] * torch.tensor([1.5, 0.5, 1.0]), atol=1e-6))

	def test_league_market_logit_mixer_mask_disables_selected_leagues(self):
		model = GatedResidualModel(
			input_dim=2,
			hidden_layers=[4],
			num_leagues=2,
			league_market_logit_mixer_enabled_leagues=[1],
		)
		with torch.no_grad():
			model.league_market_logit_mixer.weight.zero_()
			model.league_market_logit_mixer.weight[1].copy_(torch.tensor([
				0.0, 0.2, 0.0,
				0.1, 0.0, -0.1,
				0.0, 0.3, 0.0,
			], dtype=torch.float32))

		implied_probs = torch.tensor([
			[0.5, 0.3, 0.2],
			[0.5, 0.3, 0.2],
		], dtype=torch.float32)
		cat_features = torch.tensor([
			[0, 0, 0],
			[1, 0, 0],
		], dtype=torch.long)

		logits = model._compute_implied_logits(implied_probs, cat_features)
		base_logits = _log_softmax_from_implied(implied_probs)
		enabled_mix = model.league_market_logit_mixer(cat_features[1:2, 0]).view(-1, 3, 3)
		enabled_mix = enabled_mix - torch.diag_embed(torch.diagonal(enabled_mix, dim1=-2, dim2=-1))
		expected_enabled = base_logits[1] + torch.bmm(base_logits[1:2].unsqueeze(1), enabled_mix).squeeze(1)[0]

		self.assertTrue(torch.allclose(logits[0], base_logits[0], atol=1e-6))
		self.assertTrue(torch.allclose(logits[1], expected_enabled, atol=1e-6))

	def test_load_model_bundle_loads_tracked_artifact(self):
		bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, torch.device("cpu"))

		self.assertEqual(bundle.model.num_leagues, 5)
		self.assertTrue(bundle.model.shared_gate)
		self.assertTrue(bundle.model.linear_gate)


if __name__ == "__main__":
	unittest.main()
