import unittest

import numpy as np

from training.result_modeling import (
	apply_implied_blend,
	blend_component_probabilities,
	blend_component_probabilities_by_regime,
	poisson_means_to_outcome_probs,
	tune_market_blend,
)


class ResultModelingTests(unittest.TestCase):
	def test_apply_implied_blend_supports_regime_specific_alpha(self):
		probs = np.array([
			[0.60, 0.20, 0.20],
			[0.20, 0.20, 0.60],
		])
		implied = np.array([
			[0.30, 0.40, 0.30],
			[0.50, 0.20, 0.30],
		])
		regime_alpha = {
			"feature": "draw_implied",
			"threshold": 0.30,
			"low_alpha": [1.0, 0.75, 0.35],
			"high_alpha": [1.0, 0.80, 0.35],
		}

		blended = apply_implied_blend(probs, implied, regime_alpha)
		expected = np.vstack([
			apply_implied_blend(probs[[0]], implied[[0]], regime_alpha["high_alpha"]),
			apply_implied_blend(probs[[1]], implied[[1]], regime_alpha["low_alpha"]),
		])

		np.testing.assert_allclose(blended, expected, atol=1e-9)

	def test_tune_market_blend_prefers_regime_grid_when_present(self):
		probs = np.array([
			[0.75, 0.15, 0.10],
			[0.70, 0.15, 0.15],
		])
		implied = np.array([
			[0.20, 0.65, 0.15],
			[0.70, 0.15, 0.15],
		])
		y_true = np.array([1, 0])
		training_config = {
			"blend_mode": "classwise",
			"class_blend_alpha_vector_grid": [
				[0.5, 0.5, 0.5],
			],
			"class_blend_alpha_regime_grid": [
				{
					"feature": "draw_implied",
					"threshold": 0.30,
					"low_alpha": 1.0,
					"high_alpha": 1.0,
				},
				{
					"feature": "draw_implied",
					"threshold": 0.30,
					"low_alpha": 1.0,
					"high_alpha": 0.0,
				},
			],
		}

		blend_mode, blend_alpha, blend_loss = tune_market_blend(training_config, probs, implied, y_true)

		self.assertEqual(blend_mode, "classwise")
		self.assertEqual(blend_alpha["feature"], "draw_implied")
		self.assertEqual(blend_alpha["threshold"], 0.30)
		self.assertEqual(blend_alpha["low_alpha"], 1.0)
		self.assertEqual(blend_alpha["high_alpha"], 0.0)
		self.assertLess(blend_loss, 0.5)

	def test_blend_component_probabilities_supports_regime_weights(self):
		component_a = np.array([
			[0.70, 0.20, 0.10],
			[0.20, 0.30, 0.50],
		])
		component_b = np.array([
			[0.20, 0.60, 0.20],
			[0.40, 0.20, 0.40],
		])
		implied = np.array([
			[0.30, 0.45, 0.25],
			[0.50, 0.20, 0.30],
		])
		regime_weight = {
			"feature": "draw_implied",
			"threshold": 0.30,
			"low_weight": [1.0, 0.9, 0.65],
			"high_weight": [1.0, 0.8, 0.35],
		}

		blended = blend_component_probabilities_by_regime(
			component_a,
			component_b,
			implied,
			regime_weight,
			mode="logit",
		)
		expected = np.vstack([
			blend_component_probabilities(component_a[[0]], component_b[[0]], regime_weight["high_weight"], mode="logit"),
			blend_component_probabilities(component_a[[1]], component_b[[1]], regime_weight["low_weight"], mode="logit"),
		])

		np.testing.assert_allclose(blended, expected, atol=1e-9)

	def test_poisson_means_to_outcome_probs_is_symmetric_for_equal_rates(self):
		probs = poisson_means_to_outcome_probs(
			np.array([1.4]),
			np.array([1.4]),
			max_goals=12,
		)

		self.assertAlmostEqual(float(probs[0, 0]), float(probs[0, 2]), places=9)
		self.assertAlmostEqual(float(probs.sum()), 1.0, places=9)

	def test_poisson_means_to_outcome_probs_moves_toward_higher_home_rate(self):
		probs = poisson_means_to_outcome_probs(
			np.array([2.1]),
			np.array([0.8]),
			max_goals=12,
		)

		self.assertGreater(float(probs[0, 0]), float(probs[0, 2]))
		self.assertGreater(float(probs[0, 0]), float(probs[0, 1]))


if __name__ == "__main__":
	unittest.main()
