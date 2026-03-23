import unittest
from itertools import product

import numpy as np

from utils.portfolio import (
	_joint_expected_log_growth_and_grad,
	allocate_bankroll_kelly,
	evaluate_bankroll_strategy,
	get_joint_quadrature_rule,
	select_best_result_value,
)


def _expected_log_wealth(weights: np.ndarray, probs: np.ndarray, odds: np.ndarray) -> float:
	total = 0.0
	for outcomes in product([0, 1], repeat=len(weights)):
		prob = 1.0
		wealth = 1.0
		for weight, outcome, win_prob, decimal_odds in zip(weights, outcomes, probs, odds):
			prob *= win_prob if outcome else (1.0 - win_prob)
			wealth += (decimal_odds - 1.0) * weight if outcome else -weight
		total += prob * np.log(wealth)
	return float(total)


class PortfolioTests(unittest.TestCase):
	def test_allocate_bankroll_kelly_uses_joint_optimization_without_forcing_full_deployment(self):
		probs = np.array([
			[0.50, 0.25, 0.25],
			[0.46, 0.27, 0.27],
			[0.33, 0.34, 0.33],
		])
		odds_matrix = np.array([
			[2.20, 3.40, 3.60],
			[2.35, 3.20, 3.10],
			[2.80, 2.80, 2.80],
		])

		selection = select_best_result_value(probs, odds_matrix)
		allocation = allocate_bankroll_kelly(selection, total_bankroll=100.0, kelly_fraction=0.5)

		self.assertLess(allocation["allocated_budget"], 100.0)
		self.assertLess(allocation["stake_shares"].sum(), 1.0)
		self.assertTrue(np.all(allocation["stake_amounts"][~selection["positive_mask"]] == 0.0))
		self.assertTrue(np.all(allocation["stake_amounts"][selection["positive_mask"]] > 0.0))

	def test_bankroll_strategy_compounds_by_group(self):
		probs = np.array([
			[0.75, 0.15, 0.10],
			[0.75, 0.15, 0.10],
		])
		y_true = np.array([0, 1])
		odds_home = np.array([2.0, 2.0])
		odds_draw = np.array([4.0, 4.0])
		odds_away = np.array([8.0, 8.0])
		groups = np.array(["2026-03-14", "2026-03-15"])

		metrics = evaluate_bankroll_strategy(
			probs=probs,
			y_true=y_true,
			odds_home=odds_home,
			odds_draw=odds_draw,
			odds_away=odds_away,
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)

		self.assertEqual(metrics["bankroll_bet_count"], 2)
		self.assertAlmostEqual(metrics["bankroll_roi"], -0.25, places=6)
		self.assertAlmostEqual(metrics["max_drawdown"], 0.5, places=6)

	def test_bankroll_drawdown_tracks_peak_to_trough_loss(self):
		probs = np.array([
			[0.75, 0.15, 0.10],
			[0.75, 0.15, 0.10],
			[0.75, 0.15, 0.10],
		])
		y_true = np.array([0, 1, 1])
		odds_home = np.array([2.0, 2.0, 2.0])
		odds_draw = np.array([4.0, 4.0, 4.0])
		odds_away = np.array([8.0, 8.0, 8.0])
		groups = np.array(["2026-03-14", "2026-03-15", "2026-03-16"])

		metrics = evaluate_bankroll_strategy(
			probs=probs,
			y_true=y_true,
			odds_home=odds_home,
			odds_draw=odds_draw,
			odds_away=odds_away,
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)

		self.assertEqual(metrics["bankroll_bet_count"], 3)
		self.assertAlmostEqual(metrics["bankroll_roi"], -0.625, places=6)
		self.assertAlmostEqual(metrics["max_drawdown"], 0.75, places=6)

	def test_joint_quadrature_matches_exact_expected_log_growth_on_small_slate(self):
		weights = np.array([0.11, 0.07, 0.05])
		probs = np.array([0.58, 0.54, 0.61])
		odds = np.array([2.05, 2.15, 1.90])
		nodes, rule_weights = get_joint_quadrature_rule()

		approx, _ = _joint_expected_log_growth_and_grad(
			weights=weights,
			selected_probs=probs,
			selected_odds=odds,
			quadrature_nodes=nodes,
			quadrature_weights=rule_weights,
		)
		exact = _expected_log_wealth(weights, probs, odds)

		np.testing.assert_allclose(approx, exact, rtol=1e-7, atol=1e-10)

	def test_joint_quadrature_rule_returns_read_only_cached_arrays(self):
		nodes, rule_weights = get_joint_quadrature_rule()

		self.assertFalse(nodes.flags.writeable)
		self.assertFalse(rule_weights.flags.writeable)
		with self.assertRaises(ValueError):
			nodes[0] = 0.0
		with self.assertRaises(ValueError):
			rule_weights[0] = 0.0

	def test_allocate_bankroll_kelly_beats_independent_full_kelly_on_crowded_slate(self):
		probs = np.array([
			[0.70, 0.15, 0.15],
			[0.70, 0.15, 0.15],
			[0.70, 0.15, 0.15],
		])
		odds_matrix = np.array([
			[1.80, 4.50, 4.50],
			[1.80, 4.50, 4.50],
			[1.80, 4.50, 4.50],
		])

		selection = select_best_result_value(probs, odds_matrix)
		allocation = allocate_bankroll_kelly(selection, total_bankroll=1.0, kelly_fraction=1.0)
		independent_full_kelly = selection["full_kelly"]

		joint_log = _expected_log_wealth(allocation["stake_shares"], selection["selected_probs"], selection["selected_odds"])
		independent_log = _expected_log_wealth(independent_full_kelly, selection["selected_probs"], selection["selected_odds"])

		self.assertGreater(joint_log, independent_log)
		self.assertTrue(np.allclose(allocation["stake_shares"], allocation["stake_shares"][0], atol=1e-8))
		self.assertTrue(np.all(allocation["stake_shares"] < independent_full_kelly))

	def test_identical_large_slate_keeps_identical_weights(self):
		n_bets = 20
		probs = np.tile(np.array([[0.55, 0.225, 0.225]]), (n_bets, 1))
		odds = np.tile(np.array([[2.10, 4.00, 4.00]]), (n_bets, 1))

		selection = select_best_result_value(probs, odds)
		allocation = allocate_bankroll_kelly(selection, total_bankroll=100.0, kelly_fraction=0.5)
		positive = allocation["stake_shares"][selection["positive_mask"]]

		self.assertEqual(len(positive), n_bets)
		self.assertAlmostEqual(float(positive.max() - positive.min()), 0.0, places=8)


if __name__ == "__main__":
	unittest.main()
