import unittest

import numpy as np

from utils.portfolio import allocate_bankroll_kelly, evaluate_bankroll_strategy, select_best_result_value


class PortfolioTests(unittest.TestCase):
	def test_allocate_bankroll_kelly_does_not_force_full_deployment(self):
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


if __name__ == "__main__":
	unittest.main()
