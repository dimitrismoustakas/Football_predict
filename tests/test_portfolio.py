import unittest

import numpy as np

from utils.portfolio import allocate_fixed_budget, evaluate_budget_strategy, select_best_result_value


class PortfolioTests(unittest.TestCase):
	def test_allocate_fixed_budget_spends_full_budget_on_positive_ev_bets(self):
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
		allocation = allocate_fixed_budget(selection, total_budget=100.0, strategy="kelly", kelly_fraction=0.5)

		self.assertAlmostEqual(allocation["allocated_budget"], 100.0, places=6)
		self.assertAlmostEqual(allocation["stake_shares"].sum(), 1.0, places=6)
		self.assertTrue(np.all(allocation["stake_amounts"][~selection["positive_mask"]] == 0.0))
		self.assertTrue(np.all(allocation["stake_amounts"][selection["positive_mask"]] > 0.0))

	def test_budget_strategy_can_be_scored_by_group(self):
		probs = np.array([
			[0.50, 0.25, 0.25],
			[0.48, 0.26, 0.26],
		])
		y_true = np.array([0, 2])
		odds_home = np.array([2.20, 2.20])
		odds_draw = np.array([3.40, 3.40])
		odds_away = np.array([3.60, 3.60])
		groups = np.array(["2026-03-14", "2026-03-15"])

		metrics = evaluate_budget_strategy(
			probs=probs,
			y_true=y_true,
			odds_home=odds_home,
			odds_draw=odds_draw,
			odds_away=odds_away,
			groups=groups,
			strategy="flat",
			group_budget=1.0,
		)

		self.assertEqual(metrics["budget_active_groups"], 2)
		self.assertEqual(metrics["budget_bet_count"], 2)
		self.assertAlmostEqual(metrics["budget_total_staked"], 2.0, places=6)
		self.assertAlmostEqual(metrics["budget_profit"], 0.2, places=6)
		self.assertAlmostEqual(metrics["budget_roi"], 0.1, places=6)


if __name__ == "__main__":
	unittest.main()
