import unittest
from itertools import product

import numpy as np

from training.evaluation.system_bets import (
	analyze_system_ticket,
	evaluate_system_bankroll_strategy,
	select_best_system_ticket,
	system_gross_return,
)


def _brute_ticket_objective(selected_probs: np.ndarray, selected_odds: np.ndarray, system_name: str, stake_share: float) -> float:
	total = 0.0
	for outcomes in product([0, 1], repeat=len(selected_probs)):
		wins = np.asarray(outcomes, dtype=bool)
		probability = 1.0
		for outcome, win_prob in zip(wins, selected_probs):
			probability *= float(win_prob if outcome else (1.0 - win_prob))
		wealth = 1.0 + float(stake_share) * (system_gross_return(selected_odds, wins, system_name) - 1.0)
		total += probability * np.log(wealth)
	return float(total)


class SystemBetTests(unittest.TestCase):
	def test_system_gross_return_matches_23_ticket_semantics(self):
		selected_odds = np.array([2.0, 3.0, 4.0])

		self.assertAlmostEqual(
			system_gross_return(selected_odds, np.array([True, True, False]), "2/3"),
			6.0 / 4.0,
			places=8,
		)
		self.assertAlmostEqual(
			system_gross_return(selected_odds, np.array([True, True, True]), "2/3"),
			(6.0 + 8.0 + 12.0 + 24.0) / 4.0,
			places=8,
		)

	def test_analyze_system_ticket_matches_brute_force_grid_for_23(self):
		selected_probs = np.array([0.58, 0.56, 0.54])
		selected_odds = np.array([2.10, 2.05, 2.20])

		ticket = analyze_system_ticket(selected_probs, selected_odds, system_name="2/3", kelly_fraction=1.0)
		grid = np.linspace(0.0, 0.99, 2001)
		values = np.array([
			_brute_ticket_objective(selected_probs, selected_odds, "2/3", stake_share=value)
			for value in grid
		])
		best_grid_idx = int(np.argmax(values))

		self.assertAlmostEqual(ticket["stake_share"], float(grid[best_grid_idx]), delta=0.01)
		self.assertAlmostEqual(ticket["scaled_log_growth"], float(values[best_grid_idx]), delta=1e-4)

	def test_select_best_system_ticket_picks_best_triplet(self):
		selection = {
			"best_index": np.array([0, 0, 0, 0]),
			"selected_probs": np.array([0.62, 0.60, 0.59, 0.51]),
			"selected_odds": np.array([2.15, 2.10, 2.05, 2.02]),
			"positive_mask": np.array([True, True, True, True]),
		}

		best = select_best_system_ticket(selection, system_name="2/3", kelly_fraction=1.0)

		self.assertIsNotNone(best)
		self.assertEqual(best["match_indices"].tolist(), [0, 1, 2])

	def test_evaluate_system_bankroll_strategy_compounds_daily_ticket_results(self):
		selection = {
			"best_index": np.array([0, 0, 0, 0, 0, 0]),
			"selected_probs": np.array([0.58, 0.57, 0.56, 0.58, 0.57, 0.56]),
			"selected_odds": np.array([2.10, 2.10, 2.10, 2.10, 2.10, 2.10]),
			"positive_mask": np.array([True, True, True, True, True, True]),
		}
		y_true = np.array([0, 0, 0, 1, 1, 1])
		groups = np.array(["2026-03-14", "2026-03-14", "2026-03-14", "2026-03-15", "2026-03-15", "2026-03-15"])

		ticket = analyze_system_ticket(
			selection["selected_probs"][:3],
			selection["selected_odds"][:3],
			system_name="2/3",
			kelly_fraction=1.0,
		)
		profit_day_one = 100.0 * ticket["stake_share"] * (
			system_gross_return(selection["selected_odds"][:3], np.array([True, True, True]), "2/3") - 1.0
		)
		bankroll_after_day_one = 100.0 + profit_day_one
		expected_final_bankroll = bankroll_after_day_one * (1.0 - ticket["stake_share"])

		metrics = evaluate_system_bankroll_strategy(
			selection=selection,
			y_true=y_true,
			system_name="2/3",
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)

		self.assertEqual(metrics["bankroll_bet_count"], 2)
		self.assertEqual(metrics["bankroll_line_count"], 8)
		self.assertAlmostEqual(metrics["final_bankroll"], expected_final_bankroll, places=6)


if __name__ == "__main__":
	unittest.main()