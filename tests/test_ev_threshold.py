import unittest

import numpy as np

from training.evaluation.ev_threshold import (
	apply_ev_threshold,
	build_group_labels,
	evaluate_selection_bankroll_strategy,
	fit_monotone_ev_threshold,
	selection_to_bet_records,
	selection_bankroll_path,
)
from utils.portfolio import evaluate_bankroll_strategy, select_best_result_value


class EvThresholdTests(unittest.TestCase):
	def test_fit_monotone_ev_threshold_finds_zero_crossing(self):
		raw_ev = np.array([0.01, 0.02, 0.03, 0.04], dtype=float)
		realized_roi = np.array([-0.05, -0.01, 0.02, 0.05], dtype=float)

		fitted = fit_monotone_ev_threshold(raw_ev, realized_roi)

		self.assertAlmostEqual(fitted.threshold, 0.03, places=6)
		np.testing.assert_allclose(
			fitted.model.predict(np.array([0.015, 0.035], dtype=float)),
			np.array([-0.03, 0.035], dtype=float),
			atol=1e-6,
		)

	def test_apply_ev_threshold_masks_bets_below_cutoff(self):
		selection = {
			"best_index": np.array([0, 1, 2]),
			"selected_probs": np.array([0.55, 0.34, 0.28]),
			"selected_implied": np.array([0.50, 0.32, 0.25]),
			"selected_odds": np.array([2.0, 3.1, 4.0]),
			"best_ev": np.array([0.10, 0.054, 0.12]),
			"positive_mask": np.array([True, True, True]),
			"edge": np.array([0.05, 0.02, 0.03]),
			"full_kelly": np.array([0.10, 0.026, 0.04]),
		}

		thresholded = apply_ev_threshold(selection, 0.10)

		self.assertEqual(thresholded["positive_mask"].tolist(), [True, False, True])
		np.testing.assert_allclose(thresholded["full_kelly"], np.array([0.10, 0.0, 0.04]))
		self.assertTrue(np.array_equal(selection["positive_mask"], np.array([True, True, True])))

	def test_selection_to_bet_records_uses_actual_decimal_odds(self):
		selection = {
			"best_index": np.array([0, 1]),
			"selected_probs": np.array([0.55, 0.30]),
			"selected_implied": np.array([0.50, 0.28]),
			"selected_odds": np.array([2.0, 4.0]),
			"best_ev": np.array([0.10, 0.20]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.05, 0.02]),
			"full_kelly": np.array([0.10, 0.0666666667]),
		}
		y_true = np.array([0, 2])

		records = selection_to_bet_records(selection, y_true)

		np.testing.assert_allclose(records["raw_ev"], np.array([0.10, 0.20]))
		np.testing.assert_allclose(records["realized_roi"], np.array([1.0, -1.0]))

	def test_custom_selection_bankroll_evaluator_matches_existing_helper(self):
		probs = np.array([
			[0.75, 0.15, 0.10],
			[0.75, 0.15, 0.10],
		])
		y_true = np.array([0, 1])
		odds_home = np.array([2.0, 2.0])
		odds_draw = np.array([4.0, 4.0])
		odds_away = np.array([8.0, 8.0])
		groups = np.array(["2026-03-14", "2026-03-15"])
		selection = select_best_result_value(probs, np.stack([odds_home, odds_draw, odds_away], axis=1))

		expected = evaluate_bankroll_strategy(
			probs=probs,
			y_true=y_true,
			odds_home=odds_home,
			odds_draw=odds_draw,
			odds_away=odds_away,
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)
		actual = evaluate_selection_bankroll_strategy(
			selection=selection,
			y_true=y_true,
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)

		self.assertAlmostEqual(actual["bankroll_roi"], expected["bankroll_roi"], places=6)
		self.assertAlmostEqual(actual["max_drawdown"], expected["max_drawdown"], places=6)
		self.assertEqual(actual["bankroll_bet_count"], expected["bankroll_bet_count"])
		self.assertAlmostEqual(actual["final_bankroll"], 75.0, places=6)

	def test_selection_bankroll_path_tracks_group_wealth(self):
		probs = np.array([
			[0.75, 0.15, 0.10],
			[0.75, 0.15, 0.10],
		])
		y_true = np.array([0, 1])
		odds_home = np.array([2.0, 2.0])
		odds_draw = np.array([4.0, 4.0])
		odds_away = np.array([8.0, 8.0])
		groups = np.array(["2026-03-14", "2026-03-15"])
		selection = select_best_result_value(probs, np.stack([odds_home, odds_draw, odds_away], axis=1))

		path = selection_bankroll_path(
			selection=selection,
			y_true=y_true,
			groups=groups,
			kelly_fraction=1.0,
			initial_bankroll=100.0,
		)

		self.assertEqual(path["groups"], ["2026-03-14", "2026-03-15"])
		np.testing.assert_allclose(path["bankroll_after_group"], np.array([150.0, 75.0]))
		self.assertAlmostEqual(path["final_bankroll"], 75.0, places=6)
		self.assertEqual(len(path["group_rows"]), 2)

	def test_build_group_labels_split_week_groups_fri_mon_and_tue_thu(self):
		groups = np.array([
			"2026-04-03",  # Friday
			"2026-04-05",  # Sunday
			"2026-04-06",  # Monday
			"2026-04-07",  # Tuesday
			"2026-04-09",  # Thursday
			"2026-04-10",  # Friday
		], dtype=object)

		labels = build_group_labels(groups, mode="split_week")

		self.assertEqual(
			labels.tolist(),
			[
				"2026-04-03_fri-mon",
				"2026-04-03_fri-mon",
				"2026-04-03_fri-mon",
				"2026-04-07_tue-thu",
				"2026-04-07_tue-thu",
				"2026-04-10_fri-mon",
			],
		)


if __name__ == "__main__":
	unittest.main()
