import unittest
from unittest.mock import patch

import polars as pl

from training.train_main_model import collect_pre_test_seasons, load_evaluation_config, split_selection_folds
from training.train_utils import generate_rolling_cv_folds


class EvaluationConfigTests(unittest.TestCase):
	def test_final_training_seed_accepts_explicit_value_and_fallback(self):
		base_config = {
			"comparison_metric": "log_loss",
			"training_seed": "42",
			"rolling_cv_n_folds": "4",
			"test_season": 2526,
			"test_role": "acceptance",
		}
		cases = (
			({"final_training_seed": "13"}, 13),
			({}, 10_042),
		)

		for overrides, expected_seed in cases:
			with self.subTest(overrides=overrides):
				with patch(
					"training.train_main_model.load_json",
					return_value={**base_config, **overrides},
				):
					config = load_evaluation_config()

				self.assertEqual(config["training_seed"], 42)
				self.assertEqual(config["final_training_seed"], expected_seed)


class TrainingSplitTests(unittest.TestCase):
	def test_latest_season_is_test_only_and_previous_season_selects_epochs(self):
		seasons = ["2020", "2021", "2022", "2023", "2024", "2025"]
		folds = generate_rolling_cv_folds(
			pl.DataFrame({"season": seasons}),
			n_folds=4,
			test_season="2025",
		)

		objective_folds, epoch_fold = split_selection_folds(folds)
		all_pre_test_seasons = collect_pre_test_seasons(folds)

		self.assertEqual(len(objective_folds), 3)
		self.assertEqual(epoch_fold, (["2020", "2021", "2022", "2023"], "2024"))
		self.assertEqual(all_pre_test_seasons, seasons[:-1])
		self.assertNotIn("2025", all_pre_test_seasons)


if __name__ == "__main__":
	unittest.main()
