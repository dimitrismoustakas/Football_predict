import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from prod_run.generate_html_report import generate_html_report
from prod_run.pipeline import allocate_recommended_stakes, build_prediction_outputs, score_result_predictions
from utils.email_utils import build_email_html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLS = json.loads((PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json").read_text(encoding="utf-8"))


class IdentityScaler:
	def transform(self, X):
		return X


class FakeModel:
	def __call__(self, X, cat_features, implied, raw_margin):
		bias = torch.tensor([0.22, -0.08, -0.14], dtype=torch.float32, device=implied.device)
		return torch.log(implied.clamp_min(1e-6)) + bias


class LeagueBiasFakeModel(FakeModel):
	learn_league_market_bias = True
	learn_league_residual_bias = True

	def __call__(self, X, cat_features, implied, raw_margin):
		if cat_features is None:
			raise ValueError("cat_features required")
		return super().__call__(X, cat_features, implied, raw_margin)


def _build_merged_frame() -> pd.DataFrame:
	rows = []
	fixtures = [
		("ENG-Premier League", "2026-03-14T13:00:00Z", "Manchester United", "Aston Villa", 2.10, 3.60, 3.90),
		("ESP-La Liga", "2026-03-14T17:30:00Z", "Barcelona", "Sevilla", 1.95, 3.80, 4.20),
		("ITA-Serie A", "2026-03-15T19:45:00Z", "Inter", "Roma", 2.45, 3.20, 2.95),
	]
	for row_id, (league, commence_time, home, away, odds_home, odds_draw, odds_away) in enumerate(fixtures):
		row = {
			"_row_id": row_id,
			"league": league,
			"commence_time": pd.Timestamp(commence_time),
			"home_team": home,
			"away_team": away,
			"odds_home": odds_home,
			"odds_draw": odds_draw,
			"odds_away": odds_away,
			"league_idx": 0,
			"home_promoted": 0,
			"away_promoted": 0,
		}
		for feature in FEATURE_COLS:
			row[feature] = 0.0
		rows.append(row)
	return pd.DataFrame(rows)


class ProductionOutputTests(unittest.TestCase):
	def test_scored_predictions_include_positive_ev_and_budget_fields(self):
		merged = _build_merged_frame()
		bundle = SimpleNamespace(
			model=FakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)

		scored = score_result_predictions(
			merged,
			bundle,
			fixed_budget=100.0,
			kelly_fraction=0.5,
		)

		self.assertIn("Result_Budget_Amount", scored.columns)
		self.assertIn("Result_Budget_Share", scored.columns)
		self.assertIn("Result_Value_Implied", scored.columns)
		self.assertIn("Result_Edge", scored.columns)
		self.assertGreater(scored["Result_Budget_Amount"].sum(), 0.0)
		self.assertLessEqual(scored["Result_Budget_Amount"].sum(), 100.0)

		output_df, value_df = build_prediction_outputs(merged, scored)
		self.assertFalse(value_df.empty)
		html = build_email_html(
			predictions_df=output_df,
			bets_df=value_df,
			report_date="2026-03-10",
		)
		self.assertIn("</html>", html)
		self.assertIn("Positive EV Games", html)
		self.assertIn("games with positive expected value", html)
		self.assertIn("open it in a browser", html)
		self.assertIn("Odds", html)
		self.assertIn("Model %", html)
		self.assertIn("Market %", html)
		self.assertIn("Edge", html)
		self.assertIn("EV %", html)
		self.assertNotIn("Stake % of Bankroll", html)
		self.assertNotIn("Stake Amount", html)
		self.assertNotIn("Proposed total stake", html)
		self.assertNotIn("Minimum stake rule", html)

	def test_html_report_renders_interactive_table(self):
		merged = _build_merged_frame()
		bundle = SimpleNamespace(
			model=FakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)
		scored = score_result_predictions(merged, bundle, fixed_budget=100.0, kelly_fraction=0.5)
		output_df, _ = build_prediction_outputs(merged, scored)

		with tempfile.TemporaryDirectory() as tmp_dir:
			output_path = Path(tmp_dir) / "report.html"
			generate_html_report(
				output_df,
				output_path,
				fixed_budget=100.0,
				kelly_fraction=0.5,
				min_bet_amount=0.1,
			)
			report_html = output_path.read_text(encoding="utf-8")

		self.assertIn("Model Home %", report_html)
		self.assertIn("Best Bet Now", report_html)
		self.assertIn("Stake % Now", report_html)
		self.assertIn("Amount Now", report_html)
		self.assertIn('id="total-budget"', report_html)
		self.assertIn('value="100.00"', report_html)
		self.assertIn("Current bankroll", report_html)
		self.assertIn('id="summary-total-amount"', report_html)
		self.assertIn('type="number"', report_html)
		self.assertIn("Enter your current bankroll below to see the bankroll allocation", report_html)
		self.assertNotIn("Minimum stake per bet", report_html)
		self.assertNotIn("Kelly", report_html)

	def test_scoring_keeps_positive_ev_rows_even_if_minimum_bet_prunes_stake(self):
		merged = _build_merged_frame().iloc[:2].copy()
		bundle = SimpleNamespace(
			model=FakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)
		crafted_selection = {
			"best_index": np.array([0, 1]),
			"selected_probs": np.array([0.55, 0.26]),
			"selected_implied": np.array([0.40, 0.25]),
			"selected_odds": np.array([2.5, 4.0]),
			"best_ev": np.array([0.375, 0.04]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.15, 0.01]),
			"full_kelly": np.array([1.0, 0.0001]),
		}

		with patch("prod_run.pipeline.select_best_result_value", return_value=crafted_selection):
			scored = score_result_predictions(
				merged,
				bundle,
				fixed_budget=100.0,
				kelly_fraction=1.0,
				min_bet_amount=0.1,
			)

		self.assertEqual(scored["Result_EV"].notna().tolist(), [True, True])
		self.assertAlmostEqual(scored["Result_Budget_Amount"].iloc[0], 100.0, places=2)
		self.assertAlmostEqual(scored["Result_Budget_Amount"].iloc[1], 0.0, places=2)

		output_df, value_df = build_prediction_outputs(merged, scored)
		self.assertEqual(len(value_df), 2)
		email_html = build_email_html(output_df, value_df, "2026-03-10")
		self.assertIn("Barcelona vs Sevilla", email_html)
		self.assertIn("Manchester United vs Aston Villa", email_html)

	def test_minimum_bet_amount_prunes_and_recomputes(self):
		selection = {
			"best_index": np.array([0, 1]),
			"selected_probs": np.array([0.55, 0.26]),
			"selected_implied": np.array([0.40, 0.25]),
			"selected_odds": np.array([2.5, 4.0]),
			"best_ev": np.array([0.375, 0.04]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.15, 0.01]),
			"full_kelly": np.array([1.0, 0.0001]),
		}

		allocation = allocate_recommended_stakes(
			selection=selection,
			total_budget=100.0,
			kelly_fraction=1.0,
			min_bet_amount=0.1,
		)

		self.assertAlmostEqual(allocation["stake_amounts"][0], 100.0, places=2)
		self.assertAlmostEqual(allocation["stake_amounts"][1], 0.0, places=2)
		self.assertEqual(allocation["recommended_mask"].tolist(), [True, False])

	def test_scoring_passes_cat_features_for_league_bias_models(self):
		merged = _build_merged_frame()
		bundle = SimpleNamespace(
			model=LeagueBiasFakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)

		scored = score_result_predictions(merged, bundle, fixed_budget=100.0)

		self.assertFalse(scored.empty)
		self.assertIn("Result_Value_Side", scored.columns)


if __name__ == "__main__":
	unittest.main()
