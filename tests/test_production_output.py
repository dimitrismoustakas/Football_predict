import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from prod_run.generate_html_report import generate_html_report
from prod_run.pipeline import build_prediction_outputs, score_result_predictions
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
	def test_scored_predictions_include_budget_fields(self):
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
			budget_strategy="kelly",
			kelly_fraction=0.5,
		)

		self.assertIn("Result_Budget_Amount", scored.columns)
		self.assertIn("Result_Budget_Share", scored.columns)
		self.assertIn("Result_Value_Implied", scored.columns)
		self.assertIn("Result_Edge", scored.columns)
		self.assertAlmostEqual(scored["Result_Budget_Amount"].sum(), 100.0, places=2)

		output_df, value_df = build_prediction_outputs(merged, scored)
		self.assertFalse(value_df.empty)
		html = build_email_html(
			predictions_df=output_df,
			bets_df=value_df,
			report_date="2026-03-10",
			fixed_budget=100.0,
			budget_strategy="kelly",
			kelly_fraction=0.5,
		)
		self.assertIn("</html>", html)
		self.assertIn("Suggested Bets", html)
		self.assertIn("Split %", html)
		self.assertIn("Odds", html)
		self.assertIn("Model %", html)
		self.assertIn("Market %", html)
		self.assertIn("Edge", html)
		self.assertIn("EV %", html)
		self.assertNotIn("Budget split strategy", html)
		self.assertNotIn("All Predictions", html)

	def test_html_report_renders_interactive_table(self):
		merged = _build_merged_frame()
		bundle = SimpleNamespace(
			model=FakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)
		scored = score_result_predictions(merged, bundle, fixed_budget=100.0)
		output_df, _ = build_prediction_outputs(merged, scored)

		with tempfile.TemporaryDirectory() as tmp_dir:
			output_path = Path(tmp_dir) / "report.html"
			generate_html_report(
				output_df,
				output_path,
				fixed_budget=100.0,
				budget_strategy="kelly",
				kelly_fraction=0.5,
			)
			report_html = output_path.read_text(encoding="utf-8")

		self.assertIn("Model Home %", report_html)
		self.assertIn("Best Bet Now", report_html)
		self.assertIn("Split % Now", report_html)
		self.assertIn("Amount Now", report_html)
		self.assertIn('id="total-budget"', report_html)
		self.assertIn('value="10.00"', report_html)
		self.assertIn('type="number"', report_html)
		self.assertNotIn("Value Picks", report_html)
		self.assertNotIn("All Predictions", report_html)

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
