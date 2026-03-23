import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from prod_run.generate_html_report import generate_html_report
from prod_run.pipeline import _round_budget_amounts, allocate_recommended_stakes, build_prediction_outputs, score_result_predictions
from utils.email_utils import build_email_html
from utils.portfolio import (
	DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP,
	DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS,
	DEFAULT_JOINT_OPTIMIZER_MIN_STEP,
	_joint_expected_log_growth_and_grad,
	get_joint_quadrature_rule,
)

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


def _round_budget_amounts_frontend(amounts: list[float], total_budget: float) -> list[float]:
	rounded = [round(float(amount), 2) for amount in amounts]
	positive_idx = [index for index, amount in enumerate(amounts) if float(amount) > 0.0]
	if not positive_idx:
		return rounded
	delta_cents = int(round((round(float(total_budget), 2) - round(float(sum(rounded)), 2)) * 100))
	if delta_cents == 0:
		return rounded
	residuals = [float(amount) - float(rounded[index]) for index, amount in enumerate(amounts)]
	if delta_cents > 0:
		order = sorted(
			positive_idx,
			key=lambda index: (residuals[index], float(amounts[index]), index),
			reverse=True,
		)
	else:
		order = sorted(
			[index for index in positive_idx if rounded[index] > 0.0],
			key=lambda index: (float(rounded[index]) - float(amounts[index]), rounded[index], index),
			reverse=True,
		)
	if not order:
		return rounded
	while delta_cents != 0:
		progress = False
		for index in order:
			if delta_cents == 0:
				break
			if delta_cents > 0:
				rounded[index] = round(float(rounded[index]) + 0.01, 2)
				delta_cents -= 1
				progress = True
				continue
			if rounded[index] >= 0.01 - 1e-12:
				rounded[index] = round(float(rounded[index]) - 0.01, 2)
				delta_cents += 1
				progress = True
		if not progress:
			break
	return rounded


def _project_nonnegative_l1_ball_frontend(values: list[float], radius: float) -> list[float]:
	clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
	if float(clipped.sum()) <= float(radius):
		return clipped.tolist()
	sorted_values = np.sort(clipped)[::-1]
	cumulative = np.cumsum(sorted_values)
	indices = np.arange(1, len(sorted_values) + 1, dtype=float)
	threshold_candidates = sorted_values - (cumulative - float(radius)) / indices
	rho = int(np.flatnonzero(threshold_candidates > 0.0)[-1])
	theta = (cumulative[rho] - float(radius)) / float(rho + 1)
	return np.clip(clipped - theta, 0.0, None).tolist()


def _simulate_frontend_stake_plan(
	selection: dict[str, np.ndarray],
	total_budget: float,
	kelly_fraction: float,
	min_bet_amount: float,
) -> tuple[list[bool], list[float], list[float]]:
	active = [bool(flag) for flag in selection["positive_mask"]]
	shares = [0.0 for _ in active]
	amounts = [0.0 for _ in active]
	quadrature_nodes, quadrature_weights = get_joint_quadrature_rule()
	while True:
		active_indices = [index for index, is_active in enumerate(active) if is_active]
		if not active_indices or not (float(total_budget) > 0.0):
			return active, shares, amounts

		selected_probs = np.asarray(selection["selected_probs"][active_indices], dtype=float)
		selected_odds = np.asarray(selection["selected_odds"][active_indices], dtype=float)
		full_kelly = np.asarray(selection["full_kelly"][active_indices], dtype=float)
		if len(active_indices) == 1:
			weights = full_kelly.copy()
		else:
			weights = np.asarray(
				_project_nonnegative_l1_ball_frontend(full_kelly.tolist(), 1.0 - 1e-12),
				dtype=float,
			)
			step = float(DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP)
			current_value, current_grad = _joint_expected_log_growth_and_grad(
				weights=weights,
				selected_probs=selected_probs,
				selected_odds=selected_odds,
				quadrature_nodes=quadrature_nodes,
				quadrature_weights=quadrature_weights,
			)
			for _ in range(int(DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS)):
				candidate = np.asarray(
					_project_nonnegative_l1_ball_frontend((weights + step * current_grad).tolist(), 1.0 - 1e-12),
					dtype=float,
				)
				next_value, next_grad = _joint_expected_log_growth_and_grad(
					weights=candidate,
					selected_probs=selected_probs,
					selected_odds=selected_odds,
					quadrature_nodes=quadrature_nodes,
					quadrature_weights=quadrature_weights,
				)
				if np.isfinite(next_value) and next_value >= current_value:
					weights = candidate
					current_value = next_value
					current_grad = next_grad
					step *= 1.05
				else:
					step *= 0.5
				if step < float(DEFAULT_JOINT_OPTIMIZER_MIN_STEP):
					break

		scaled = np.asarray(
			_project_nonnegative_l1_ball_frontend((weights * max(0.0, float(kelly_fraction))).tolist(), 1.0 - 1e-12),
			dtype=float,
		)
		raw_amounts = [float(weight) * float(total_budget) for weight in scaled]
		rounded_active = _round_budget_amounts_frontend(raw_amounts, total_budget=float(sum(raw_amounts)))
		amounts = [0.0 for _ in active]
		shares = [0.0 for _ in active]
		for local_index, row_index in enumerate(active_indices):
			amounts[row_index] = rounded_active[local_index]
			shares[row_index] = rounded_active[local_index] / float(total_budget)
		too_small = [
			is_active and amount > 0.0 and amount + 1e-12 < float(min_bet_amount)
			for amount, is_active in zip(amounts, active)
		]
		if not any(too_small):
			return active, shares, amounts
		active = [is_active and not is_too_small for is_active, is_too_small in zip(active, too_small)]


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
		self.assertIn('id="kelly-fraction"', report_html)
		self.assertIn('min="0.1"', report_html)
		self.assertIn('max="1"', report_html)
		self.assertIn('value="100.00"', report_html)
		self.assertIn("Current bankroll", report_html)
		self.assertIn("Kelly fraction", report_html)
		self.assertIn('id="summary-total-amount"', report_html)
		self.assertIn('type="number"', report_html)
		self.assertIn("single best side", report_html)
		self.assertIn("function roundHalfEven", report_html)
		self.assertIn("function roundBudgetAmounts", report_html)
		self.assertIn("function projectNonnegativeL1Ball", report_html)
		self.assertIn("function jointObjectiveAndGrad", report_html)
		self.assertIn("function optimizeJointStakePlan", report_html)
		self.assertIn("const QUADRATURE_NODES =", report_html)
		self.assertIn("const QUADRATURE_WEIGHTS =", report_html)
		self.assertNotIn("function computeStakePlan", report_html)
		self.assertNotIn("Minimum stake per bet", report_html)
		self.assertIn("change your bankroll and Kelly fraction", build_email_html(output_df, None, "2026-03-10"))

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
			"selected_probs": np.array([0.55, 0.250705]),
			"selected_implied": np.array([0.40, 0.25]),
			"selected_odds": np.array([2.5, 4.0]),
			"best_ev": np.array([0.375, 0.00282]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.15, 0.000705]),
			"full_kelly": np.array([0.25, 0.00094]),
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
		self.assertAlmostEqual(scored["Result_Budget_Amount"].iloc[0], 25.0, places=2)
		self.assertAlmostEqual(scored["Result_Budget_Amount"].iloc[1], 0.0, places=2)

		output_df, value_df = build_prediction_outputs(merged, scored)
		self.assertEqual(len(value_df), 2)
		email_html = build_email_html(output_df, value_df, "2026-03-10")
		self.assertIn("Barcelona vs Sevilla", email_html)
		self.assertIn("Manchester United vs Aston Villa", email_html)

	def test_minimum_bet_amount_prunes_and_recomputes(self):
		selection = {
			"best_index": np.array([0, 1]),
			"selected_probs": np.array([0.55, 0.250705]),
			"selected_implied": np.array([0.40, 0.25]),
			"selected_odds": np.array([2.5, 4.0]),
			"best_ev": np.array([0.375, 0.00282]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.15, 0.000705]),
			"full_kelly": np.array([0.25, 0.00094]),
		}

		allocation = allocate_recommended_stakes(
			selection=selection,
			total_budget=100.0,
			kelly_fraction=1.0,
			min_bet_amount=0.1,
		)

		self.assertAlmostEqual(allocation["stake_amounts"][0], 25.0, places=2)
		self.assertAlmostEqual(allocation["stake_amounts"][1], 0.0, places=2)
		self.assertEqual(allocation["recommended_mask"].tolist(), [True, False])

	def test_frontend_stake_plan_matches_backend_threshold_rounding(self):
		selection = {
			"best_index": np.array([0, 1]),
			"selected_probs": np.array([0.55, 0.250705]),
			"selected_implied": np.array([0.40, 0.25]),
			"selected_odds": np.array([2.5, 4.0]),
			"best_ev": np.array([0.375, 0.00282]),
			"positive_mask": np.array([True, True]),
			"edge": np.array([0.15, 0.000705]),
			"full_kelly": np.array([0.25, 0.00094]),
		}

		backend = allocate_recommended_stakes(
			selection=selection,
			total_budget=100.0,
			kelly_fraction=1.0,
			min_bet_amount=0.1,
		)
		frontend_active, frontend_shares, frontend_amounts = _simulate_frontend_stake_plan(
			selection=selection,
			total_budget=100.0,
			kelly_fraction=1.0,
			min_bet_amount=0.1,
		)

		self.assertEqual(frontend_active, backend["recommended_mask"].tolist())
		np.testing.assert_allclose(frontend_shares, backend["stake_shares"], atol=1e-9)
		np.testing.assert_allclose(frontend_amounts, backend["stake_amounts"], atol=1e-9)

	def test_frontend_rounding_matches_backend_for_tied_allocations(self):
		positive_delta_amounts = [1.005, 1.005]
		negative_delta_amounts = [0.335, 0.335]

		positive_delta_backend = _round_budget_amounts(
			np.array(positive_delta_amounts, dtype=float),
			total_budget=float(sum(positive_delta_amounts)),
		)
		negative_delta_backend = _round_budget_amounts(
			np.array(negative_delta_amounts, dtype=float),
			total_budget=float(sum(negative_delta_amounts)),
		)

		np.testing.assert_allclose(
			_round_budget_amounts_frontend(positive_delta_amounts, total_budget=float(sum(positive_delta_amounts))),
			positive_delta_backend,
			atol=1e-9,
		)
		np.testing.assert_allclose(
			_round_budget_amounts_frontend(negative_delta_amounts, total_budget=float(sum(negative_delta_amounts))),
			negative_delta_backend,
			atol=1e-9,
		)

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

	def test_scoring_drops_rows_with_missing_model_features_and_reports_it(self):
		merged = _build_merged_frame()
		merged.loc[1, FEATURE_COLS[0]] = np.nan
		bundle = SimpleNamespace(
			model=FakeModel(),
			scaler=IdentityScaler(),
			feature_cols=FEATURE_COLS,
			cat_config=None,
		)

		stdout = io.StringIO()
		with redirect_stdout(stdout):
			scored = score_result_predictions(merged, bundle, fixed_budget=100.0)

		self.assertEqual(len(scored), 2)
		self.assertNotIn(1, scored["_row_id"].tolist())
		self.assertIn("Dropped 1 matched games due to missing model features", stdout.getvalue())
		self.assertIn(FEATURE_COLS[0], stdout.getvalue())


if __name__ == "__main__":
	unittest.main()
