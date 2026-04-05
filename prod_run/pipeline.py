"""
Production pipeline for the canonical match-result model.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from dotenv import load_dotenv

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prod_run import build_prod_features, fetch_odds
from prod_run.generate_html_report import generate_html_report
from training.inference import predict_probabilities
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from utils import send_email
from utils.portfolio import DEFAULT_KELLY_FRACTION, allocate_bankroll_kelly, select_best_result_value

load_dotenv()

DATA_DIR = Path("data")
PROD_DIR = DATA_DIR / "prod"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PROD_FEATURES_PATH = PROD_DIR / "features_season.parquet"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"
OUTPUT_HTML_PATH = PREDICTIONS_DIR / "upcoming_predictions.html"
STATIC_REPORT_PATH = Path("site") / "index.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_LABELS = np.array(["Home", "Draw", "Away"])
DEFAULT_MIN_BET_AMOUNT = 0.1
DEFAULT_PREDICTION_WINDOW_DAYS = 7


def _env_flag(name: str, default: bool) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
	value = os.environ.get(name)
	if value is None:
		return default
	return float(value)


def _env_path(name: str, default: Path) -> Path:
	value = os.environ.get(name)
	if value is None or not value.strip():
		return default
	return Path(value)


def load_model():
	bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=DEVICE)
	print(f"Loading model bundle: {bundle.name}")
	return bundle


def resolve_merged_col(frame: pd.DataFrame, base_name: str) -> str:
	if f"{base_name}_odds" in frame.columns:
		return f"{base_name}_odds"
	return base_name


def _round_budget_amounts(stake_amounts: np.ndarray, total_budget: float) -> np.ndarray:
	"""Round currency amounts while preserving the requested total budget."""

	rounded = np.round(stake_amounts, 2)
	positive_idx = np.flatnonzero(stake_amounts > 0.0)
	if positive_idx.size == 0:
		return rounded
	delta_cents = int(round((round(float(total_budget), 2) - round(float(rounded.sum()), 2)) * 100))
	if delta_cents == 0:
		return rounded

	residuals = stake_amounts - rounded
	if delta_cents > 0:
		order = positive_idx[np.lexsort((-positive_idx, -stake_amounts[positive_idx], -residuals[positive_idx]))]
	else:
		nonzero_idx = positive_idx[rounded[positive_idx] > 0.0]
		if nonzero_idx.size == 0:
			return rounded
		overround = rounded[nonzero_idx] - stake_amounts[nonzero_idx]
		order = nonzero_idx[np.lexsort((-nonzero_idx, -rounded[nonzero_idx], -overround))]

	while delta_cents != 0:
		changed = False
		for index in order:
			if delta_cents == 0:
				break
			if delta_cents > 0:
				rounded[index] = np.round(rounded[index] + 0.01, 2)
				delta_cents -= 1
				changed = True
				continue
			if rounded[index] >= 0.01 - 1e-12:
				rounded[index] = np.round(rounded[index] - 0.01, 2)
				delta_cents += 1
				changed = True
		if not changed:
			break
	return rounded


def allocate_recommended_stakes(
	selection: dict[str, np.ndarray],
	total_budget: float,
	kelly_fraction: float,
	min_bet_amount: float = DEFAULT_MIN_BET_AMOUNT,
) -> dict[str, np.ndarray | float | str]:
	"""Allocate bankroll stakes, pruning sub-minimum bets until stable."""

	active_mask = selection["positive_mask"].astype(bool).copy()
	final_allocation = {
		"strategy": "joint_bankroll_kelly",
		"kelly_fraction": float(kelly_fraction),
		"raw_weights": np.zeros_like(selection["best_ev"], dtype=float),
		"stake_shares": np.zeros_like(selection["best_ev"], dtype=float),
		"stake_amounts": np.zeros_like(selection["best_ev"], dtype=float),
		"allocated_budget": 0.0,
	}

	while True:
		constrained_selection = {
			key: value.copy() if isinstance(value, np.ndarray) else value
			for key, value in selection.items()
		}
		constrained_selection["positive_mask"] = active_mask.copy()
		allocation = allocate_bankroll_kelly(
			selection=constrained_selection,
			total_bankroll=total_budget,
			kelly_fraction=kelly_fraction,
		)
		rounded_amounts = _round_budget_amounts(
			allocation["stake_amounts"],
			total_budget=float(allocation["allocated_budget"]),
		)
		too_small_mask = (rounded_amounts > 0.0) & (rounded_amounts + 1e-12 < float(min_bet_amount))
		final_allocation = allocation | {"stake_amounts": rounded_amounts}
		if not np.any(too_small_mask):
			break
		next_active_mask = active_mask & ~too_small_mask
		if np.array_equal(next_active_mask, active_mask):
			break
		active_mask = next_active_mask

	display_shares = np.zeros_like(final_allocation["stake_amounts"], dtype=float)
	if float(total_budget) > 0.0:
		display_shares = final_allocation["stake_amounts"] / float(total_budget)

	return final_allocation | {
		"stake_shares": display_shares,
		"allocated_budget": float(final_allocation["stake_amounts"].sum()),
		"recommended_mask": final_allocation["stake_amounts"] > 0.0,
		"min_bet_amount": float(min_bet_amount),
	}


def score_result_predictions(
	merged: pd.DataFrame,
	model_bundle,
	fixed_budget: float = 100.0,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	min_bet_amount: float = DEFAULT_MIN_BET_AMOUNT,
) -> pd.DataFrame:
	model = model_bundle.model
	scaler = model_bundle.scaler
	feature_cols = model_bundle.feature_cols
	odds_home_col = resolve_merged_col(merged, "odds_home")
	odds_draw_col = resolve_merged_col(merged, "odds_draw")
	odds_away_col = resolve_merged_col(merged, "odds_away")
	working = merged.copy()
	missing_feature_cols = [col for col in feature_cols if col not in working.columns]
	if missing_feature_cols:
		raise RuntimeError(
			"Production features are missing model-required columns: "
			f"{missing_feature_cols[:8]}{'...' if len(missing_feature_cols) > 8 else ''}"
		)
	required_cols = [odds_home_col, odds_draw_col, odds_away_col, "league_idx", "home_promoted", "away_promoted"]

	ready = working.dropna(subset=required_cols).copy()
	dropped_required_count = len(working) - len(ready)
	if dropped_required_count:
		print(
			f"Dropped {dropped_required_count} matched games before scoring due to missing odds or league-side metadata"
		)
	feature_null_mask = ready[feature_cols].isna().any(axis=1)
	if feature_null_mask.any():
		dropped_feature_rows = ready.loc[feature_null_mask, ["league", "home_team", "away_team"] + feature_cols].copy()
		top_missing_features = (
			dropped_feature_rows[feature_cols]
			.isna()
			.sum()
			.sort_values(ascending=False)
		)
		top_missing_features = [
			(col, int(count))
			for col, count in top_missing_features.items()
			if int(count) > 0
		][:8]
		fixture_preview = ", ".join(
			f"{row.home_team} vs {row.away_team} ({row.league})"
			for row in dropped_feature_rows[["league", "home_team", "away_team"]].head(5).itertuples(index=False)
		)
		print(
			f"Dropped {int(feature_null_mask.sum())} matched games due to missing model features. "
			f"Top missing columns: {top_missing_features}. "
			f"Examples: {fixture_preview}"
		)
		ready = ready.loc[~feature_null_mask].copy()
	if ready.empty:
		return pd.DataFrame()

	inv_odds = np.stack([
		1 / ready[odds_home_col].to_numpy(dtype=float),
		1 / ready[odds_draw_col].to_numpy(dtype=float),
		1 / ready[odds_away_col].to_numpy(dtype=float),
	], axis=1)
	norm = inv_odds.sum(axis=1, keepdims=True)
	implied_probs = inv_odds / norm
	raw_margin = norm.reshape(-1) - 1
	cat_features = ready[["league_idx", "home_promoted", "away_promoted"]].to_numpy(dtype=np.int64)

	probs = predict_probabilities(
		model=model,
		scaler=scaler,
		X_raw=ready[feature_cols].to_numpy(dtype=float),
		device=DEVICE,
		cat_features=cat_features,
		implied_probs=implied_probs,
		raw_margin=raw_margin,
	)
	result_pick_idx = np.argmax(probs, axis=1)
	odds_matrix = np.stack([
		ready[odds_home_col].to_numpy(dtype=float),
		ready[odds_draw_col].to_numpy(dtype=float),
		ready[odds_away_col].to_numpy(dtype=float),
	], axis=1)
	selection = select_best_result_value(probs, odds_matrix, implied_probs=implied_probs)
	allocation = allocate_recommended_stakes(
		selection=selection,
		total_budget=fixed_budget,
		kelly_fraction=kelly_fraction,
		min_bet_amount=min_bet_amount,
	)
	positive_mask = selection["positive_mask"]
	recommended_mask = allocation["recommended_mask"]

	return pd.DataFrame({
		"_row_id": ready["_row_id"],
		"Prob_Home": probs[:, 0].round(3),
		"Prob_Draw": probs[:, 1].round(3),
		"Prob_Away": probs[:, 2].round(3),
		"Implied_Home": implied_probs[:, 0].round(3),
		"Implied_Draw": implied_probs[:, 1].round(3),
		"Implied_Away": implied_probs[:, 2].round(3),
		"Odds_Home": ready[odds_home_col].round(3),
		"Odds_Draw": ready[odds_draw_col].round(3),
		"Odds_Away": ready[odds_away_col].round(3),
		"Result_Model_Pick": RESULT_LABELS[result_pick_idx],
		"Result_Value_Side": np.where(positive_mask, RESULT_LABELS[selection["best_index"]], ""),
		"Result_Value_Prob": np.where(positive_mask, np.round(selection["selected_probs"], 3), np.nan),
		"Result_Value_Implied": np.where(positive_mask, np.round(selection["selected_implied"], 3), np.nan),
		"Result_Edge": np.where(positive_mask, np.round(selection["edge"], 4), np.nan),
		"Result_EV": np.where(positive_mask, np.round(selection["best_ev"], 4), np.nan),
		"Result_Kelly_Fraction": np.where(
			positive_mask,
			np.round(allocation["raw_weights"], 4),
			np.nan,
		),
		"Result_Budget_Share": np.where(recommended_mask, np.round(allocation["stake_shares"], 4), 0.0),
		"Result_Budget_Amount": np.where(recommended_mask, allocation["stake_amounts"], 0.0),
	})


def build_prediction_outputs(merged: pd.DataFrame, result_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Assemble final production tables from merged fixtures and scored outputs."""

	base_output = pd.DataFrame({
		"_row_id": merged["_row_id"],
		"League": merged["league"],
		"Date": merged["commence_time"].dt.tz_convert("Europe/Athens").dt.strftime("%Y-%m-%d"),
		"Time": merged["commence_time"].dt.tz_convert("Europe/Athens").dt.strftime("%H:%M"),
		"Home": merged["home_team"],
		"Away": merged["away_team"],
	})
	output_df = base_output.merge(result_predictions, on="_row_id", how="left")
	output_df = output_df.drop(columns=["_row_id"])
	output_df = output_df.sort_values(["League", "Date", "Time", "Home", "Away"]).reset_index(drop=True)
	value_output = output_df[output_df["Result_EV"].notna()].copy()
	return output_df, value_output


def main():
	odds_api_key = os.environ.get("ODDS_API_KEY")
	send_email_enabled = _env_flag("SEND_EMAIL", True)
	publish_static_report = _env_flag("PUBLISH_STATIC_REPORT", True)
	prediction_window_days = int(os.environ.get("PREDICTION_WINDOW_DAYS", str(DEFAULT_PREDICTION_WINDOW_DAYS)))
	fixed_budget = _env_float("FIXED_BUDGET", 100.0)
	kelly_fraction = _env_float("KELLY_FRACTION", DEFAULT_KELLY_FRACTION)
	min_bet_amount = _env_float("MIN_BET_AMOUNT", DEFAULT_MIN_BET_AMOUNT)
	static_report_path = _env_path("STATIC_REPORT_PATH", STATIC_REPORT_PATH)
	report_public_url = os.environ.get("REPORT_PUBLIC_URL", "").strip() or None

	print("=" * 60)
	print("FOOTBALL PRODUCTION PIPELINE")
	print("=" * 60)

	print("\n--- Step 1: Fetching Odds ---")
	raw_odds = fetch_odds.get_all_leagues_odds(odds_api_key)
	parsed_odds = fetch_odds.parse_odds_data(raw_odds)
	print(f"Fetched {len(parsed_odds)} games with odds across all leagues")
	odds_df = pd.DataFrame(parsed_odds)
	odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True)

	now_utc = datetime.now(timezone.utc)
	today_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
	window_end_utc = today_utc + pd.Timedelta(days=prediction_window_days)
	odds_df = odds_df[(odds_df["commence_time"] >= today_utc) & (odds_df["commence_time"] < window_end_utc)].copy()
	print(f"Found {len(odds_df)} upcoming games")
	if odds_df.empty:
		raise RuntimeError("No upcoming games found in odds data")

	print("\n--- Step 2: Building Production Features ---")
	build_prod_features.main(upcoming_fixtures=odds_df)

	print("\n--- Step 3: Loading Model ---")
	model_bundle = load_model()
	print(f"Loaded result model with {len(model_bundle.feature_cols)} features")

	features_df = pl.read_parquet(PROD_FEATURES_PATH)
	supported_leagues = list(fetch_odds.LEAGUE_TO_SPORT_KEY.keys())
	features_df = features_df.filter(pl.col("league").is_in(supported_leagues))
	features_pd = features_df.to_pandas()
	features_pd["date"] = pd.to_datetime(features_pd["date"], utc=True, errors="coerce")
	features_pd = features_pd[(features_pd["date"] >= today_utc) & (features_pd["date"] < window_end_utc)].copy()
	print(f"Upcoming games in features window: {len(features_pd)}")

	odds_merge_df = odds_df.rename(columns={"league_id": "league"})
	merged = odds_merge_df.merge(features_pd, on=["league", "home_team", "away_team"], how="inner", suffixes=("_odds", "_feat"))
	print(f"Matched {len(merged)} games between odds and features")
	if merged.empty:
		raise RuntimeError("No games matched between odds and features")

	merged = merged.reset_index(drop=True)
	merged["_row_id"] = merged.index

	print("\n--- Step 4: Scoring Model ---")
	result_predictions = score_result_predictions(
		merged,
		model_bundle,
		fixed_budget=fixed_budget,
		kelly_fraction=kelly_fraction,
		min_bet_amount=min_bet_amount,
	)
	if result_predictions.empty:
		raise RuntimeError("No games had complete model features for scoring after dropping incomplete rows")
	output_df, value_output = build_prediction_outputs(merged, result_predictions)
	recommended_output = value_output[value_output["Result_Budget_Amount"] > 0.0].copy()

	print("\n--- Step 5: Saving Output ---")
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(OUTPUT_CSV_PATH, index=False)
	print(f"Saved predictions to {OUTPUT_CSV_PATH}")
	generate_html_report(
		output_df,
		OUTPUT_HTML_PATH,
		fixed_budget=fixed_budget,
		kelly_fraction=kelly_fraction,
		min_bet_amount=min_bet_amount,
	)
	print(f"Saved local interactive report to {OUTPUT_HTML_PATH}")
	if publish_static_report:
		generate_html_report(
			output_df,
			static_report_path,
			fixed_budget=fixed_budget,
			kelly_fraction=kelly_fraction,
			min_bet_amount=min_bet_amount,
		)
		print(f"Published static report to {static_report_path}")
	elif report_public_url:
		print("Warning: REPORT_PUBLIC_URL is set but PUBLISH_STATIC_REPORT is disabled")

	print("\n" + "=" * 60)
	print("PREDICTIONS SUMMARY")
	print("=" * 60)
	print(output_df.to_string(index=False))

	if not value_output.empty:
		print("\n" + "=" * 60)
		print("POSITIVE EV GAMES")
		print("=" * 60)
		print(
			value_output[
				[
					"Date",
					"Time",
					"League",
					"Home",
					"Away",
					"Result_Value_Side",
					"Result_Value_Prob",
					"Result_Value_Implied",
					"Result_Edge",
					"Result_EV",
					"Result_Budget_Share",
					"Result_Budget_Amount",
				]
			].to_string(index=False)
		)
		print(f"\nPositive EV games: {len(value_output)}")
		print(f"Suggested bets after min-stake filter: {len(recommended_output)}")
		print(
			f"Proposed total stake ({kelly_fraction:.2f} fraction, min_bet={min_bet_amount:.2f}): "
			f"{recommended_output['Result_Budget_Amount'].sum():.2f} from {fixed_budget:.2f}"
		)
	else:
		print("\nNo positive EV result bets found")

	print("\n--- Step 6: Email ---")
	if send_email_enabled:
		recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
		recipients = [recipient.strip() for recipient in recipients_str.split(",") if recipient.strip()]
		send_email(
			OUTPUT_HTML_PATH,
			output_df,
			value_output if not value_output.empty else None,
			recipients,
			report_url=report_public_url,
			attach_html=report_public_url is None,
		)
	else:
		print("SEND_EMAIL is disabled. Skipping email.")

	print("\nPipeline completed successfully.")


if __name__ == "__main__":
	main()
