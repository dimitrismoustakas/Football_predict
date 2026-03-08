"""
Production pipeline for the canonical match-result model.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import torch
from dotenv import load_dotenv

def _load_project_modules():
	try:
		from prod_run import build_prod_features, fetch_odds
		from prod_run.generate_html_report import generate_html_report
		from training.models.neural_net import CategoricalConfig, GatedResidualModel
		from utils import send_email
		from utils.paths import MODELS_DIR
	except ModuleNotFoundError:
		project_root = Path(__file__).resolve().parent.parent
		if str(project_root) not in sys.path:
			sys.path.insert(0, str(project_root))
		from prod_run import build_prod_features, fetch_odds
		from prod_run.generate_html_report import generate_html_report
		from training.models.neural_net import CategoricalConfig, GatedResidualModel
		from utils import send_email
		from utils.paths import MODELS_DIR
	return build_prod_features, fetch_odds, generate_html_report, CategoricalConfig, GatedResidualModel, send_email, MODELS_DIR


build_prod_features, fetch_odds, generate_html_report, CategoricalConfig, GatedResidualModel, send_email, MODELS_DIR = _load_project_modules()

load_dotenv()

DATA_DIR = Path("data")
PROD_DIR = DATA_DIR / "prod"
PREDICTIONS_DIR = DATA_DIR / "predictions"
MODEL_BUNDLE = {
	"name": "result_main",
	"model": MODELS_DIR / "result_model.pt",
	"config": MODELS_DIR / "result_model_config.json",
	"scaler": MODELS_DIR / "result_model_scaler.joblib",
}
PROD_FEATURES_PATH = PROD_DIR / "features_season.parquet"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"
OUTPUT_HTML_PATH = PREDICTIONS_DIR / "upcoming_predictions.html"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_LABELS = np.array(["Home", "Draw", "Away"])


def _env_flag(name: str, default: bool) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_model_bundle() -> dict:
	if not all(path.exists() for key, path in MODEL_BUNDLE.items() if key != "name"):
		raise FileNotFoundError(
			f"Missing production model bundle. Expected: {MODEL_BUNDLE['model']}, {MODEL_BUNDLE['config']}, {MODEL_BUNDLE['scaler']}"
		)
	return MODEL_BUNDLE


def load_model():
	bundle = resolve_model_bundle()
	print(f"Loading model bundle: {bundle['name']}")
	with open(bundle["config"], "r", encoding="utf-8") as file:
		meta = json.load(file)

	feature_cols = meta.get("feature_cols")
	if not feature_cols:
		raise ValueError(f"No feature column list found in {bundle['config']}")

	cat_config = None
	cat_config_dict = meta.get("cat_config")
	if cat_config_dict is not None:
		cat_config = CategoricalConfig(
			num_leagues=cat_config_dict["num_leagues"],
			league_embed_dim=cat_config_dict.get("league_embed_dim", 3),
		)

	model = GatedResidualModel(
		input_dim=len(feature_cols),
		hidden_layers=meta["hidden_layers"],
		n_classes=3,
		cat_config=cat_config,
		gate_hidden_dim=meta.get("gate_hidden_dim", 32),
		dropout=meta.get("dropout", 0.3),
		norm=meta.get("norm", "none"),
		activation=meta.get("activation", "relu"),
		gate_target_budget=meta.get("gate_target_budget", 0.2),
	)
	try:
		state_dict = torch.load(bundle["model"], map_location=DEVICE, weights_only=True)
	except TypeError:
		state_dict = torch.load(bundle["model"], map_location=DEVICE)
	model.load_state_dict(state_dict)
	model.to(DEVICE)
	model.eval()
	scaler = joblib.load(bundle["scaler"])
	return model, scaler, feature_cols, cat_config


def resolve_merged_col(frame: pd.DataFrame, base_name: str) -> str:
	if f"{base_name}_odds" in frame.columns:
		return f"{base_name}_odds"
	return base_name


def predict_result(model, scaler, feature_cols, X_raw, cat_features, implied_probs, raw_margin) -> np.ndarray:
	X_scaled = scaler.transform(X_raw)
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
	cat_tensor = None
	if cat_features is not None:
		cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(DEVICE)
	implied_tensor = torch.tensor(implied_probs, dtype=torch.float32).to(DEVICE)
	raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32).to(DEVICE)
	with torch.no_grad():
		pred_logits = model(X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
		return torch.softmax(pred_logits, dim=-1).cpu().numpy()


def score_result_predictions(merged: pd.DataFrame, model_bundle) -> pd.DataFrame:
	model, scaler, feature_cols, cat_config = model_bundle
	odds_home_col = resolve_merged_col(merged, "odds_home")
	odds_draw_col = resolve_merged_col(merged, "odds_draw")
	odds_away_col = resolve_merged_col(merged, "odds_away")
	required_cols = list(feature_cols) + [odds_home_col, odds_draw_col, odds_away_col]
	if cat_config is not None:
		required_cols.extend(["league_idx", "home_promoted", "away_promoted"])

	ready = merged.dropna(subset=required_cols).copy()
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
	cat_features = None
	if cat_config is not None:
		cat_features = ready[["league_idx", "home_promoted", "away_promoted"]].to_numpy(dtype=np.int64)

	probs = predict_result(
		model=model,
		scaler=scaler,
		feature_cols=feature_cols,
		X_raw=ready[feature_cols].to_numpy(dtype=float),
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
	ev = probs * odds_matrix - 1
	value_pick_idx = np.argmax(ev, axis=1)
	best_ev = ev[np.arange(len(ev)), value_pick_idx]

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
		"Result_Value_Side": np.where(best_ev > 0, RESULT_LABELS[value_pick_idx], ""),
		"Result_EV": np.where(best_ev > 0, np.round(best_ev, 4), np.nan),
	})


def main():
	odds_api_key = os.environ.get("ODDS_API_KEY")
	send_email_enabled = _env_flag("SEND_EMAIL", True)
	prediction_window_days = int(os.environ.get("PREDICTION_WINDOW_DAYS", "5"))

	print("=" * 60)
	print("FOOTBALL PRODUCTION PIPELINE")
	print("=" * 60)

	print("\n--- Step 1: Building Production Features ---")
	build_prod_features.main()

	print("\n--- Step 2: Loading Model ---")
	model_bundle = load_model()
	print(f"Loaded result model with {len(model_bundle[2])} features")

	print("\n--- Step 3: Fetching Odds ---")
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
	result_predictions = score_result_predictions(merged, model_bundle)
	if result_predictions.empty:
		raise RuntimeError("No games had all required features for scoring")

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
	output_df = output_df.sort_values(["Date", "Time", "League", "Home", "Away"]).reset_index(drop=True)
	value_output = output_df[output_df["Result_EV"].notna()].copy()

	print("\n--- Step 5: Saving Output ---")
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(OUTPUT_CSV_PATH, index=False)
	print(f"Saved predictions to {OUTPUT_CSV_PATH}")
	generate_html_report(output_df, OUTPUT_HTML_PATH)

	print("\n" + "=" * 60)
	print("PREDICTIONS SUMMARY")
	print("=" * 60)
	print(output_df.to_string(index=False))

	if not value_output.empty:
		print("\n" + "=" * 60)
		print("RESULT VALUE RECOMMENDATIONS")
		print("=" * 60)
		print(value_output[["Date", "Time", "League", "Home", "Away", "Result_Value_Side", "Result_EV"]].to_string(index=False))
	else:
		print("\nNo positive EV result bets found")

	print("\n--- Step 6: Email ---")
	if send_email_enabled:
		recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
		recipients = [recipient.strip() for recipient in recipients_str.split(",") if recipient.strip()]
		send_email(OUTPUT_CSV_PATH, OUTPUT_HTML_PATH, output_df, value_output if not value_output.empty else None, recipients)
	else:
		print("SEND_EMAIL is disabled. Skipping email.")

	print("\nPipeline completed successfully.")


if __name__ == "__main__":
	main()
