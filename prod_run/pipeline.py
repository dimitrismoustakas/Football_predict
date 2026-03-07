"""
Production Pipeline for Over/Under Neural Network Model

NOTE: This pipeline uses the GatedResidualModelBinary architecture.
Requires a trained gated model - run architecture search to generate one.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import polars as pl
import torch
import joblib
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from prod_run import build_prod_features
from prod_run import fetch_odds
from prod_run.generate_html_report import generate_html_report
from training.models.neural_net import CategoricalConfig, GatedResidualModelBinary
from utils.paths import MODELS_DIR
from utils import calculate_betting_allocations, send_email

# Paths
DATA_DIR = Path("data")
PROD_DIR = DATA_DIR / "prod"
PREDICTIONS_DIR = DATA_DIR / "predictions"

MODEL_BUNDLE_CANDIDATES = [
	{
		"name": "legacy_gated",
		"model": MODELS_DIR / "over_under_gated.pt",
		"config": MODELS_DIR / "over_under_gated_metadata.json",
		"scaler": MODELS_DIR / "scaler_gated.joblib",
	},
	{
		"name": "fixed_arch_sweep",
		"model": MODELS_DIR / "over_under_fixed_arch_sweep.pt",
		"config": MODELS_DIR / "over_under_fixed_arch_sweep_config.json",
		"scaler": MODELS_DIR / "over_under_fixed_arch_sweep_scaler.joblib",
	},
]

PROD_FEATURES_PATH = PROD_DIR / "features_season.parquet"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"
OUTPUT_HTML_PATH = PREDICTIONS_DIR / "upcoming_predictions.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL LOADING
# ============================================================================

def _env_flag(name: str, default: bool) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_model_bundle() -> dict:
	"""Resolve the binary gated model artifact trio to use for production."""
	overrides = {
		"model": os.environ.get("MODEL_PATH"),
		"config": os.environ.get("MODEL_CONFIG_PATH"),
		"scaler": os.environ.get("SCALER_PATH"),
	}
	if any(overrides.values()):
		if not all(overrides.values()):
			raise RuntimeError(
				"MODEL_PATH, MODEL_CONFIG_PATH, and SCALER_PATH must all be set together."
			)
		bundle = {key: Path(value) for key, value in overrides.items()}
		bundle["name"] = "env_override"
		return bundle

	for candidate in MODEL_BUNDLE_CANDIDATES:
		if candidate["model"].exists() and candidate["config"].exists() and candidate["scaler"].exists():
			return candidate

	searched = [
		str(candidate["model"])
		for candidate in MODEL_BUNDLE_CANDIDATES
	]
	raise FileNotFoundError(
		"No production-ready binary gated model bundle found. Expected one of: "
		+ ", ".join(searched)
	)


def load_model():
	"""Load the gated model, metadata/config, and scaler."""
	print("Loading model and metadata...")
	bundle = resolve_model_bundle()
	print(f"Using model bundle: {bundle['name']}")

	with open(bundle["config"], "r") as f:
		meta = json.load(f)

	feature_cols = meta.get("features") or meta.get("feature_cols")
	if not feature_cols:
		raise ValueError(f"No feature column list found in {bundle['config']}")

	arch = meta.get("architecture", {})
	hidden_layers = meta.get("hidden_layers") or arch.get("hidden_layers")
	if not hidden_layers:
		raise ValueError(f"No hidden layer configuration found in {bundle['config']}")

	dropout = meta.get("dropout", arch.get("dropout", 0.3))
	norm = meta.get("norm", arch.get("norm", "none"))
	activation = meta.get("activation", arch.get("activation", "relu"))
	gate_hidden_dim = meta.get("gate_hidden_dim", arch.get("gate_hidden_dim", 32))
	gate_target_budget = meta.get("gate_target_budget", arch.get("gate_target_budget", 0.2))

	cat_config = None
	cat_config_dict = meta.get("cat_config")
	if cat_config_dict is not None:
		cat_config = CategoricalConfig(
			num_leagues=cat_config_dict["num_leagues"],
			league_embed_dim=cat_config_dict.get("league_embed_dim", 3),
		)

	model = GatedResidualModelBinary(
		input_dim=len(feature_cols),
		hidden_layers=hidden_layers,
		dropout=dropout,
		norm=norm,
		activation=activation,
		cat_config=cat_config,
		gate_hidden_dim=gate_hidden_dim,
		gate_target_budget=gate_target_budget,
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


# ============================================================================
# PREDICTION
# ============================================================================

def predict(
	model,
	scaler,
	feature_cols,
	X_raw: np.ndarray,
	cat_features: np.ndarray | None,
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
) -> np.ndarray:
	"""Generate predictions using the gated model."""
	# Scale features
	X_scaled = scaler.transform(X_raw)
	
	# Convert to tensors
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
	cat_tensor = None
	if cat_features is not None:
		cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(DEVICE)
	implied_tensor = torch.tensor(implied_probs, dtype=torch.float32).to(DEVICE)
	raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32).to(DEVICE)
	
	# Predict using gated model
	with torch.no_grad():
		pred_logits = model(X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
		probs = torch.sigmoid(pred_logits).view(-1).cpu().numpy()
	
	return probs


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
	odds_api_key = os.environ.get("ODDS_API_KEY")
	send_email_enabled = _env_flag("SEND_EMAIL", True)
	prediction_window_days = int(os.environ.get("PREDICTION_WINDOW_DAYS", "5"))

	print("=" * 60)
	print("OVER/UNDER NN PRODUCTION PIPELINE")
	print("=" * 60)
	
	# 1. Build/update features
	print("\n--- Step 1: Building Production Features ---")
	try:
		build_prod_features.main()
	except Exception as e:
		print(f"Error building features: {e}")
		return
	
	# 2. Load model
	print("\n--- Step 2: Loading Model ---")
	try:
		model, scaler, feature_cols, cat_config = load_model()
		print(f"Loaded model with {len(feature_cols)} features")
	except Exception as e:
		print(f"Error loading model: {e}")
		return
	
	# 3. Fetch odds from API (with caching)
	print("\n--- Step 3: Fetching Odds ---")
	raw_odds = fetch_odds.get_all_leagues_odds(odds_api_key)
	parsed_odds = fetch_odds.parse_odds_data(raw_odds)
	print(f"Fetched {len(parsed_odds)} games with over/under odds across all leagues")
	
	odds_df = pd.DataFrame(parsed_odds)
	
	# Convert commence_time to datetime (UTC timezone-aware)
	odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True)
	
	# Filter for upcoming games (today onwards) - use timezone-aware datetime
	now_utc = datetime.now(timezone.utc)
	today_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
	window_end_utc = today_utc + pd.Timedelta(days=prediction_window_days)
	odds_df = odds_df[(odds_df["commence_time"] >= today_utc) & (odds_df["commence_time"] < window_end_utc)]
	print(f"Found {len(odds_df)} upcoming games")
	
	if odds_df.empty:
		raise RuntimeError("No upcoming games found in odds data")
	
	features_df = pl.read_parquet(PROD_FEATURES_PATH)
	
	# Filter features to only supported leagues (same as odds)
	supported_leagues = list(fetch_odds.LEAGUE_TO_SPORT_KEY.keys())
	features_df = features_df.filter(pl.col("league").is_in(supported_leagues))
	print(f"Games in features for supported leagues: {len(features_df)}")
	
	# Convert features to pandas for merging
	features_pd = features_df.to_pandas()
	features_pd["date"] = pd.to_datetime(features_pd["date"], utc=True, errors="coerce")
	features_pd = features_pd[
		(features_pd["date"] >= today_utc) & (features_pd["date"] < window_end_utc)
	].copy()
	print(f"Upcoming games in features window: {len(features_pd)}")

	odds_merge_df = odds_df.copy()
	merge_keys = ["home_team", "away_team"]
	if "league" in features_pd.columns:
		odds_merge_df = odds_merge_df.rename(columns={"league_id": "league"})
		merge_keys = ["league", "home_team", "away_team"]
	elif "league_id" in features_pd.columns:
		merge_keys = ["league_id", "home_team", "away_team"]
	
	# Merge by team names (odds have the game, features have the stats)
	# Team names in odds_df are already mapped to canonical names by parse_odds_data
	merged = odds_merge_df.merge(
		features_pd,
		on=merge_keys,
		how="inner",
		suffixes=("_odds", "_feat")
	)
	
	print(f"Matched {len(merged)} games between odds and features")
	
	if merged.empty:
		# Debug: show what teams don't match
		print("\nOdds teams (mapped):")
		print(odds_merge_df[[c for c in ["league", "league_id", "home_team", "away_team", "home_team_raw", "away_team_raw"] if c in odds_merge_df.columns]].to_string())
		print("\nFeature teams (supported leagues, upcoming):")
		upcoming_features = features_pd[features_pd["date"] >= today_utc]
		print(upcoming_features[[c for c in ["league", "league_id", "home_team", "away_team"] if c in upcoming_features.columns]].head(30).to_string())
		raise RuntimeError("No games matched between odds and features")
	
	# 5. Check for required feature columns
	print("\n--- Step 5: Checking Features ---")
	missing_cols = [c for c in feature_cols if c not in merged.columns]
	if missing_cols:
		print(f"Missing feature columns: {missing_cols}")
		return

	cat_features = None
	if cat_config is not None:
		cat_cols = ["league_idx", "home_promoted", "away_promoted"]
		missing_cat_cols = [c for c in cat_cols if c not in merged.columns]
		if missing_cat_cols:
			print(f"Missing categorical feature columns: {missing_cat_cols}")
			return
	
	# Check for nulls in feature columns
	initial_count = len(merged)
	required_non_null = list(feature_cols)
	if cat_config is not None:
		required_non_null.extend(["league_idx", "home_promoted", "away_promoted"])
	merged = merged.dropna(subset=required_non_null).reset_index(drop=True)
	final_count = len(merged)
	
	if initial_count != final_count:
		print(f"Dropped {initial_count - final_count} games due to missing features")
	
	if merged.empty:
		print("No games remaining after dropping nulls")
		return
	
	# 6. Calculate implied probabilities and raw margin
	print("\n--- Step 6: Predicting ---")
	# Use odds from the odds file (suffixed with _odds after merge)
	odds_over_col = "odds_over_odds" if "odds_over_odds" in merged.columns else "odds_over"
	odds_under_col = "odds_under_odds" if "odds_under_odds" in merged.columns else "odds_under"
	
	implied_over = 1 / merged[odds_over_col]
	implied_under = 1 / merged[odds_under_col]
	norm = implied_over + implied_under
	implied_probs = (implied_over / norm).values
	
	# Raw margin = sum of implied probs - 1 (bookmaker's vig)
	raw_margin = (implied_over + implied_under - 1).values
	
	# Get feature matrix
	X_raw = merged[feature_cols].values
	if cat_config is not None:
		cat_features = merged[["league_idx", "home_promoted", "away_promoted"]].to_numpy(dtype=np.int64)
	
	# Predict
	probs = predict(model, scaler, feature_cols, X_raw, cat_features, implied_probs, raw_margin)
	
	print(f"Generated predictions for {len(probs)} games")
	
	# 7. Calculate betting allocations
	print("\n--- Step 7: Calculating Betting Allocations ---")
	allocations_df = calculate_betting_allocations(
		probs=probs,
		odds_over=merged[odds_over_col].values,
		odds_under=merged[odds_under_col].values,
		home_teams=merged["home_team"].tolist(),
		away_teams=merged["away_team"].tolist(),
		dates=merged["commence_time"].tolist(),
	)
	
	# 8. Build output DataFrame
	print("\n--- Step 8: Building Output ---")
	
	# Convert to Greek time (Europe/Athens)
	greek_times = merged["commence_time"].dt.tz_convert("Europe/Athens")
	
	output_df = pd.DataFrame({
		"Date": greek_times.dt.strftime("%Y-%m-%d"),
		"Time": greek_times.dt.strftime("%H:%M"),
		"Home": merged["home_team"],
		"Away": merged["away_team"],
		"Prob_Over": probs.round(3),
		"Prob_Under": (1 - probs).round(3),
		"Implied_Over": implied_probs.round(3),
		"Implied_Under": (1 - implied_probs).round(3),
		"Odds_Over": merged[odds_over_col],
		"Odds_Under": merged[odds_under_col],
		"Model_Odds_Over": (1 / probs).round(2),
		"Model_Odds_Under": (1 / (1 - probs)).round(2),
		"Bet_Side": allocations_df["bet_side"],
		"EV": allocations_df["mu"].round(4),
		"Allocation_Pct": allocations_df["allocation_pct"],
	})
	
	# Sort by date and time
	output_df = output_df.sort_values(["Date", "Time"])
	
	# 9. Save CSV and HTML
	print("\n--- Step 9: Saving Output ---")
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(OUTPUT_CSV_PATH, index=False)
	print(f"Saved predictions to {OUTPUT_CSV_PATH}")
	
	# Generate interactive HTML report
	generate_html_report(output_df, OUTPUT_HTML_PATH)
	
	# Print summary
	print("\n" + "=" * 60)
	print("PREDICTIONS SUMMARY")
	print("=" * 60)
	print(output_df.to_string(index=False))
	
	# Print betting recommendations
	bets = output_df[output_df["Allocation_Pct"] > 0].copy()
	if not bets.empty:
		print("\n" + "=" * 60)
		print("BETTING RECOMMENDATIONS")
		print("=" * 60)
		bets_display = bets[["Date", "Time", "Home", "Away", "Bet_Side", "Odds_Over", "Odds_Under", "EV", "Allocation_Pct"]]
		print(bets_display.to_string(index=False))
		print(f"\nTotal allocation: {bets['Allocation_Pct'].sum():.2f}%")
	else:
		print("\nNo positive EV bets found")
	
	# 10. Send email
	print("\n--- Step 10: Email ---")
	if send_email_enabled:
		recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
		recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
		send_email(OUTPUT_CSV_PATH, OUTPUT_HTML_PATH, output_df, bets if not bets.empty else None, recipients)
	else:
		print("SEND_EMAIL is disabled. Skipping email.")
	
	print("\nPipeline completed successfully.")


if __name__ == "__main__":
	main()
