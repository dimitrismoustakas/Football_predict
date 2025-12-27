"""
Production Pipeline for Over/Under Neural Network Model
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
from training.models.neural_net import MLP, _logits
from utils import calculate_betting_allocations, send_email

# API Key
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
if not ODDS_API_KEY:
	raise RuntimeError("ODDS_API_KEY not set in environment")

# Paths
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
PROD_DIR = DATA_DIR / "prod"
PREDICTIONS_DIR = DATA_DIR / "predictions"

# Use the Sharpe-optimized model (final model from Phase 2)
MODEL_PATH = MODELS_DIR / "over_under_decorrelated.pt"
META_PATH = MODELS_DIR / "over_under_metadata.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PROD_FEATURES_PATH = PROD_DIR / "features_season.parquet"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"
OUTPUT_HTML_PATH = PREDICTIONS_DIR / "upcoming_predictions.html"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
	"""Load the model, metadata, and scaler."""
	print("Loading model and metadata...")
	
	if not META_PATH.exists():
		raise FileNotFoundError(f"Meta file not found: {META_PATH}")
	if not MODEL_PATH.exists():
		raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
	if not SCALER_PATH.exists():
		raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")
	
	with open(META_PATH, "r") as f:
		meta = json.load(f)
	
	feature_cols = meta["features"]
	arch = meta["architecture"]
	
	# Build model using imported MLP class
	model = MLP(
		input_dim=len(feature_cols),
		hidden_layers=arch["hidden_layers"],
		dropout=arch["dropout"],
		norm=arch["norm"],
	)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
	model.to(DEVICE)
	model.eval()
	
	scaler = joblib.load(SCALER_PATH)
	
	return model, scaler, feature_cols


# ============================================================================
# PREDICTION
# ============================================================================

def predict(model, scaler, feature_cols, X_raw: np.ndarray, implied_probs: np.ndarray) -> np.ndarray:
	# Scale features
	X_scaled = scaler.transform(X_raw)
	
	# Convert to tensors
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
	implied_tensor = torch.tensor(implied_probs, dtype=torch.float32).to(DEVICE)
	
	# Predict
	with torch.no_grad():
		residual_logits = model(X_tensor).squeeze(1)
		implied_logits = _logits(implied_tensor)
		pred_logits = residual_logits + implied_logits
		probs = torch.sigmoid(pred_logits).cpu().numpy()
	
	return probs


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
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
		model, scaler, feature_cols = load_model()
		print(f"Loaded model with {len(feature_cols)} features")
	except Exception as e:
		print(f"Error loading model: {e}")
		return
	
	# 3. Fetch odds from API (with caching)
	print("\n--- Step 3: Fetching Odds ---")
	raw_odds = fetch_odds.get_all_leagues_odds(ODDS_API_KEY)
	parsed_odds = fetch_odds.parse_odds_data(raw_odds)
	print(f"Fetched {len(parsed_odds)} games with over/under odds across all leagues")
	
	odds_df = pd.DataFrame(parsed_odds)
	
	# Convert commence_time to datetime (UTC timezone-aware)
	odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True)
	
	# Filter for upcoming games (today onwards) - use timezone-aware datetime
	now_utc = datetime.now(timezone.utc)
	today_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
	five_days_later = today_utc + pd.Timedelta(days=5)
	odds_df = odds_df[(odds_df["commence_time"] >= today_utc) & (odds_df["commence_time"] < five_days_later)]
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
	
	# Merge by team names (odds have the game, features have the stats)
	# Team names in odds_df are already mapped to canonical names by parse_odds_data
	merged = odds_df.merge(
		features_pd,
		on=["home_team", "away_team"],
		how="inner",
		suffixes=("_odds", "_feat")
	)
	
	print(f"Matched {len(merged)} games between odds and features")
	
	if merged.empty:
		# Debug: show what teams don't match
		print("\nOdds teams (mapped):")
		print(odds_df[["home_team", "away_team", "home_team_raw", "away_team_raw"]].to_string())
		print("\nFeature teams (supported leagues, upcoming):")
		upcoming_features = features_pd[pd.to_datetime(features_pd["date"]) >= pd.Timestamp.now()]
		print(upcoming_features[["league", "home_team", "away_team"]].head(30).to_string())
		raise RuntimeError("No games matched between odds and features")
	
	# 5. Check for required feature columns
	print("\n--- Step 5: Checking Features ---")
	missing_cols = [c for c in feature_cols if c not in merged.columns]
	if missing_cols:
		print(f"Missing feature columns: {missing_cols}")
		return
	
	# Check for nulls in feature columns
	initial_count = len(merged)
	merged = merged.dropna(subset=feature_cols).reset_index(drop=True)
	final_count = len(merged)
	
	if initial_count != final_count:
		print(f"Dropped {initial_count - final_count} games due to missing features")
	
	if merged.empty:
		print("No games remaining after dropping nulls")
		return
	
	# 6. Calculate implied probabilities
	print("\n--- Step 6: Predicting ---")
	# Use odds from the odds file (suffixed with _odds after merge)
	odds_over_col = "odds_over_odds" if "odds_over_odds" in merged.columns else "odds_over"
	odds_under_col = "odds_under_odds" if "odds_under_odds" in merged.columns else "odds_under"
	
	implied_over = 1 / merged[odds_over_col]
	implied_under = 1 / merged[odds_under_col]
	norm = implied_over + implied_under
	implied_probs = (implied_over / norm).values
	
	# Get feature matrix
	X_raw = merged[feature_cols].values
	
	# Predict
	probs = predict(model, scaler, feature_cols, X_raw, implied_probs)
	
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
	print("\n--- Step 10: Sending Email ---")
	recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
	recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
	
	send_email(OUTPUT_CSV_PATH, OUTPUT_HTML_PATH, output_df, bets if not bets.empty else None, recipients)
	
	print("\nPipeline completed successfully.")


if __name__ == "__main__":
	main()
