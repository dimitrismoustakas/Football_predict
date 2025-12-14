"""
Production Pipeline for Over/Under Neural Network Model
"""
import os
import sys
import json
import smtplib
import numpy as np
import pandas as pd
import polars as pl
import torch
import joblib
from pathlib import Path
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from prod_run import build_prod_features
from training.pytorch_testing import MLP, _logits

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
ODDS_FILE_PATH = DATA_DIR / "odds for england.js"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# ODDS PARSING
# ============================================================================

def parse_odds_file(odds_path: Path) -> pd.DataFrame:
	"""
	Parse the JSON odds file and extract best odds for each game.
	Returns DataFrame with: home_team, away_team, commence_time, odds_over, odds_under
	"""
	with open(odds_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	
	games = []
	for game in data:
		home_team = game["home_team"]
		away_team = game["away_team"]
		commence_time = game["commence_time"]
		
		# Find best over/under odds across all bookmakers
		best_over = None
		best_under = None
		
		for bookmaker in game.get("bookmakers", []):
			for market in bookmaker.get("markets", []):
				if market["key"] == "totals":
					for outcome in market["outcomes"]:
						if outcome.get("point") == 2.5:
							if outcome["name"] == "Over":
								if best_over is None or outcome["price"] > best_over:
									best_over = outcome["price"]
							elif outcome["name"] == "Under":
								if best_under is None or outcome["price"] > best_under:
									best_under = outcome["price"]
		
		if best_over is not None and best_under is not None:
			games.append({
				"home_team_odds": home_team,
				"away_team_odds": away_team,
				"commence_time": commence_time,
				"odds_over": best_over,
				"odds_under": best_under,
			})
	
	return pd.DataFrame(games)


def create_team_mapping() -> dict:
	"""
	Create mapping from odds file team names to canonical names used in features.
	"""
	return {
		"Brighton and Hove Albion": "Brighton",
		"West Ham United": "West Ham",
		"Crystal Palace": "Crystal Palace",
		"Fulham": "Fulham",
		"Wolverhampton Wanderers": "Wolverhampton Wanderers",
		"Manchester United": "Manchester United",
		"Arsenal": "Arsenal",
		"Everton": "Everton",
		"Nottingham Forest": "Nottingham Forest",
		"Aston Villa": "Aston Villa",
		"Manchester City": "Manchester City",
		"Chelsea": "Chelsea",
		"Tottenham Hotspur": "Tottenham",
		"Leicester City": "Leicester",
		"Newcastle United": "Newcastle United",
		"Liverpool": "Liverpool",
		"Bournemouth": "Bournemouth",
		"Southampton": "Southampton",
		"Brentford": "Brentford",
		"Ipswich Town": "Ipswich",
	}


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
# PORTFOLIO STRATEGY
# ============================================================================

def calculate_betting_allocations(
	probs: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	home_teams: list,
	away_teams: list,
	dates: list,
	budget: float = 100.0,
) -> pd.DataFrame:
	"""
	Calculate betting allocations using Sharpe-weighted portfolio strategy.
	"""
	df = pd.DataFrame({
		"home_team": home_teams,
		"away_team": away_teams,
		"date": dates,
		"prob_over": probs,
		"odds_over": odds_over,
		"odds_under": odds_under,
	})
	
	p = df["prob_over"]
	o_over = df["odds_over"]
	o_under = df["odds_under"]
	
	# Expected value calculations
	# For Over bet: E[profit] = p * (odds - 1) + (1-p) * (-1) = p * odds - 1
	mu_over = p * o_over - 1
	# For Under bet: E[profit] = (1-p) * (odds - 1) + p * (-1) = (1-p) * odds - 1
	mu_under = (1 - p) * o_under - 1
	
	# Variance calculations
	# Var(profit) = E[profit^2] - E[profit]^2
	# For Over: outcomes are (odds-1) with prob p, or -1 with prob (1-p)
	e_x2_over = p * (o_over - 1) ** 2 + (1 - p) * 1
	var_over = e_x2_over - mu_over ** 2
	
	e_x2_under = (1 - p) * (o_under - 1) ** 2 + p * 1
	var_under = e_x2_under - mu_under ** 2
	
	# Determine better side for each game
	better_is_over = mu_over >= mu_under
	mu_best = np.where(better_is_over, mu_over, mu_under)
	var_best = np.where(better_is_over, var_over, var_under)
	odds_best = np.where(better_is_over, o_over, o_under)
	
	df["bet_side"] = np.where(better_is_over, "Over", "Under")
	df["mu"] = mu_best
	df["var"] = var_best
	df["odds_selected"] = odds_best
	df["eligible"] = df["mu"] > 0  # Only bet on positive EV
	
	# Calculate Sharpe-weighted allocations
	eligible_df = df[df["eligible"]].copy()
	
	if len(eligible_df) > 0:
		# Sharpe weight = mu / var (capped variance at small value for stability)
		vars_ = eligible_df["var"].values + 1e-6
		mus = eligible_df["mu"].values
		raw_weights = np.maximum(0, mus / vars_)
		sum_weights = raw_weights.sum()
		
		if sum_weights > 0:
			norm_weights = raw_weights / sum_weights
			eligible_df["allocation_pct"] = (norm_weights * 100).round(2)
		else:
			eligible_df["allocation_pct"] = 0.0
	
	# Merge back
	df = df.merge(
		eligible_df[["home_team", "away_team", "allocation_pct"]],
		on=["home_team", "away_team"],
		how="left"
	)
	df["allocation_pct"] = df["allocation_pct"].fillna(0.0)
	
	return df


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
	
	# 3. Load and parse odds
	print("\n--- Step 3: Parsing Odds ---")
	if not ODDS_FILE_PATH.exists():
		print(f"Odds file not found: {ODDS_FILE_PATH}")
		return
	
	odds_df = parse_odds_file(ODDS_FILE_PATH)
	print(f"Parsed {len(odds_df)} games with over/under odds")
	
	# Convert commence_time to datetime (UTC timezone-aware)
	odds_df["commence_time"] = pd.to_datetime(odds_df["commence_time"], utc=True)
	
	# Filter for upcoming games (today onwards) - use timezone-aware datetime
	now_utc = datetime.now(timezone.utc)
	today_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
	five_days_later = today_utc + pd.Timedelta(days=5)
	odds_df = odds_df[(odds_df["commence_time"] >= today_utc) & (odds_df["commence_time"] < five_days_later)]
	print(f"Found {len(odds_df)} upcoming games")
	
	if odds_df.empty:
		print("No upcoming games found in odds file")
		return
	
	features_df = pl.read_parquet(PROD_FEATURES_PATH)
	
	# Filter for EPL only
	features_df = features_df.filter(pl.col("league") == "ENG-Premier League")
	print(f"EPL games in features: {len(features_df)}")
	
	# Create team name mapping
	team_mapping = create_team_mapping()
	
	# Map odds team names to canonical names
	odds_df["home_team"] = odds_df["home_team_odds"].map(team_mapping).fillna(odds_df["home_team_odds"])
	odds_df["away_team"] = odds_df["away_team_odds"].map(team_mapping).fillna(odds_df["away_team_odds"])
	
	# Convert features to pandas for merging
	features_pd = features_df.to_pandas()
	
	# Merge by team names (odds have the game, features have the stats)
	merged = odds_df.merge(
		features_pd,
		on=["home_team", "away_team"],
		how="inner",
		suffixes=("_odds", "_feat")
	)
	
	print(f"Matched {len(merged)} games between odds and features")
	
	if merged.empty:
		print("No games matched between odds and features")
		# Debug: show what teams don't match
		print("\nOdds teams:")
		print(odds_df[["home_team", "away_team"]].to_string())
		print("\nFeature teams (EPL, upcoming):")
		upcoming_features = features_pd[pd.to_datetime(features_pd["date"]) >= pd.Timestamp.now()]
		print(upcoming_features[["home_team", "away_team"]].head(20).to_string())
		return
	
	# 5. Check for required feature columns
	print("\n--- Step 5: Checking Features ---")
	missing_cols = [c for c in feature_cols if c not in merged.columns]
	if missing_cols:
		print(f"Missing feature columns: {missing_cols}")
		return
	
	# Check for nulls in feature columns
	initial_count = len(merged)
	merged = merged.dropna(subset=feature_cols)
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
	
	output_df = pd.DataFrame({
		"Date": merged["commence_time"].dt.strftime("%Y-%m-%d"),
		"Time": merged["commence_time"].dt.strftime("%H:%M"),
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
	
	# 9. Save CSV
	print("\n--- Step 9: Saving Output ---")
	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(OUTPUT_CSV_PATH, index=False)
	print(f"Saved predictions to {OUTPUT_CSV_PATH}")
	
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
	
	send_email(OUTPUT_CSV_PATH, output_df, bets if not bets.empty else None, recipients)
	
	print("\nPipeline completed successfully.")


def send_email(csv_path: Path, predictions_df: pd.DataFrame, bets_df: pd.DataFrame, recipients: list):
	"""Send email with predictions CSV and betting recommendations."""
	if not recipients:
		print("No email recipients defined. Skipping email.")
		return
	
	sender_email = os.environ.get("EMAIL_USER")
	sender_password = os.environ.get("EMAIL_PASS")
	
	if not sender_email or not sender_password:
		print("EMAIL_USER or EMAIL_PASS not set. Skipping email.")
		return
	
	print(f"Sending email to {recipients}...")
	
	# Build email body
	today_str = datetime.now().strftime("%Y-%m-%d")
	
	html_body = f"""
	<html>
	<head>
		<style>
			table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }}
			th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
			th {{ background-color: #4CAF50; color: white; }}
			tr:nth-child(even) {{ background-color: #f2f2f2; }}
			.positive {{ color: green; font-weight: bold; }}
			.negative {{ color: red; }}
			h2 {{ color: #333; }}
		</style>
	</head>
	<body>
		<h2>Football Predictions - {today_str}</h2>
		<h3>Over/Under 2.5 Goals - English Premier League</h3>
		
		<h4>All Predictions</h4>
		{predictions_df.to_html(index=False, classes='predictions')}
	"""
	
	if bets_df is not None and not bets_df.empty:
		html_body += f"""
		<h4>Betting Recommendations (Positive EV)</h4>
		<p>Budget allocation percentages based on Sharpe-weighted portfolio strategy:</p>
		{bets_df[["Date", "Time", "Home", "Away", "Bet_Side", "Odds_Over", "Odds_Under", "EV", "Allocation_Pct"]].to_html(index=False, classes='bets')}
		<p><strong>Total Allocation: {bets_df['Allocation_Pct'].sum():.2f}%</strong></p>
		<p><em>To use: If your total budget is €10, multiply each Allocation_Pct by €0.1 to get the bet amount.</em></p>
		"""
	else:
		html_body += """
		<h4>Betting Recommendations</h4>
		<p>No positive expected value bets found for this period.</p>
		"""
	
	html_body += """
	</body>
	</html>
	"""
	
	# Create message
	msg = MIMEMultipart("alternative")
	msg["Subject"] = f"Football Predictions (NN) - {today_str}"
	msg["From"] = sender_email
	msg["To"] = ", ".join(recipients)
	
	# Attach HTML body
	msg.attach(MIMEText(html_body, "html"))
	
	# Attach CSV
	with open(csv_path, "rb") as f:
		part = MIMEBase("application", "octet-stream")
		part.set_payload(f.read())
		encoders.encode_base64(part)
		part.add_header("Content-Disposition", f"attachment; filename={csv_path.name}")
		msg.attach(part)
	
	try:
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
			smtp.login(sender_email, sender_password)
			smtp.send_message(msg)
		print("Email sent successfully.")
	except Exception as e:
		print(f"Failed to send email: {e}")


if __name__ == "__main__":
	main()
