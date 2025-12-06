import os
import sys
import json
import smtplib
import pandas as pd
import polars as pl
import joblib
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path to allow imports
sys.path.append(os.getcwd())

# Import the feature building script
from prod_run import build_prod_features

# Paths
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
PROD_DIR = DATA_DIR / "prod"
PREDICTIONS_DIR = DATA_DIR / "predictions"

FEATURES_META_PATH = MODELS_DIR / "features_and_meta.json"
RESULT_MODEL_PATH = MODELS_DIR / "result_model.joblib"
OVER_MODEL_PATH = MODELS_DIR / "over_model.joblib"
PROD_FEATURES_PATH = PROD_DIR / "features_season.parquet"
OUTPUT_CSV_PATH = PREDICTIONS_DIR / "upcoming_predictions.csv"

def load_models_and_meta():
    print("Loading models and metadata...")
    if not FEATURES_META_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {FEATURES_META_PATH}")
    
    with open(FEATURES_META_PATH, "r") as f:
        meta = json.load(f)
    
    result_model = joblib.load(RESULT_MODEL_PATH)
    over_model = joblib.load(OVER_MODEL_PATH)
    
    return result_model, over_model, meta["feature_cols"]

def predict_upcoming_games(result_model, over_model, feature_cols):
    print(f"Loading production features from {PROD_FEATURES_PATH}...")
    if not PROD_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found at {PROD_FEATURES_PATH}. Did build_prod_features.py run successfully?")
    
    df = pl.read_parquet(PROD_FEATURES_PATH)
    
    if df.is_empty():
        print("No upcoming games found in the features file.")
        return pd.DataFrame()

    # Filter for Today + 3 days (Today, Tomorrow, +2 days)
    now = datetime.now()
    today = datetime(now.year, now.month, now.day)
    end_window = today + pd.Timedelta(days=4)
    
    print(f"Filtering games from {today.date()} to {(end_window - pd.Timedelta(days=1)).date()}")
    
    df = df.filter(
        (pl.col("date") >= today) & 
        (pl.col("date") < end_window)
    )

    if df.is_empty():
        print("No games found in the selected date window.")
        return pd.DataFrame()

    # Ensure all required features exist
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in production data: {missing_cols}")
    
    # Check for nulls in feature columns
    initial_count = df.height
    df = df.drop_nulls(subset=feature_cols)
    final_count = df.height
    dropped_count = initial_count - final_count
    
    if dropped_count > 0:
        print(f"Dropped {dropped_count} games due to missing features (insufficient history). Remaining: {final_count}")
    
    if df.is_empty():
        print("No games remaining after dropping null features.")
        return pd.DataFrame()

    # Select features and convert to pandas for sklearn
    X = df.select(feature_cols).to_pandas()
    
    # Predict Result (RF)
    print("Predicting Match Results...")
    res_probs = result_model.predict_proba(X)
    res_preds = result_model.predict(X)
    
    # Predict Over/Under (LR)
    print("Predicting Over/Under 2.5...")
    over_probs = over_model.predict_proba(X)[:, 1] # Probability of class 1 (Over)
    
    # Construct Output DataFrame
    # We need: Home, Away, Date, Time, Prediction, Probabilities
    
    # Select basic info plus bookie odds
    cols_to_select = ["date", "home_team", "away_team", "league"]
    odds_cols = ["odds_h", "odds_d", "odds_a", "odds_over", "odds_under"]
    
    # Only select odds columns if they exist in the dataframe
    for col in odds_cols:
        if col in df.columns:
            cols_to_select.append(col)
            
    output_df = df.select(cols_to_select).to_pandas()
    
    # Format Date and Time
    # Assuming 'date' is datetime64[ns]
    # Convert to Athens time if possible, or just keep as is if it's already local/UTC
    # The build_prod_features script ensures 'date' is datetime.
    # Usually soccerdata returns timestamps. Let's try to localize if naive, or convert if aware.
    
    # Check if timezone aware
    if output_df["date"].dt.tz is None:
        # Assume UTC if naive, or just leave it. 
        # If the source was soccerdata, it might be UTC.
        # Let's assume UTC for safety and convert to Athens.
        output_df["date"] = output_df["date"].dt.tz_localize("UTC")
    
    output_df["local_time"] = output_df["date"].dt.tz_convert("Europe/Athens")
    
    output_df["Day"] = output_df["local_time"].dt.strftime("%Y-%m-%d")
    output_df["Time"] = output_df["local_time"].dt.strftime("%H:%M")
    
    # Add Predictions
    output_df["Result_Pred"] = res_preds
    
    # Add Probabilities (RF classes are usually sorted, but let's check model.classes_)
    classes = result_model.classes_ # e.g. ['A', 'D', 'H']
    for i, cls in enumerate(classes):
        prob = res_probs[:, i]
        output_df[f"Prob_{cls}"] = prob.round(3)
        output_df[f"Odds_{cls}"] = (1 / prob).round(2)
        
    output_df["Prob_Over_2.5"] = over_probs.round(3)
    output_df["Odds_Over_2.5"] = (1 / over_probs).round(2)
    
    prob_under = 1 - over_probs
    output_df["Prob_Under_2.5"] = prob_under.round(3)
    output_df["Odds_Under_2.5"] = (1 / prob_under).round(2)
    
    output_df["Over_Under_Pred"] = ["Over" if p >= 0.5 else "Under" for p in over_probs]
    
    # Rename bookie odds columns for clarity
    rename_map = {
        "odds_h": "Bookie_H",
        "odds_d": "Bookie_D",
        "odds_a": "Bookie_A",
        "odds_over": "Bookie_Over",
        "odds_under": "Bookie_Under"
    }
    output_df.rename(columns=rename_map, inplace=True)

    # Reorder columns
    cols_order = [
        "Day", "Time", "league", "home_team", "away_team", 
        "Result_Pred", "Prob_H", "Odds_H", "Bookie_H", "Prob_D", "Odds_D", "Bookie_D", "Prob_A", "Odds_A", "Bookie_A",
        "Over_Under_Pred", "Prob_Over_2.5", "Odds_Over_2.5", "Bookie_Over", "Prob_Under_2.5", "Odds_Under_2.5", "Bookie_Under"
    ]
    
    # Filter cols_order to only include columns that actually exist
    cols_order = [c for c in cols_order if c in output_df.columns]
    
    return output_df[cols_order]

def send_email(csv_path, recipients):
    if not recipients:
        print("No email recipients defined. Skipping email.")
        return

    sender_email = os.environ.get("EMAIL_USER")
    sender_password = os.environ.get("EMAIL_PASS")
    
    if not sender_email or not sender_password:
        print("EMAIL_USER or EMAIL_PASS environment variables not set. Skipping email.")
        return

    print(f"Sending email to {recipients}...")
    
    msg = EmailMessage()
    msg["Subject"] = f"Football Predictions - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg.set_content("Please find attached the predictions for the upcoming football games.")

    with open(csv_path, "rb") as f:
        file_data = f.read()
        file_name = csv_path.name
        msg.add_attachment(file_data, maintype="text", subtype="csv", filename=file_name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def main():
    print("Starting Prediction Pipeline...")
    
    # 1. Update Data and Features
    print("\n--- Step 1: Building Production Features ---")
    try:
        build_prod_features.main()
    except Exception as e:
        print(f"Error building features: {e}")
        return

    # 2. Load Models
    print("\n--- Step 2: Loading Models ---")
    try:
        result_model, over_model, feature_cols = load_models_and_meta()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 3. Predict
    print("\n--- Step 3: Generating Predictions ---")
    try:
        predictions_df = predict_upcoming_games(result_model, over_model, feature_cols)
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return

    if predictions_df.empty:
        print("No predictions generated.")
        return

    # 4. Save Output
    print("\n--- Step 4: Saving Output ---")
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_CSV_PATH}")
    
    # 5. Email
    print("\n--- Step 5: Sending Email ---")
    # Get recipients from env var (comma separated)
    recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    
    send_email(OUTPUT_CSV_PATH, recipients)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
