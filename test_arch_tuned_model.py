"""
Simple script to test the architecture-tuned model on validation and test sets.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from training.models.neural_net import MLP
from training.train_utils import (
	load_frame,
	filter_min_history,
	add_targets_and_implied,
	prepare_data,
)
from utils.portfolio import evaluate_portfolio


def main():
	# Paths
	model_dir = Path("data/models")
	config_path = model_dir / "architecture_config.json"
	model_path = model_dir / "over_under_arch_tuned.pt"
	scaler_path = model_dir / "scaler_arch_tuned.joblib"
	data_path = Path("data/training/understat_df.parquet")

	# Load config
	print("Loading configuration...")
	with open(config_path) as f:
		config = json.load(f)

	feature_cols = config["feature_cols"]
	input_dim = config["input_dim"]
	hidden_layers = config["hidden_layers"]
	dropout = config["dropout"]
	norm = config["norm"]
	activation = config["activation"]

	# Load scaler
	print("Loading scaler...")
	scaler = joblib.load(scaler_path)

	# Load model
	print("Loading model...")
	model = MLP(
		input_dim=input_dim,
		hidden_layers=hidden_layers,
		dropout=dropout,
		norm=norm,
		activation=activation,
	)
	model.load_state_dict(torch.load(model_path, weights_only=True))
	model.eval()

	# Load and prepare data
	print("Loading data...")
	df = load_frame(data_path)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)

	# Get splits
	seasons = (
		df.select(df["season"].cast(str))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	val_season = seasons[-2]  # 2425
	test_season = seasons[-1]  # 2526 (current)
	train_seasons = seasons[:-2]
	
	print(f"Train seasons: {train_seasons}")
	print(f"Validation season: {val_season}")
	print(f"Test season: {test_season}")

	# Prepare validation data
	print("\nPreparing validation data...")
	val_data = prepare_data(df, feature_cols, [val_season], scaler=scaler, fit_scaler=False)
	X_val = val_data["X"]
	y_val = val_data["y"]

	# Prepare test data
	print("Preparing test data...")
	test_data = prepare_data(df, feature_cols, [test_season], scaler=scaler, fit_scaler=False)
	X_test = test_data["X"]
	y_test = test_data["y"]

	# Evaluate on validation set
	print("\n" + "="*60)
	print("VALIDATION SET RESULTS")
	print("="*60)
	with torch.no_grad():
		X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
		residuals = model(X_val_tensor).squeeze().numpy()
		implied_logits = np.log(val_data["implied"]) - np.log(1 - val_data["implied"])
		pred_logits = residuals + implied_logits
		val_probs = 1 / (1 + np.exp(-pred_logits))

	val_preds = (val_probs > 0.5).astype(int)
	val_acc = accuracy_score(y_val, val_preds)
	val_brier = brier_score_loss(y_val, val_probs)
	val_logloss = log_loss(y_val, val_probs)

	# Portfolio evaluation
	val_portfolio = evaluate_portfolio(
		probs=val_probs,
		y_true=y_val,
		odds_over=val_data["odds_over"],
		odds_under=val_data["odds_under"],
		dates=val_data["dates"],
		budget_per_day=10.0,
	)

	print(f"Samples: {len(y_val)}")
	print(f"Accuracy: {val_acc:.4f}")
	print(f"Brier Score: {val_brier:.4f}")
	print(f"Log Loss: {val_logloss:.4f}")
	print(f"\nPortfolio Metrics:")
	print(f"  Total Profit: {val_portfolio['sharpe_total_profit']:.2f}")
	print(f"  Avg Daily Profit: {val_portfolio['sharpe_avg_daily_profit']:.2f}")
	print(f"  Sharpe Ratio: {val_portfolio['sharpe_ratio']:.4f}")
	print(f"  Trading Days: {val_portfolio['n_days']}")

	# Evaluate on test set
	print("\n" + "="*60)
	print("TEST SET RESULTS")
	print("="*60)
	with torch.no_grad():
		X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
		residuals = model(X_test_tensor).squeeze().numpy()
		implied_logits = np.log(test_data["implied"]) - np.log(1 - test_data["implied"])
		pred_logits = residuals + implied_logits
		test_probs = 1 / (1 + np.exp(-pred_logits))

	test_preds = (test_probs > 0.5).astype(int)
	test_acc = accuracy_score(y_test, test_preds)
	test_brier = brier_score_loss(y_test, test_probs)
	test_logloss = log_loss(y_test, test_probs)

	# Portfolio evaluation
	test_portfolio = evaluate_portfolio(
		probs=test_probs,
		y_true=y_test,
		odds_over=test_data["odds_over"],
		odds_under=test_data["odds_under"],
		dates=test_data["dates"],
		budget_per_day=10.0,
	)

	print(f"Samples: {len(y_test)}")
	print(f"Accuracy: {test_acc:.4f}")
	print(f"Brier Score: {test_brier:.4f}")
	print(f"Log Loss: {test_logloss:.4f}")
	print(f"\nPortfolio Metrics:")
	print(f"  Total Profit: {test_portfolio['sharpe_total_profit']:.2f}")
	print(f"  Avg Daily Profit: {test_portfolio['sharpe_avg_daily_profit']:.2f}")
	print(f"  Sharpe Ratio: {test_portfolio['sharpe_ratio']:.4f}")
	print(f"  Trading Days: {test_portfolio['n_days']}")

	print("\nDone!")


if __name__ == "__main__":
	main()
