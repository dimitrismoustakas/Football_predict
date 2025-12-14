"""
Over/Under Neural Network with Hyperparameter Optimization

To view MLflow results:
	cd to project root, then run: mlflow ui
	Open http://127.0.0.1:5000 in your browser
"""

import os
import json
from pathlib import Path
from typing import Dict, List

import joblib
import mlflow
import optuna
import polars as pl
import torch

from training.evaluation import evaluate_model, plot_losses
from training.models import TrainConfig
from training.train_utils import (
	add_targets_and_implied,
	filter_min_history,
	load_frame,
	prepare_data,
	select_feature_columns,
	to_loader,
	train_model,
	train_test_season_splits,
	evaluate_implied_baseline
)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

# Hyperparameter search settings
N_TRIALS_ARCHITECTURE = 50
N_TRIALS_LAMBDA = 50
MAX_EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 128

MANUAL_ARCH_PARAMS = None

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Device Detection ---\nDevice: {DEVICE}")
if DEVICE.type == "cuda":
	print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print("------------------------\n")

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================


def create_architecture_objective(
	df: pl.DataFrame,
	feature_cols: List[str],
	train_seasons: List[str],
	val_season: str,
):
	"""
	Creates an Optuna objective for architecture search.
	lambda_penalty is fixed at 0 during this phase.
	"""
	# Prepare data once
	data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
	train_loader = to_loader(data_train, BATCH_SIZE, device=DEVICE)
	val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False, device=DEVICE)
	input_dim = data_train["X"].shape[1]

	def objective(trial: optuna.Trial) -> float:
		# Sample architecture hyperparameters
		n_layers = trial.suggest_int("n_layers", 2, 5)
		layer_sizes = []
		for i in range(n_layers):
			size = trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256, 512])
			layer_sizes.append(size)

		dropout = trial.suggest_float("dropout", 0.1, 0.5)
		norm = trial.suggest_categorical("norm", ["none", "ln"])
		lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
		weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

		config = TrainConfig(
			input_dim=input_dim,
			hidden_layers=layer_sizes,
			dropout=dropout,
			norm=norm,
			lr=lr,
			weight_decay=weight_decay,
			lambda_repulsion=0.0,
			lambda_corr=0.0,
		)

		_, _, best_val_loss = train_model(
			config, train_loader, val_loader, device=DEVICE, trial=trial, verbose=False
		)

		return best_val_loss

	return objective, data_train, data_val


def create_lambda_objective(
	df: pl.DataFrame,
	feature_cols: List[str],
	train_seasons: List[str],
	val_season: str,
	best_arch_params: Dict,
):
	"""
	Creates an Optuna objective for lambda tuning.
	Takes the best architecture from Phase 1 and retrains it with different
	lambda values.
	Architecture (layers, dropout, lr, etc.) is fixed from best_arch_params.
	Only lambda_repulsion and lambda_corr vary across trials.
	"""
	data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
	train_loader = to_loader(data_train, BATCH_SIZE, device=DEVICE)
	val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False, device=DEVICE)
	input_dim = data_train["X"].shape[1]

	# Extract architecture from best params (from Phase 1)
	n_layers = best_arch_params["n_layers"]
	hidden_layers = [best_arch_params[f"layer_{i}_size"] for i in range(n_layers)]

	def objective(trial: optuna.Trial) -> float:
		lambda_repulsion = trial.suggest_float("lambda_repulsion", 0.0, 0.5)
		lambda_corr = trial.suggest_float("lambda_corr", 0.0001, 0.3, log=True)

		# Same architecture, only lambda changes
		config = TrainConfig(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=best_arch_params["dropout"],
			norm=best_arch_params["norm"],
			lr=best_arch_params["lr"],
			weight_decay=best_arch_params["weight_decay"],
			lambda_repulsion=lambda_repulsion,
			lambda_corr=lambda_corr,
		)

		# Retrain model from scratch with this lambda value
		model, _, _ = train_model(config, train_loader, val_loader, device=DEVICE, verbose=False)
		metrics = evaluate_model(model, data_val, device=DEVICE, verbose=False)

		# Log all metrics to MLflow
		if mlflow.active_run():
			mlflow.log_metrics(
				{
					"val_accuracy": metrics["accuracy"],
					"val_brier": metrics["brier"],
					"val_log_loss": metrics["log_loss"],
					"val_corr": metrics["corr_with_implied"],
					"val_sharpe_ratio": metrics["sharpe_ratio"],
					"val_sharpe_profit": metrics["sharpe_total_profit"],
					"val_n_bets": metrics["n_bets"],
				}
			)

		# Maximize Sharpe ratio (return negative for minimization)
		return -metrics["sharpe_ratio"]

	return objective, data_train, data_val


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline():
	"""Main entry point for the training pipeline."""
	print("=" * 60)
	print("OVER/UNDER NEURAL NETWORK TRAINING PIPELINE")
	print("=" * 60)

	def set_seed(s=42):
		import random
		random.seed(s)
		import numpy as np
		np.random.seed(s)
		torch.manual_seed(s)
		torch.cuda.manual_seed_all(s)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	set_seed(42)

	# Load and prepare data
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)

	train_seasons, val_season, test_season = train_test_season_splits(df)
	print(f"Train: {train_seasons}")
	print(f"Validation: {val_season}")
	print(f"Test (current): {test_season}")

	base_feats = select_feature_columns(df)
	print(f"Features: {len(base_feats)} columns")

	# Filter to rows with valid odds
	df = df.drop_nulls(subset=["odds_over", "odds_under"])
	print(f"Total rows with odds: {len(df)}")

	# Set up MLflow
	mlflow.set_experiment("over_under_nn")

	best_arch_params = None
	data_train = None
	data_val = None

	# ========================================================================
	# PHASE 1: Architecture Search (lambdas=0)
	# ========================================================================
	
	with mlflow.start_run(run_name="architecture_search"):
		mlflow.log_param("phase", "architecture_search")
		mlflow.log_param("n_trials", N_TRIALS_ARCHITECTURE)
		mlflow.log_param("train_seasons", str(train_seasons))
		mlflow.log_param("val_season", val_season)

		objective_arch, data_train, data_val = create_architecture_objective(
			df, base_feats, train_seasons, val_season
		)

		study_arch = optuna.create_study(
			direction="minimize",
			pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
			study_name="architecture_search",
		)

		study_arch.optimize(
			objective_arch,
			n_trials=N_TRIALS_ARCHITECTURE,
			show_progress_bar=True,
		)

		best_arch_params = study_arch.best_params
		print(f"\nBest architecture params: {best_arch_params}")
		print(f"Best validation loss: {study_arch.best_value:.5f}")

		mlflow.log_params(best_arch_params)
		mlflow.log_metric("best_val_loss", study_arch.best_value)

	# ========================================================================
	# Train final model with best architecture (lambda=0)
	# ========================================================================
	print("\n" + "=" * 60)
	print("TRAINING FINAL MODEL WITH BEST ARCHITECTURE")
	print("=" * 60)

	n_layers = best_arch_params["n_layers"]
	best_hidden_layers = [best_arch_params[f"layer_{i}_size"] for i in range(n_layers)]

	with mlflow.start_run(run_name="best_model_lambda0"):
		mlflow.log_params(best_arch_params)
		mlflow.log_param("lambda_repulsion", 0.0)
		mlflow.log_param("lambda_corr", 0.0)

		config = TrainConfig(
			input_dim=data_train["X"].shape[1],
			hidden_layers=best_hidden_layers,
			dropout=best_arch_params["dropout"],
			norm=best_arch_params["norm"],
			lr=best_arch_params["lr"],
			weight_decay=best_arch_params["weight_decay"],
			lambda_repulsion=0.0,
			lambda_corr=0.0,
		)

		train_loader = to_loader(data_train, BATCH_SIZE, device=DEVICE)
		val_loader = to_loader(data_val, BATCH_SIZE, shuffle=False, device=DEVICE)

		print("Training best model...")
		best_model, history, _ = train_model(
			config, train_loader, val_loader, device=DEVICE, verbose=True
		)

		# Plot and log loss curves
		plot_path = PLOTS_DIR / "loss_best_model.png"
		plot_losses(history, "Best Model (lambdas=0)", plot_path)
		mlflow.log_artifact(str(plot_path))

		# Evaluate on validation
		print("\n--- Validation Set ---")
		print("\nMarket Baseline (Implied Probabilities):")
		val_baseline = evaluate_implied_baseline(data_val)
		print(f"  Accuracy: {val_baseline['accuracy']:.4f}, Brier: {val_baseline['brier']:.4f}, LogLoss: {val_baseline['log_loss']:.4f}")
		mlflow.log_metrics({f"val_baseline_{k}": v for k, v in val_baseline.items()})
		
		print("\nModel Performance:")
		val_metrics = evaluate_model(best_model, data_val, device=DEVICE)
		mlflow.log_metrics(
			{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
		)

		# Evaluate on test set
		print("\n--- Test Set (Current Season) ---")
		data_test = prepare_data(df, base_feats, [test_season], scaler=data_train["scaler"])
		print("\nMarket Baseline (Implied Probabilities):")
		test_baseline = evaluate_implied_baseline(data_test)
		print(f"  Accuracy: {test_baseline['accuracy']:.4f}, Brier: {test_baseline['brier']:.4f}, LogLoss: {test_baseline['log_loss']:.4f}")
		mlflow.log_metrics({f"test_baseline_{k}": v for k, v in test_baseline.items()})
		
		print("\nModel Performance:")
		test_metrics = evaluate_model(best_model, data_test, device=DEVICE)
		mlflow.log_metrics(
			{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
		)

		# Save model
		model_path = MODELS_DIR / "over_under.pt"
		torch.save(best_model.state_dict(), model_path)
		mlflow.log_artifact(str(model_path))

		# Save scaler
		scaler_path = MODELS_DIR / "scaler.joblib"
		joblib.dump(data_train["scaler"], scaler_path)
		mlflow.log_artifact(str(scaler_path))

		# Save metadata
		meta = {
			"architecture": {
				"hidden_layers": best_hidden_layers,
				"dropout": best_arch_params["dropout"],
				"norm": best_arch_params["norm"],
			},
			"training": {
				"lr": best_arch_params["lr"],
				"weight_decay": best_arch_params["weight_decay"],
				"lambda_repulsion": 0.0,
				"lambda_corr": 0.0,
			},
			"features": base_feats,
			"train_seasons": train_seasons,
			"val_season": val_season,
			"test_season": test_season,
			"val_metrics": val_metrics,
			"test_metrics": test_metrics,
		}
		meta_path = MODELS_DIR / "over_under_metadata.json"
		meta_path.write_text(json.dumps(meta, indent=2))
		mlflow.log_artifact(str(meta_path))

	# ========================================================================
	# PHASE 2: Lambda Tuning (maximize Sharpe)
	# ========================================================================
	print("\n" + "=" * 60)
	print("PHASE 2: LAMBDA TUNING (maximize Sharpe ratio)")
	print("=" * 60)

	# End previous run before lambda tuning to avoid conflicts
	objective_lambda, _, _ = create_lambda_objective(
		df, base_feats, train_seasons, val_season, best_arch_params
	)

	study_lambda = optuna.create_study(
		direction="minimize",  # We return negative Sharpe
		study_name="lambda_tuning",
	)

	# Log each lambda trial as a separate MLflow run
	def log_trial_callback(study, trial):
		if trial.state == optuna.trial.TrialState.COMPLETE:
			with mlflow.start_run(run_name=f"lambda_trial_{trial.number}", nested=True):
				mlflow.log_params(trial.params)
				mlflow.log_param("trial_number", trial.number)
				mlflow.log_metric("neg_sharpe_ratio", trial.value)
				# The objective function already logged other metrics

	with mlflow.start_run(run_name="lambda_tuning_study"):
		mlflow.log_param("phase", "lambda_tuning")
		mlflow.log_param("n_trials", N_TRIALS_LAMBDA)
		mlflow.log_params(best_arch_params)

		study_lambda.optimize(
			objective_lambda,
			n_trials=N_TRIALS_LAMBDA,
			show_progress_bar=True,
			callbacks=[log_trial_callback],
		)

		best_lambda_repulsion = study_lambda.best_params["lambda_repulsion"]
		best_lambda_corr = study_lambda.best_params["lambda_corr"]
		best_sharpe = -study_lambda.best_value
		print(f"\nBest lambda_repulsion: {best_lambda_repulsion:.4f}")
		print(f"Best lambda_corr: {best_lambda_corr:.4f}")
		print(f"Best Sharpe ratio: {best_sharpe:.4f}")

		mlflow.log_param("best_lambda_repulsion", best_lambda_repulsion)
		mlflow.log_metric("best_sharpe_ratio", best_sharpe)

	# ========================================================================
	# Train final model with best lambdas (if lambda > 0)
	# ========================================================================
	if best_lambda_repulsion != 0.0 or best_lambda_corr != 0.0:
		print("\n" + "=" * 60)
		print(
			f"TRAINING FINAL MODEL WITH LAMBDA_REPULSION={best_lambda_repulsion:.4f} "
			f"AND LAMBDA_CORR={best_lambda_corr:.4f}"
		)
		print("=" * 60)

		with mlflow.start_run(run_name="best_model_with_lambda"):
			mlflow.log_params(best_arch_params)
			mlflow.log_param("lambda_repulsion", best_lambda_repulsion)
			mlflow.log_param("lambda_corr", best_lambda_corr)

			config = TrainConfig(
				input_dim=data_train["X"].shape[1],
				hidden_layers=best_hidden_layers,
				dropout=best_arch_params["dropout"],
				norm=best_arch_params["norm"],
				lr=best_arch_params["lr"],
				weight_decay=best_arch_params["weight_decay"],
				lambda_repulsion=best_lambda_repulsion,
				lambda_corr=best_lambda_corr,
			)

			best_model_lambda, history, _ = train_model(
				config, train_loader, val_loader, device=DEVICE, verbose=True
			)

			# Evaluate
			print("\n--- Validation Set ---")
			val_metrics = evaluate_model(best_model_lambda, data_val, device=DEVICE)
			mlflow.log_metrics(
				{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
			)

			print("\n--- Test Set ---")
			data_test = prepare_data(df, base_feats, [test_season], scaler=data_train["scaler"])
			test_metrics = evaluate_model(best_model_lambda, data_test, device=DEVICE)
			mlflow.log_metrics(
				{f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
			)

			# Save
			model_path = MODELS_DIR / "over_under_decorrelated.pt"
			torch.save(best_model_lambda.state_dict(), model_path)
			mlflow.log_artifact(str(model_path))

if __name__ == "__main__":
	run_pipeline()
