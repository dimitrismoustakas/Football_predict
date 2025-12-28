"""
Betting Model Lambda Tuning for Over/Under Neural Network

Optimizes lambda_repulsion and lambda_corr parameters to maximize Sharpe ratio,
starting from the base architecture trained in architecture_search.py.

Pipeline:
- Phase 1: Lambda search with rolling CV and Hyperband pruning (~50 trials)
- Phase 2: Multi-seed evaluation on top configs
- Phase 3: Final training and comparison against baseline (lambda=0)

To view MLflow results:
	cd to project root, then run: mlflow ui
	Open http://127.0.0.1:5000 in your browser
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import numpy as np
import optuna
import polars as pl
import torch

from training.evaluation import evaluate_model
from training.models import TrainConfig
from training.train_utils import (
	add_targets_and_implied,
	filter_min_history,
	fold_data_to_loaders,
	generate_rolling_cv_folds,
	get_val_data_dict,
	load_frame,
	precompute_fold_data,
	prepare_data,
	to_loader,
	train_model,
)

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

# Lambda search settings
N_TRIALS = 50
MAX_EPOCHS = 80
PATIENCE = 20

# Multi-seed evaluation
TOP_K_CONFIGS = 5
SEEDS_PER_CONFIG = 5

# Rolling CV settings
N_CV_FOLDS = 3

# Hyperband pruner settings
PRUNER_MIN_RESOURCE = 1  # Minimum folds before pruning
PRUNER_REDUCTION_FACTOR = 3

# Paths
DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# UTILITIES
# ============================================================================


def set_seed(seed: int = 42, deterministic: bool = False):
	"""Set random seeds for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	
	if deterministic:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True


def print_header(text: str):
	"""Print a formatted header."""
	print("\n" + "=" * 60)
	print(text)
	print("=" * 60)


def load_base_config() -> Dict[str, Any]:
	"""Load the base architecture configuration from architecture_search.py output."""
	config_path = MODELS_DIR / "architecture_config.json"
	if not config_path.exists():
		raise FileNotFoundError(
			f"Base config not found at {config_path}. "
			"Run architecture_search.py first to train the base model."
		)
	
	with open(config_path) as f:
		config = json.load(f)
	
	print(f"Loaded base config from {config_path}")
	print(f"  Architecture: {config['hidden_layers']}")
	print(f"  Activation: {config['activation']}, Norm: {config['norm']}")
	print(f"  LR: {config['lr']:.6f}, WD: {config['weight_decay']:.6f}")
	
	return config


def load_base_scaler():
	"""Load the scaler from architecture_search.py output."""
	scaler_path = MODELS_DIR / "scaler_arch_tuned.joblib"
	if not scaler_path.exists():
		raise FileNotFoundError(
			f"Base scaler not found at {scaler_path}. "
			"Run architecture_search.py first."
		)
	
	return joblib.load(scaler_path)


# ============================================================================
# LAMBDA SEARCH OBJECTIVE
# ============================================================================


def create_lambda_objective(
	fold_data: List[Dict[str, Any]],
	base_config: Dict[str, Any],
):
	"""
	Create objective function for lambda parameter search.
	
	Searches lambda_repulsion and lambda_corr while keeping
	architecture fixed from base_config.
	
	Objective: negative Sharpe ratio (minimize to maximize Sharpe).
	"""
	input_dim = base_config["input_dim"]
	hidden_layers = base_config["hidden_layers"]
	activation = base_config["activation"]
	norm = base_config["norm"]
	dropout = base_config["dropout"]
	lr = base_config["lr"]
	weight_decay = base_config["weight_decay"]
	scheduler_type = base_config["scheduler_type"]
	batch_size = base_config["batch_size"]
	
	def objective(trial: optuna.Trial) -> float:
		# === Lambda hyperparameters (include 0 in search space) ===
		lambda_repulsion = trial.suggest_float("lambda_repulsion", 0.0, 0.5)
		lambda_corr = trial.suggest_float("lambda_corr", 0.0, 0.01)
		
		# === Train across all CV folds ===
		fold_sharpes = []
		
		for fold_idx, fold in enumerate(fold_data):
			train_loader, val_loader = fold_data_to_loaders(fold, batch_size)
			
			config = TrainConfig(
				input_dim=input_dim,
				hidden_layers=hidden_layers,
				dropout=dropout,
				norm=norm,
				lr=lr,
				weight_decay=weight_decay,
				lambda_repulsion=lambda_repulsion,
				lambda_corr=lambda_corr,
				activation=activation,
				scheduler_type=scheduler_type,
				epochs=MAX_EPOCHS,
				patience=PATIENCE,
				batch_size=batch_size,
			)
			
			# Train model
			model, _, _ = train_model(
				config, train_loader, val_loader, device=DEVICE, trial=None, verbose=False
			)
			
			# Evaluate on validation fold
			val_data = get_val_data_dict(fold)
			metrics = evaluate_model(model, val_data, device=DEVICE, verbose=False)
			fold_sharpes.append(metrics["sharpe_ratio"])
			
			# Report running mean for Hyperband pruning (fold-level)
			running_mean_sharpe = np.mean(fold_sharpes)
			trial.report(-running_mean_sharpe, fold_idx)  # Negative because we minimize
			if trial.should_prune():
				raise optuna.TrialPruned()
		
		mean_sharpe = np.mean(fold_sharpes)
		
		# Store fold results for analysis
		trial.set_user_attr("fold_sharpes", fold_sharpes)
		trial.set_user_attr("mean_sharpe", mean_sharpe)
		
		# Return negative Sharpe (Optuna minimizes)
		return -mean_sharpe
	
	return objective


# ============================================================================
# PHASE 1: LAMBDA SEARCH
# ============================================================================


def run_lambda_search(
	fold_data: List[Dict[str, Any]],
	base_config: Dict[str, Any],
) -> optuna.Study:
	"""
	Phase 1: Search over lambda parameters with Hyperband pruning.
	"""
	print_header("PHASE 1: LAMBDA SEARCH")
	print(f"Trials: {N_TRIALS}")
	print(f"Epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
	print(f"CV Folds: {len(fold_data)}")
	print(f"Search space: lambda_repulsion=[0, 0.5], lambda_corr=[0, 0.01]")
	
	with mlflow.start_run(run_name="phase1_lambda_search"):
		mlflow.log_params({
			"phase": "1_lambda_search",
			"n_trials": N_TRIALS,
			"max_epochs": MAX_EPOCHS,
			"n_folds": len(fold_data),
			"base_architecture": str(base_config["hidden_layers"]),
		})
		
		objective = create_lambda_objective(fold_data, base_config)
		
		pruner = optuna.pruners.HyperbandPruner(
			min_resource=PRUNER_MIN_RESOURCE,
			max_resource=N_CV_FOLDS,
			reduction_factor=PRUNER_REDUCTION_FACTOR,
		)
		
		study = optuna.create_study(
			direction="minimize",
			pruner=pruner,
			study_name="lambda_search",
		)
		
		study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
		
		best_sharpe = -study.best_value
		print(f"\nBest params: {study.best_params}")
		print(f"Best mean Sharpe ratio: {best_sharpe:.4f}")
		
		mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
		mlflow.log_metric("best_sharpe_ratio", best_sharpe)
	
	return study


# ============================================================================
# PHASE 2: MULTI-SEED EVALUATION
# ============================================================================


def retrain_with_seeds(
	base_config: Dict[str, Any],
	lambda_repulsion: float,
	lambda_corr: float,
	fold_data: List[Dict[str, Any]],
	seeds: List[int],
) -> Tuple[float, float, List[float]]:
	"""
	Retrain config with multiple seeds across all CV folds.
	
	Returns (mean_sharpe, std_sharpe, all_sharpes).
	"""
	all_sharpes = []
	
	for seed in seeds:
		set_seed(seed, deterministic=False)
		
		fold_sharpes = []
		for fold in fold_data:
			train_loader, val_loader = fold_data_to_loaders(fold, base_config["batch_size"])
			
			config = TrainConfig(
				input_dim=base_config["input_dim"],
				hidden_layers=base_config["hidden_layers"],
				dropout=base_config["dropout"],
				norm=base_config["norm"],
				lr=base_config["lr"],
				weight_decay=base_config["weight_decay"],
				lambda_repulsion=lambda_repulsion,
				lambda_corr=lambda_corr,
				activation=base_config["activation"],
				scheduler_type=base_config["scheduler_type"],
				epochs=MAX_EPOCHS,
				patience=PATIENCE,
				batch_size=base_config["batch_size"],
			)
			
			model, _, _ = train_model(
				config, train_loader, val_loader, device=DEVICE, trial=None, verbose=False
			)
			
			val_data = get_val_data_dict(fold)
			metrics = evaluate_model(model, val_data, device=DEVICE, verbose=False)
			fold_sharpes.append(metrics["sharpe_ratio"])
		
		all_sharpes.append(np.mean(fold_sharpes))
	
	return np.mean(all_sharpes), np.std(all_sharpes), all_sharpes


def run_multi_seed_evaluation(
	study: optuna.Study,
	fold_data: List[Dict[str, Any]],
	base_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], float, float]:
	"""
	Phase 2: Multi-seed evaluation of top configs.
	
	Returns (best_params, mean_sharpe, std_sharpe).
	"""
	print_header(f"PHASE 2: MULTI-SEED EVALUATION (Top {TOP_K_CONFIGS}, {SEEDS_PER_CONFIG} seeds)")
	
	# Get top K completed trials
	completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:TOP_K_CONFIGS]
	
	seeds = list(range(42, 42 + SEEDS_PER_CONFIG))
	results = []
	
	with mlflow.start_run(run_name="phase2_multi_seed"):
		mlflow.log_params({
			"phase": "2_multi_seed",
			"top_k": TOP_K_CONFIGS,
			"seeds_per_config": SEEDS_PER_CONFIG,
		})
		
		for i, trial in enumerate(top_trials):
			print(f"\nConfig {i+1}/{len(top_trials)} (trial {trial.number}):")
			print(f"  lambda_repulsion={trial.params['lambda_repulsion']:.4f}")
			print(f"  lambda_corr={trial.params['lambda_corr']:.4f}")
			
			mean_sharpe, std_sharpe, all_sharpes = retrain_with_seeds(
				base_config,
				trial.params["lambda_repulsion"],
				trial.params["lambda_corr"],
				fold_data,
				seeds,
			)
			
			print(f"  Sharpe: {mean_sharpe:.4f} ± {std_sharpe:.4f}")
			
			results.append({
				"trial_number": trial.number,
				"params": trial.params,
				"mean_sharpe": mean_sharpe,
				"std_sharpe": std_sharpe,
				"all_sharpes": all_sharpes,
			})
		
		# Select best by mean Sharpe
		best_result = max(results, key=lambda r: r["mean_sharpe"])
		
		print(f"\nBest config (trial {best_result['trial_number']}):")
		print(f"  lambda_repulsion={best_result['params']['lambda_repulsion']:.4f}")
		print(f"  lambda_corr={best_result['params']['lambda_corr']:.4f}")
		print(f"  Mean Sharpe: {best_result['mean_sharpe']:.4f} ± {best_result['std_sharpe']:.4f}")
		
		mlflow.log_metric("best_mean_sharpe", best_result["mean_sharpe"])
		mlflow.log_metric("best_std_sharpe", best_result["std_sharpe"])
		mlflow.log_params({f"best_{k}": str(v) for k, v in best_result["params"].items()})
	
	return best_result["params"], best_result["mean_sharpe"], best_result["std_sharpe"]


# ============================================================================
# PHASE 3: FINAL TRAINING AND COMPARISON
# ============================================================================


def get_current_season(df: pl.DataFrame) -> str:
	"""Get the current (most recent) season for final evaluation."""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	return seasons[-1]


def train_final_model(
	base_config: Dict[str, Any],
	best_lambda_params: Dict[str, float],
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
	current_season: str,
):
	"""
	Phase 3: Train final model on all training data, evaluate on current season.
	
	Compares against baseline (lambda=0) and saves only if Sharpe improves.
	"""
	print_header("PHASE 3: FINAL MODEL TRAINING & COMPARISON")
	
	# Combine all train + val seasons from folds
	all_train_seasons = set()
	for train_seasons, val_season in folds:
		all_train_seasons.update(train_seasons)
		all_train_seasons.add(val_season)
	all_train_seasons = sorted(all_train_seasons)
	
	print(f"Training seasons: {all_train_seasons[0]}..{all_train_seasons[-1]} ({len(all_train_seasons)} total)")
	print(f"Evaluation season: {current_season}")
	
	set_seed(42, deterministic=True)
	
	# Prepare data
	data_train = prepare_data(df, feature_cols, all_train_seasons, fit_scaler=True)
	data_current = prepare_data(df, feature_cols, [current_season], scaler=data_train["scaler"])
	
	train_loader = to_loader(data_train, base_config["batch_size"], shuffle=True, device=DEVICE)
	val_loader = to_loader(data_current, base_config["batch_size"], shuffle=False, device=DEVICE)
	
	with mlflow.start_run(run_name="phase3_final_comparison"):
		# === Train baseline model (lambda=0) ===
		print("\n--- Training baseline model (lambda=0) ---")
		
		baseline_config = TrainConfig(
			input_dim=base_config["input_dim"],
			hidden_layers=base_config["hidden_layers"],
			dropout=base_config["dropout"],
			norm=base_config["norm"],
			lr=base_config["lr"],
			weight_decay=base_config["weight_decay"],
			lambda_repulsion=0.0,
			lambda_corr=0.0,
			activation=base_config["activation"],
			scheduler_type=base_config["scheduler_type"],
			epochs=MAX_EPOCHS,
			patience=PATIENCE,
			batch_size=base_config["batch_size"],
		)
		
		baseline_model, _, _ = train_model(
			baseline_config, train_loader, val_loader, device=DEVICE, verbose=True
		)
		
		print("\nBaseline evaluation on current season:")
		baseline_metrics = evaluate_model(baseline_model, data_current, device=DEVICE, verbose=True)
		
		mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items() if isinstance(v, (int, float))})
		
		# === Train lambda-tuned model ===
		print(f"\n--- Training lambda-tuned model ---")
		print(f"  lambda_repulsion={best_lambda_params['lambda_repulsion']:.4f}")
		print(f"  lambda_corr={best_lambda_params['lambda_corr']:.4f}")
		
		set_seed(42, deterministic=True)  # Reset for fair comparison
		
		# Recreate loaders (consumed by baseline training)
		train_loader = to_loader(data_train, base_config["batch_size"], shuffle=True, device=DEVICE)
		val_loader = to_loader(data_current, base_config["batch_size"], shuffle=False, device=DEVICE)
		
		tuned_config = TrainConfig(
			input_dim=base_config["input_dim"],
			hidden_layers=base_config["hidden_layers"],
			dropout=base_config["dropout"],
			norm=base_config["norm"],
			lr=base_config["lr"],
			weight_decay=base_config["weight_decay"],
			lambda_repulsion=best_lambda_params["lambda_repulsion"],
			lambda_corr=best_lambda_params["lambda_corr"],
			activation=base_config["activation"],
			scheduler_type=base_config["scheduler_type"],
			epochs=MAX_EPOCHS,
			patience=PATIENCE,
			batch_size=base_config["batch_size"],
		)
		
		tuned_model, _, _ = train_model(
			tuned_config, train_loader, val_loader, device=DEVICE, verbose=True
		)
		
		print("\nTuned model evaluation on current season:")
		tuned_metrics = evaluate_model(tuned_model, data_current, device=DEVICE, verbose=True)
		
		mlflow.log_metrics({f"tuned_{k}": v for k, v in tuned_metrics.items() if isinstance(v, (int, float))})
		
		# === Compare and decide ===
		baseline_sharpe = baseline_metrics["sharpe_ratio"]
		tuned_sharpe = tuned_metrics["sharpe_ratio"]
		
		print(f"\n{'='*40}")
		print(f"COMPARISON (Current Season Sharpe Ratio):")
		print(f"  Baseline (lambda=0): {baseline_sharpe:.4f}")
		print(f"  Tuned model:         {tuned_sharpe:.4f}")
		print(f"  Improvement:         {tuned_sharpe - baseline_sharpe:+.4f}")
		print(f"{'='*40}")
		
		if tuned_sharpe > baseline_sharpe:
			print("\n✓ Tuned model WINS! Saving...")
			
			model_path = MODELS_DIR / "over_under_sharpe_optimized.pt"
			torch.save(tuned_model.state_dict(), model_path)
			print(f"  Model saved to {model_path}")
			
			# Save updated config with lambda values
			config_to_save = {
				**base_config,
				"lambda_repulsion": best_lambda_params["lambda_repulsion"],
				"lambda_corr": best_lambda_params["lambda_corr"],
				"current_season_sharpe": tuned_sharpe,
				"baseline_sharpe": baseline_sharpe,
			}
			
			config_path = MODELS_DIR / "sharpe_optimized_config.json"
			with open(config_path, "w") as f:
				json.dump(config_to_save, f, indent=2)
			print(f"  Config saved to {config_path}")
			
			# Save scaler
			scaler_path = MODELS_DIR / "scaler_sharpe_optimized.joblib"
			joblib.dump(data_train["scaler"], scaler_path)
			print(f"  Scaler saved to {scaler_path}")
			
			mlflow.log_artifact(str(model_path))
			mlflow.log_artifact(str(config_path))
			mlflow.log_artifact(str(scaler_path))
			
			mlflow.log_metric("improvement", tuned_sharpe - baseline_sharpe)
			mlflow.log_param("winner", "tuned")
			
			return tuned_model, tuned_metrics
		else:
			print("\n✗ Baseline wins or tie. Not saving tuned model.")
			mlflow.log_metric("improvement", tuned_sharpe - baseline_sharpe)
			mlflow.log_param("winner", "baseline")
			
			return baseline_model, baseline_metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
	"""Main entry point for betting model lambda search pipeline."""
	print_header("BETTING MODEL LAMBDA SEARCH PIPELINE")
	
	print(f"Device: {DEVICE}")
	if DEVICE.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	
	set_seed(42, deterministic=False)
	
	# === Load base configuration ===
	print("\nLoading base architecture configuration...")
	base_config = load_base_config()
	feature_cols = base_config["feature_cols"]
	
	# === Load and prepare data ===
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	
	# Filter to rows with valid odds
	df = df.drop_nulls(subset=["odds_over", "odds_under"])
	print(f"Total rows with odds: {len(df)}")
	
	# === Generate CV folds ===
	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	current_season = get_current_season(df)
	print(f"Current season (final evaluation): {current_season}")
	
	# === Precompute scaled data for all folds ===
	print("\nPrecomputing scaled data for CV folds...")
	fold_data = precompute_fold_data(df, feature_cols, folds)
	
	# Verify input dimension matches config
	actual_input_dim = fold_data[0]["X_train"].shape[1]
	if actual_input_dim != base_config["input_dim"]:
		raise ValueError(
			f"Input dimension mismatch: data has {actual_input_dim} features, "
			f"but config expects {base_config['input_dim']}. "
			"Re-run architecture_search.py with current features."
		)
	
	# === Set up MLflow ===
	mlflow.set_experiment("betting_lambda_search")
	
	# === Phase 1: Lambda search ===
	study = run_lambda_search(fold_data, base_config)
	
	# === Phase 2: Multi-seed evaluation ===
	best_lambda_params, mean_sharpe, std_sharpe = run_multi_seed_evaluation(
		study, fold_data, base_config
	)
	
	# === Phase 3: Final training and comparison ===
	final_model, final_metrics = train_final_model(
		base_config, best_lambda_params, df, feature_cols, folds, current_season
	)
	
	print_header("PIPELINE COMPLETE")
	print(f"\nBest lambda parameters:")
	print(f"  lambda_repulsion: {best_lambda_params['lambda_repulsion']:.4f}")
	print(f"  lambda_corr: {best_lambda_params['lambda_corr']:.4f}")
	print(f"\nFinal Sharpe ratio: {final_metrics['sharpe_ratio']:.4f}")
	print(f"Artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
	main()
