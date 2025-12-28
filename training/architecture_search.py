"""
Joint Architecture & Optimizer Tuning for Over/Under Neural Network

Two-phase hyperparameter search with rolling cross-validation:
- Phase 1 (Coarse): Joint search over architecture + training params with aggressive pruning
- Phase 2 (Refine): Narrowed search around top regions with longer training
- Phase 3: Multi-seed evaluation on final shortlist
- Phase 4: Final training on train+val, evaluate once on held-out test

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

# Add project root to path
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
	build_hidden_layers,
	filter_min_history,
	fold_data_to_loaders,
	generate_rolling_cv_folds,
	get_test_season,
	load_existing_model,
	load_frame,
	precompute_fold_data,
	prepare_data,
	select_feature_columns,
	to_loader,
	train_model,
)

# ============================================================================
# SEARCH CONFIGURATION - Tune these as needed
# ============================================================================

# Phase 1: Coarse joint search (wide ranges, cheap training)
COARSE_TRIALS = 120
COARSE_EPOCHS = 40
COARSE_PATIENCE = 12

# Phase 2: Refinement search (narrowed ranges, longer training)
REFINE_TRIALS = 60
REFINE_EPOCHS = 80
REFINE_PATIENCE = 20

# Phase 3: Multi-seed evaluation
TOP_K_CONFIGS = 8
SEEDS_PER_CONFIG = 5

# Rolling CV settings
N_CV_FOLDS = 3

# Hyperband pruner settings (ASHA)
# - min_resource: minimum epochs before a trial can be pruned
# - max_resource: aligned with training epochs for each phase
# - reduction_factor: keep top 1/n at each rung (3 = keep top 33%)
PRUNER_MIN_RESOURCE = 5
PRUNER_REDUCTION_FACTOR = 3

# Data paths
DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
# os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"

# Device configuration - set once at module level
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# UTILITIES
# ============================================================================


def set_seed(seed: int = 42, deterministic: bool = False):
	"""
	Set random seeds for reproducibility.
	
	Args:
		seed: Random seed value
		deterministic: If True, use deterministic cuDNN (slower but reproducible).
			Set False during search for speed, True for final training.
	"""
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


# ============================================================================
# JOINT SEARCH OBJECTIVE
# ============================================================================


def create_joint_objective(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
	max_epochs: int,
	patience: int,
	lr_range: Tuple[float, float] = (1e-5, 1e-2),
	wd_range: Tuple[float, float] = (1e-6, 1e-2),
	allowed_activations: List[str] = None,
	allowed_norms: List[str] = None,
	allowed_shapes: List[str] = None,
	allowed_base_widths: List[int] = None,
):
	"""
	Create objective function for joint architecture + optimizer search.
	
	Searches:
	- Architecture: base_width, n_layers, shape, activation, norm
	- Training: lr, weight_decay, dropout, batch_size, scheduler_type
	
	Objective is mean validation loss across all CV folds.
	"""
	# Defaults for search space (can be narrowed in refinement phase)
	if allowed_activations is None:
		allowed_activations = ["relu", "silu", "gelu", "geglu"]
	if allowed_norms is None:
		allowed_norms = ["none", "ln"]
	if allowed_shapes is None:
		allowed_shapes = ["constant", "pyramid", "inverted", "diamond"]
	if allowed_base_widths is None:
		allowed_base_widths = [128, 256, 512]
	
	def objective(trial: optuna.Trial) -> float:
		# === Architecture hyperparameters ===
		base_width = trial.suggest_categorical("base_width", allowed_base_widths)
		n_layers = trial.suggest_int("n_layers", 2, 5)
		shape = trial.suggest_categorical("shape", allowed_shapes)
		activation = trial.suggest_categorical("activation", allowed_activations)
		norm = trial.suggest_categorical("norm", allowed_norms)
		
		hidden_layers = build_hidden_layers(base_width, n_layers, shape)
		
		# === Training hyperparameters ===
		lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
		weight_decay = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)
		dropout = trial.suggest_float("dropout", 0.05, 0.5)
		batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
		scheduler_type = trial.suggest_categorical("scheduler_type", ["plateau", "cosine", "onecycle"])
		
		# === Train across all CV folds ===
		# Note: We don't pass `trial` to train_model to avoid duplicate step reports
		# when the same epoch is trained across multiple folds. Instead, we report
		# the mean loss after each fold completes, using fold_idx as the step.
		fold_losses = []
		
		for fold_idx, fold in enumerate(fold_data):
			train_loader, val_loader = fold_data_to_loaders(fold, batch_size)
			
			config = TrainConfig(
				input_dim=input_dim,
				hidden_layers=hidden_layers,
				dropout=dropout,
				norm=norm,
				lr=lr,
				weight_decay=weight_decay,
				lambda_repulsion=0.0,
				lambda_corr=0.0,
				activation=activation,
				scheduler_type=scheduler_type,
				epochs=max_epochs,
				patience=patience,
				batch_size=batch_size,
			)
			
			try:
				# Don't pass trial here - we'll report after fold completes
				_, _, best_val_loss = train_model(
					config, train_loader, val_loader, device=DEVICE, trial=None, verbose=False
				)
				fold_losses.append(best_val_loss)
				
				# Report running mean after each fold for intermediate pruning
				running_mean = np.mean(fold_losses)
				trial.report(running_mean, fold_idx)
				if trial.should_prune():
					raise optuna.TrialPruned()
					
			except optuna.TrialPruned:
				raise  # Re-raise pruning
			except Exception as e:
				print(f"  Fold {fold_idx} failed: {e}")
				return float("inf")
		
		mean_loss = np.mean(fold_losses)
		
		# Store fold losses for analysis
		trial.set_user_attr("fold_losses", fold_losses)
		trial.set_user_attr("hidden_layers", hidden_layers)
		
		return mean_loss
	
	return objective


# ============================================================================
# PHASE 1: COARSE JOINT SEARCH
# ============================================================================


def run_coarse_search(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
) -> optuna.Study:
	"""
	Phase 1: Coarse joint search over full parameter space.
	
	Uses aggressive Hyperband pruning to efficiently explore.
	"""
	print_header("PHASE 1: COARSE JOINT SEARCH")
	print(f"Trials: {COARSE_TRIALS}")
	print(f"Epochs: {COARSE_EPOCHS}, Patience: {COARSE_PATIENCE}")
	print(f"CV Folds: {len(fold_data)}")
	
	with mlflow.start_run(run_name="phase1_coarse_search"):
		mlflow.log_params({
			"phase": "1_coarse",
			"n_trials": COARSE_TRIALS,
			"max_epochs": COARSE_EPOCHS,
			"n_folds": len(fold_data),
			"pruner": "hyperband",
			"pruner_min_resource": PRUNER_MIN_RESOURCE,
			"pruner_reduction_factor": PRUNER_REDUCTION_FACTOR,
		})
		
		objective = create_joint_objective(
			fold_data=fold_data,
			input_dim=input_dim,
			max_epochs=COARSE_EPOCHS,
			patience=COARSE_PATIENCE,
		)
		
		pruner = optuna.pruners.HyperbandPruner(
			min_resource=PRUNER_MIN_RESOURCE,
			max_resource=COARSE_EPOCHS,
			reduction_factor=PRUNER_REDUCTION_FACTOR,
		)
		
		study = optuna.create_study(
			direction="minimize",
			pruner=pruner,
			study_name="phase1_coarse",
		)
		
		study.optimize(objective, n_trials=COARSE_TRIALS, show_progress_bar=True)
		
		print(f"\nBest params: {study.best_params}")
		print(f"Best mean val loss: {study.best_value:.5f}")
		
		mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
		mlflow.log_metric("best_val_loss", study.best_value)
	
	return study


# ============================================================================
# PHASE 2: REFINEMENT SEARCH
# ============================================================================


def extract_refinement_ranges(study: optuna.Study, top_n: int = 20) -> Dict[str, Any]:
	"""
	Extract narrowed search ranges from top trials.
	
	For continuous params (lr, weight_decay): use quartile bounds
	For categorical params: restrict to values appearing in top trials
	"""
	# Get top N completed trials
	completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:top_n]
	
	if len(top_trials) < 5:
		print(f"Warning: Only {len(top_trials)} completed trials, using all for refinement")
	
	# Extract values from top trials
	lrs = [t.params["lr"] for t in top_trials]
	wds = [t.params["weight_decay"] for t in top_trials]
	activations = [t.params["activation"] for t in top_trials]
	norms = [t.params["norm"] for t in top_trials]
	shapes = [t.params["shape"] for t in top_trials]
	base_widths = [t.params["base_width"] for t in top_trials]
	
	# Compute narrowed ranges
	lr_range = (np.percentile(lrs, 10), np.percentile(lrs, 90))
	wd_range = (np.percentile(wds, 10), np.percentile(wds, 90))
	
	# For categoricals: keep values appearing in >20% of top trials
	threshold = max(1, len(top_trials) // 5)
	
	def filter_categorical(values):
		from collections import Counter
		counts = Counter(values)
		return [v for v, c in counts.items() if c >= threshold]
	
	allowed_activations = filter_categorical(activations) or list(set(activations))
	allowed_norms = filter_categorical(norms) or list(set(norms))
	allowed_shapes = filter_categorical(shapes) or list(set(shapes))
	allowed_base_widths = filter_categorical(base_widths) or list(set(base_widths))
	
	return {
		"lr_range": lr_range,
		"wd_range": wd_range,
		"allowed_activations": allowed_activations,
		"allowed_norms": allowed_norms,
		"allowed_shapes": allowed_shapes,
		"allowed_base_widths": allowed_base_widths,
	}


def run_refinement_search(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
	coarse_study: optuna.Study,
) -> optuna.Study:
	"""
	Phase 2: Refinement search around top regions from Phase 1.
	
	Narrowed search space, longer training budget.
	"""
	print_header("PHASE 2: REFINEMENT SEARCH")
	
	# Extract narrowed ranges from coarse search
	ranges = extract_refinement_ranges(coarse_study, top_n=20)
	
	print(f"Trials: {REFINE_TRIALS}")
	print(f"Epochs: {REFINE_EPOCHS}, Patience: {REFINE_PATIENCE}")
	print(f"LR range: [{ranges['lr_range'][0]:.2e}, {ranges['lr_range'][1]:.2e}]")
	print(f"WD range: [{ranges['wd_range'][0]:.2e}, {ranges['wd_range'][1]:.2e}]")
	print(f"Activations: {ranges['allowed_activations']}")
	print(f"Norms: {ranges['allowed_norms']}")
	print(f"Shapes: {ranges['allowed_shapes']}")
	print(f"Base widths: {ranges['allowed_base_widths']}")
	
	with mlflow.start_run(run_name="phase2_refinement"):
		mlflow.log_params({
			"phase": "2_refine",
			"n_trials": REFINE_TRIALS,
			"max_epochs": REFINE_EPOCHS,
			"lr_low": ranges["lr_range"][0],
			"lr_high": ranges["lr_range"][1],
		})
		
		objective = create_joint_objective(
			fold_data=fold_data,
			input_dim=input_dim,
			max_epochs=REFINE_EPOCHS,
			patience=REFINE_PATIENCE,
			lr_range=ranges["lr_range"],
			wd_range=ranges["wd_range"],
			allowed_activations=ranges["allowed_activations"],
			allowed_norms=ranges["allowed_norms"],
			allowed_shapes=ranges["allowed_shapes"],
			allowed_base_widths=ranges["allowed_base_widths"],
		)
		
		pruner = optuna.pruners.HyperbandPruner(
			min_resource=PRUNER_MIN_RESOURCE,
			max_resource=REFINE_EPOCHS,
			reduction_factor=PRUNER_REDUCTION_FACTOR,
		)
		
		study = optuna.create_study(
			direction="minimize",
			pruner=pruner,
			study_name="phase2_refine",
		)
		
		study.optimize(objective, n_trials=REFINE_TRIALS, show_progress_bar=True)
		
		print(f"\nBest params: {study.best_params}")
		print(f"Best mean val loss: {study.best_value:.5f}")
		
		mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
		mlflow.log_metric("best_val_loss", study.best_value)
	
	return study


# ============================================================================
# PHASE 3: MULTI-SEED EVALUATION
# ============================================================================


def extract_config_from_params(params: Dict, input_dim: int) -> TrainConfig:
	"""Extract TrainConfig from trial params."""
	hidden_layers = build_hidden_layers(
		params["base_width"],
		params["n_layers"],
		params["shape"],
	)
	
	return TrainConfig(
		input_dim=input_dim,
		hidden_layers=hidden_layers,
		dropout=params["dropout"],
		norm=params["norm"],
		lr=params["lr"],
		weight_decay=params["weight_decay"],
		lambda_repulsion=0.0,
		lambda_corr=0.0,
		activation=params["activation"],
		scheduler_type=params["scheduler_type"],
		epochs=REFINE_EPOCHS,
		patience=REFINE_PATIENCE,
		batch_size=params["batch_size"],
	)


def retrain_with_seeds(
	config: TrainConfig,
	fold_data: List[Dict[str, Any]],
	seeds: List[int],
) -> Tuple[float, float, List[float], List[int]]:
	"""
	Retrain config with multiple seeds across all CV folds.
	
	Returns (mean_loss, std_loss, all_losses, stopping_epochs).
	"""
	all_losses = []
	stopping_epochs = []
	
	for seed in seeds:
		set_seed(seed, deterministic=False)  # Speed during multi-seed eval
		
		fold_losses = []
		for fold in fold_data:
			train_loader, val_loader = fold_data_to_loaders(fold, config.batch_size)
			
			_, history, best_val_loss = train_model(
				config, train_loader, val_loader, device=DEVICE, verbose=False
			)
			fold_losses.append(best_val_loss)
			stopping_epochs.append(len(history["val_loss"]))
		
		all_losses.append(np.mean(fold_losses))
	
	return np.mean(all_losses), np.std(all_losses), all_losses, stopping_epochs


def run_multi_seed_evaluation(
	refine_study: optuna.Study,
	fold_data: List[Dict[str, Any]],
	input_dim: int,
) -> Tuple[TrainConfig, Dict, int]:
	"""
	Phase 3: Multi-seed evaluation of top configs.
	
	Returns best config, result dict, and median stopping epoch.
	"""
	print_header(f"PHASE 3: MULTI-SEED EVALUATION (Top {TOP_K_CONFIGS}, {SEEDS_PER_CONFIG} seeds)")
	
	# Get top K completed trials
	completed = [t for t in refine_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:TOP_K_CONFIGS]
	
	seeds = list(range(42, 42 + SEEDS_PER_CONFIG))
	results = []
	all_stopping_epochs = []
	
	with mlflow.start_run(run_name="phase3_multi_seed"):
		mlflow.log_params({
			"phase": "3_multi_seed",
			"top_k": TOP_K_CONFIGS,
			"seeds_per_config": SEEDS_PER_CONFIG,
		})
		
		for i, trial in enumerate(top_trials):
			config = extract_config_from_params(trial.params, input_dim)
			print(f"\nConfig {i+1}/{TOP_K_CONFIGS}: {config.hidden_layers} | {config.activation} | {config.norm}")
			
			mean_loss, std_loss, losses, stopping_epochs = retrain_with_seeds(
				config, fold_data, seeds
			)
			all_stopping_epochs.extend(stopping_epochs)
			
			print(f"  Mean val loss: {mean_loss:.5f} ± {std_loss:.5f}")
			
			results.append({
				"trial_number": trial.number,
				"config": config,
				"params": trial.params,
				"mean_loss": mean_loss,
				"std_loss": std_loss,
				"all_losses": losses,
			})
		
		# Select best by mean loss
		best_result = min(results, key=lambda r: r["mean_loss"])
		median_epochs = int(np.median(all_stopping_epochs))
		
		print(f"\nBest config (trial {best_result['trial_number']}):")
		print(f"  Architecture: {best_result['config'].hidden_layers}")
		print(f"  Activation: {best_result['config'].activation}, Norm: {best_result['config'].norm}")
		print(f"  Mean val loss: {best_result['mean_loss']:.5f} ± {best_result['std_loss']:.5f}")
		print(f"  Median stopping epoch: {median_epochs}")
		
		mlflow.log_metric("best_mean_val_loss", best_result["mean_loss"])
		mlflow.log_metric("best_std_val_loss", best_result["std_loss"])
		mlflow.log_metric("median_stopping_epoch", median_epochs)
		mlflow.log_params({f"best_{k}": str(v) for k, v in best_result["params"].items()})
	
	return best_result["config"], best_result, median_epochs


# ============================================================================
# PHASE 4: FINAL MODEL TRAINING
# ============================================================================


def compare_and_save_model(
	new_model: Any,
	new_metrics: Dict,
	new_config: TrainConfig,
	feature_cols: List[str],
	final_epochs: int,
	data_train: Dict,
	data_test: Dict,
) -> bool:
	"""
	Compare new model against existing saved model by evaluating both on test data.
	
	Uses Brier score as comparison metric (lower is better).
	Returns True if new model was saved, False if existing was kept.
	"""
	model_path = MODELS_DIR / "over_under_arch_tuned.pt"
	config_path = MODELS_DIR / "architecture_config.json"
	scaler_path = MODELS_DIR / "scaler_arch_tuned.joblib"
	
	existing_model, existing_config = load_existing_model(config_path, model_path, DEVICE)
	
	new_brier = new_metrics["brier"]
	
	if existing_model is None:
		print("\nNo existing model found. Saving new model.")
		save_new = True
	else:
		# Evaluate existing model on same test data
		print("\nEvaluating existing model on test set...")
		existing_metrics = evaluate_model(existing_model, data_test, device=DEVICE, verbose=False)
		existing_brier = existing_metrics["brier"]
		
		print(f"\n{'='*40}")
		print("MODEL COMPARISON (on same test set)")
		print(f"{'='*40}")
		print(f"Existing model Brier: {existing_brier:.5f}")
		print(f"New model Brier:      {new_brier:.5f}")
		
		if new_brier < existing_brier:
			improvement = (existing_brier - new_brier) / existing_brier * 100
			print(f"New model is BETTER by {improvement:.2f}%")
			save_new = True
		else:
			degradation = (new_brier - existing_brier) / existing_brier * 100
			print(f"Existing model is better by {degradation:.2f}%")
			print("Keeping existing model.")
			save_new = False
	
	if save_new:
		torch.save(new_model.state_dict(), model_path)
		print(f"Model saved to {model_path}")
		
		joblib.dump(data_train["scaler"], scaler_path)
		print(f"Scaler saved to {scaler_path}")
		
		config_dict = {
			"input_dim": new_config.input_dim,
			"hidden_layers": new_config.hidden_layers,
			"activation": new_config.activation,
			"norm": new_config.norm,
			"dropout": new_config.dropout,
			"lr": new_config.lr,
			"weight_decay": new_config.weight_decay,
			"scheduler_type": new_config.scheduler_type,
			"batch_size": new_config.batch_size,
			"lambda_repulsion": new_config.lambda_repulsion,
			"lambda_corr": new_config.lambda_corr,
			"final_epochs": final_epochs,
			"feature_cols": feature_cols,
		}
		with open(config_path, "w") as f:
			json.dump(config_dict, f, indent=2)
		print(f"Config saved to {config_path}")
		
		mlflow.log_artifact(str(model_path))
		mlflow.log_artifact(str(scaler_path))
		mlflow.log_artifact(str(config_path))
	
	return save_new


def train_final_model(
	config: TrainConfig,
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
	test_season: str,
	final_epochs: int,
):
	"""
	Phase 4: Train final model on all training data, evaluate on held-out test.
	
	IMPORTANT: No early stopping on test data to prevent leakage.
	Training runs for fixed `final_epochs` determined from Phase 3.
	"""
	print_header("PHASE 4: FINAL MODEL TRAINING")
	
	# Combine all train + val seasons from folds
	all_train_seasons = set()
	for train_seasons, val_season in folds:
		all_train_seasons.update(train_seasons)
		all_train_seasons.add(val_season)
	all_train_seasons = sorted(all_train_seasons)
	
	print(f"Training seasons: {all_train_seasons[0]}..{all_train_seasons[-1]} ({len(all_train_seasons)} total)")
	print(f"Test season: {test_season}")
	print(f"Training for exactly {final_epochs} epochs (no early stopping on test)")
	
	set_seed(42, deterministic=True)  # Deterministic for final model
	
	with mlflow.start_run(run_name="phase4_final_model"):
		# Prepare data
		data_train = prepare_data(df, feature_cols, all_train_seasons, fit_scaler=True)
		data_test = prepare_data(df, feature_cols, [test_season], scaler=data_train["scaler"])
		
		train_loader = to_loader(data_train, config.batch_size, device=DEVICE)
		
		# Update config for final training
		config.input_dim = data_train["X"].shape[1]
		config.epochs = final_epochs
		config.patience = final_epochs + 1  # Effectively disable early stopping
		
		mlflow.log_params({
			"phase": "4_final",
			"hidden_layers": str(config.hidden_layers),
			"activation": config.activation,
			"norm": config.norm,
			"dropout": config.dropout,
			"lr": config.lr,
			"weight_decay": config.weight_decay,
			"scheduler_type": config.scheduler_type,
			"batch_size": config.batch_size,
			"final_epochs": final_epochs,
			"train_seasons": f"{all_train_seasons[0]}..{all_train_seasons[-1]}",
			"test_season": test_season,
		})
		
		print("\nTraining final model...")
		
		# Create a dummy val loader for training (won't affect stopping)
		# We use a small subset of training data just for monitoring
		dummy_val_loader = to_loader(data_train, config.batch_size, shuffle=False, device=DEVICE)
		
		model, history, _ = train_model(
			config, train_loader, dummy_val_loader, device=DEVICE, verbose=True
		)
		
		# Evaluate on held-out test
		print("\nEvaluating on held-out test set...")
		metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True)
		
		mlflow.log_metrics({
			"test_accuracy": metrics["accuracy"],
			"test_brier": metrics["brier"],
			"test_log_loss": metrics["log_loss"],
			"test_corr": metrics["corr_with_implied"],
		})
		
		# Compare against existing model and save only if better
		model_saved = compare_and_save_model(
			new_model=model,
			new_metrics=metrics,
			new_config=config,
			feature_cols=feature_cols,
			final_epochs=final_epochs,
			data_train=data_train,
			data_test=data_test,
		)
		
		mlflow.log_metric("model_saved", int(model_saved))
	
	return model, data_train["scaler"], metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
	"""Main entry point for joint architecture search pipeline."""
	print_header("JOINT ARCHITECTURE & OPTIMIZER SEARCH PIPELINE")
	
	print(f"Device: {DEVICE}")
	if DEVICE.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	
	set_seed(42, deterministic=False)  # Speed during search
	
	# === Load and prepare data ===
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	
	# Filter to rows with valid odds
	df = df.drop_nulls(subset=["odds_over", "odds_under"])
	print(f"Total rows with odds: {len(df)}")
	
	feature_cols = select_feature_columns(df)
	print(f"Features: {len(feature_cols)} columns")
	
	# === Generate CV folds ===
	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Test season (held out): {test_season}")
	
	# === Precompute scaled data for all folds ===
	print("\nPrecomputing scaled data for CV folds...")
	fold_data = precompute_fold_data(df, feature_cols, folds)
	input_dim = fold_data[0]["X_train"].shape[1]
	print(f"Input dimension: {input_dim}")
	
	# === Set up MLflow ===
	mlflow.set_experiment("joint_architecture_search")
	
	# === Phase 1: Coarse search ===
	coarse_study = run_coarse_search(fold_data, input_dim)
	
	# === Phase 2: Refinement search ===
	refine_study = run_refinement_search(fold_data, input_dim, coarse_study)
	
	# === Phase 3: Multi-seed evaluation ===
	best_config, best_result, median_epochs = run_multi_seed_evaluation(
		refine_study, fold_data, input_dim
	)
	
	# === Phase 4: Final model ===
	model, scaler, test_metrics = train_final_model(
		best_config, df, feature_cols, folds, test_season, median_epochs
	)
	
	print_header("PIPELINE COMPLETE")
	print(f"\nFinal architecture: {best_config.hidden_layers}")
	print(f"Activation: {best_config.activation}, Norm: {best_config.norm}")
	print(f"Test metrics: {test_metrics}")
	print(f"\nArtifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
	main()