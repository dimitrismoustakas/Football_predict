"""
Unified Architecture Search Pipeline for Neural Network Training

Supports both task types:
- binary: Over/Under 2.5 goals prediction
- multiclass: Home/Draw/Away result prediction

To view MLflow results:
	cd to project root, then run: mlflow ui
	Open http://127.0.0.1:5000 in your browser
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import numpy as np
import optuna
import polars as pl
import torch

from training.evaluation import evaluate_model
from training.models import TrainConfig, CategoricalConfig
from training.train_utils import (
	add_targets_and_implied,
	add_targets_and_implied_result,
	build_hidden_layers,
	evaluate_implied_baseline,
	filter_min_history,
	fold_data_to_loaders,
	generate_rolling_cv_folds,
	get_num_leagues,
	get_test_season,
	load_existing_model,
	load_frame,
	precompute_fold_data,
	prepare_data,
	prepare_data_result,
	select_feature_columns,
	to_loader,
	train_model,
)

TaskType = Literal["binary", "multiclass"]

# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================

COARSE_TRIALS = 120
COARSE_EPOCHS = 40
COARSE_PATIENCE = 12

REFINE_TRIALS = 60
REFINE_EPOCHS = 80
REFINE_PATIENCE = 20

TOP_K_CONFIGS = 8
SEEDS_PER_CONFIG = 5

N_CV_FOLDS = 3

PRUNER_MIN_RESOURCE = 5
PRUNER_REDUCTION_FACTOR = 3

DEFAULT_PARQUET = Path("data/training/understat_df.parquet")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# TASK-SPECIFIC CONFIGURATION
# ============================================================================

TASK_CONFIG = {
	"binary": {
		"add_targets_fn": add_targets_and_implied,
		"prepare_fn": prepare_data,
		"odds_cols": ["odds_over", "odds_under"],
		"model_path": "over_under_arch_tuned.pt",
		"config_path": "architecture_config.json",
		"scaler_path": "scaler_arch_tuned.joblib",
		"comparison_metric": "brier",
		"experiment_name": "joint_architecture_search",
		"run_prefix": "",
	},
	"multiclass": {
		"add_targets_fn": add_targets_and_implied_result,
		"prepare_fn": prepare_data_result,
		"odds_cols": ["odds_home", "odds_draw", "odds_away"],
		"model_path": "result_arch_tuned.pt",
		"config_path": "result_architecture_config.json",
		"scaler_path": "result_scaler_arch_tuned.joblib",
		"comparison_metric": "log_loss",
		"experiment_name": "result_architecture_search",
		"run_prefix": "result_",
	},
}


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


# ============================================================================
# JOINT SEARCH OBJECTIVE
# ============================================================================


def create_joint_objective(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
	max_epochs: int,
	patience: int,
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
	lr_range: Tuple[float, float] = (1e-5, 1e-2),
	wd_range: Tuple[float, float] = (1e-6, 1e-2),
	allowed_activations: List[str] = None,
	allowed_norms: List[str] = None,
	allowed_shapes: List[str] = None,
	allowed_base_widths: List[int] = None,
):
	"""Create objective function for joint architecture + optimizer search."""
	if allowed_activations is None:
		allowed_activations = ["relu", "silu", "gelu", "geglu"]
	if allowed_norms is None:
		allowed_norms = ["none", "ln"]
	if allowed_shapes is None:
		allowed_shapes = ["constant", "pyramid", "inverted", "diamond"]
	if allowed_base_widths is None:
		allowed_base_widths = [128, 256, 512]
	
	def objective(trial: optuna.Trial) -> float:
		base_width = trial.suggest_categorical("base_width", allowed_base_widths)
		n_layers = trial.suggest_int("n_layers", 2, 5)
		shape = trial.suggest_categorical("shape", allowed_shapes)
		activation = trial.suggest_categorical("activation", allowed_activations)
		norm = trial.suggest_categorical("norm", allowed_norms)
		
		hidden_layers = build_hidden_layers(base_width, n_layers, shape)
		
		lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
		weight_decay = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)
		dropout = trial.suggest_float("dropout", 0.05, 0.5)
		batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
		scheduler_type = trial.suggest_categorical("scheduler_type", ["plateau", "cosine", "onecycle"])
		
		fold_losses = []
		# Reverse folds so that fold 0 is the most recent validation season.
		# This is intentional: pruning is only applied on fold 0, and we want
		# pruning decisions to be based on the fold most similar to the deployment/test distribution.
		# See ARCHITECTURE_SEARCH_ISSUES.txt for rationale.
		folds_for_objective = list(reversed(fold_data))

		for fold_idx, fold in enumerate(folds_for_objective):
			train_loader, val_loader = fold_data_to_loaders(fold, batch_size, task_type=task_type)
			
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
				task_type=task_type,
				cat_config=cat_config,
			)
			
			try:
				trial_for_fold = trial if fold_idx == 0 else None
				_, _, best_val_loss = train_model(
					config, train_loader, val_loader, device=DEVICE, trial=trial_for_fold, verbose=False
				)
				fold_losses.append(best_val_loss)
					
			except optuna.TrialPruned:
				raise
			except Exception as e:
				print(f"  Fold {fold_idx} failed: {e}")
				return float("inf")
		
		mean_loss = np.mean(fold_losses)
		trial.set_user_attr("fold_losses", fold_losses)
		trial.set_user_attr("hidden_layers", hidden_layers)
		
		return mean_loss
	
	return objective


# ============================================================================
# PHASE 1: COARSE SEARCH
# ============================================================================


def run_coarse_search(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
) -> optuna.Study:
	"""Phase 1: Coarse joint search over full parameter space."""
	task_label = "Result Prediction" if task_type == "multiclass" else "Over/Under"
	prefix = TASK_CONFIG[task_type]["run_prefix"]
	
	print_header(f"PHASE 1: COARSE JOINT SEARCH ({task_label})")
	print(f"Trials: {COARSE_TRIALS}")
	print(f"Epochs: {COARSE_EPOCHS}, Patience: {COARSE_PATIENCE}")
	print(f"CV Folds: {len(fold_data)}")
	if cat_config:
		print(f"Categorical: {cat_config.num_leagues} leagues (embed_dim={cat_config.league_embed_dim})")
	
	with mlflow.start_run(run_name=f"{prefix}phase1_coarse_search"):
		mlflow.log_params({
			"phase": "1_coarse",
			"task": task_type,
			"n_trials": COARSE_TRIALS,
			"max_epochs": COARSE_EPOCHS,
			"n_folds": len(fold_data),
			"pruner": "hyperband",
			"pruner_min_resource": PRUNER_MIN_RESOURCE,
			"pruner_reduction_factor": PRUNER_REDUCTION_FACTOR,
			"has_categorical": cat_config is not None,
		})
		
		objective = create_joint_objective(
			fold_data=fold_data,
			input_dim=input_dim,
			max_epochs=COARSE_EPOCHS,
			patience=COARSE_PATIENCE,
			task_type=task_type,
			cat_config=cat_config,
		)
		
		pruner = optuna.pruners.HyperbandPruner(
			min_resource=PRUNER_MIN_RESOURCE,
			max_resource=COARSE_EPOCHS,
			reduction_factor=PRUNER_REDUCTION_FACTOR,
		)
		
		study = optuna.create_study(
			direction="minimize",
			pruner=pruner,
			study_name=f"{prefix}phase1_coarse",
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
	"""Extract narrowed search ranges from top trials."""
	completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:top_n]
	
	if len(top_trials) < 5:
		print(f"Warning: Only {len(top_trials)} completed trials, using all for refinement")
	
	lrs = [t.params["lr"] for t in top_trials]
	wds = [t.params["weight_decay"] for t in top_trials]
	activations = [t.params["activation"] for t in top_trials]
	norms = [t.params["norm"] for t in top_trials]
	shapes = [t.params["shape"] for t in top_trials]
	base_widths = [t.params["base_width"] for t in top_trials]
	
	lr_range = (np.percentile(lrs, 10), np.percentile(lrs, 90))
	wd_range = (np.percentile(wds, 10), np.percentile(wds, 90))
	
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
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
) -> optuna.Study:
	"""Phase 2: Refinement search around top regions from Phase 1."""
	task_label = "Result Prediction" if task_type == "multiclass" else "Over/Under"
	prefix = TASK_CONFIG[task_type]["run_prefix"]
	
	print_header(f"PHASE 2: REFINEMENT SEARCH ({task_label})")
	
	ranges = extract_refinement_ranges(coarse_study, top_n=20)
	
	print(f"Trials: {REFINE_TRIALS}")
	print(f"Epochs: {REFINE_EPOCHS}, Patience: {REFINE_PATIENCE}")
	print(f"LR range: [{ranges['lr_range'][0]:.2e}, {ranges['lr_range'][1]:.2e}]")
	print(f"WD range: [{ranges['wd_range'][0]:.2e}, {ranges['wd_range'][1]:.2e}]")
	print(f"Activations: {ranges['allowed_activations']}")
	print(f"Norms: {ranges['allowed_norms']}")
	print(f"Shapes: {ranges['allowed_shapes']}")
	print(f"Base widths: {ranges['allowed_base_widths']}")
	
	with mlflow.start_run(run_name=f"{prefix}phase2_refinement"):
		mlflow.log_params({
			"phase": "2_refine",
			"task": task_type,
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
			task_type=task_type,
			cat_config=cat_config,
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
			study_name=f"{prefix}phase2_refine",
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


def extract_config_from_params(
	params: Dict,
	input_dim: int,
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
) -> TrainConfig:
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
		task_type=task_type,
		cat_config=cat_config,
	)


def retrain_with_seeds(
	config: TrainConfig,
	fold_data: List[Dict[str, Any]],
	seeds: List[int],
	task_type: TaskType,
) -> Tuple[float, float, List[float], List[int]]:
	"""Retrain config with multiple seeds across all CV folds."""
	all_losses = []
	stopping_epochs = []
	
	for seed in seeds:
		set_seed(seed, deterministic=False)
		
		fold_losses = []
		for fold in fold_data:
			train_loader, val_loader = fold_data_to_loaders(fold, config.batch_size, task_type=task_type)
			
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
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
) -> Tuple[TrainConfig, Dict, int]:
	"""Phase 3: Multi-seed evaluation of top configs."""
	prefix = TASK_CONFIG[task_type]["run_prefix"]
	
	print_header(f"PHASE 3: MULTI-SEED EVALUATION (Top {TOP_K_CONFIGS}, {SEEDS_PER_CONFIG} seeds)")
	
	completed = [t for t in refine_study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:TOP_K_CONFIGS]
	
	seeds = list(range(42, 42 + SEEDS_PER_CONFIG))
	results = []
	all_stopping_epochs = []
	
	with mlflow.start_run(run_name=f"{prefix}phase3_multi_seed"):
		mlflow.log_params({
			"phase": "3_multi_seed",
			"task": task_type,
			"top_k": TOP_K_CONFIGS,
			"seeds_per_config": SEEDS_PER_CONFIG,
		})
		
		for i, trial in enumerate(top_trials):
			config = extract_config_from_params(trial.params, input_dim, task_type, cat_config)
			print(f"\nConfig {i+1}/{TOP_K_CONFIGS}: {config.hidden_layers} | {config.activation} | {config.norm}")
			
			mean_loss, std_loss, losses, stopping_epochs = retrain_with_seeds(
				config, fold_data, seeds, task_type
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
	df: pl.DataFrame,
	test_season: str,
	task_type: TaskType,
) -> bool:
	"""Compare new model against existing and save if better."""
	cfg = TASK_CONFIG[task_type]
	model_path = MODELS_DIR / cfg["model_path"]
	config_path = MODELS_DIR / cfg["config_path"]
	scaler_path = MODELS_DIR / cfg["scaler_path"]
	comparison_metric = cfg["comparison_metric"]
	prepare_fn = cfg["prepare_fn"]
	
	existing_model, existing_config = load_existing_model(config_path, model_path, DEVICE, task_type=task_type)
	
	new_metric_val = new_metrics[comparison_metric]
	
	if existing_model is None:
		print("\nNo existing model found. Saving new model.")
		save_new = True
	else:
		print("\nEvaluating existing model on test set...")
		try:
			old_feature_cols = existing_config.get("feature_cols") if existing_config else None
			old_scaler = joblib.load(scaler_path) if scaler_path.exists() else None
			
			if old_feature_cols and set(old_feature_cols) != set(feature_cols):
				print(f"Feature sets differ ({len(old_feature_cols)} vs {len(feature_cols)} features).")
				print("Re-preparing test data with old model's features and scaler...")
				if old_scaler is None:
					raise FileNotFoundError(f"Missing saved scaler: {scaler_path}")
				data_test_old = prepare_fn(df, old_feature_cols, [test_season], scaler=old_scaler)
			else:
				scaler_for_existing = old_scaler if old_scaler is not None else data_train["scaler"]
				data_test_old = prepare_fn(df, feature_cols, [test_season], scaler=scaler_for_existing)
			
			existing_metrics = evaluate_model(existing_model, data_test_old, device=DEVICE, verbose=False, task_type=task_type)
			existing_metric_val = existing_metrics[comparison_metric]
			
			print(f"\n{'='*40}")
			print("MODEL COMPARISON (on test set)")
			print(f"{'='*40}")
			print(f"Existing model {comparison_metric}: {existing_metric_val:.5f}")
			print(f"New model {comparison_metric}:      {new_metric_val:.5f}")
			
			if new_metric_val < existing_metric_val:
				improvement = (existing_metric_val - new_metric_val) / existing_metric_val * 100
				print(f"New model is BETTER by {improvement:.2f}%")
				save_new = True
			else:
				degradation = (new_metric_val - existing_metric_val) / existing_metric_val * 100
				print(f"Existing model is better by {degradation:.2f}%")
				print("Keeping existing model.")
				save_new = False
		except Exception as e:
			print(f"Failed to evaluate existing model: {e}")
			print("Saving new model (existing model incompatible).")
			save_new = True
	
	if save_new:
		torch.save(new_model.state_dict(), model_path)
		print(f"Model saved to {model_path}")
		
		joblib.dump(data_train["scaler"], scaler_path)
		print(f"Scaler saved to {scaler_path}")
		
		cat_config_dict = None
		if new_config.cat_config is not None:
			cat_config_dict = {
				"num_leagues": new_config.cat_config.num_leagues,
				"league_embed_dim": new_config.cat_config.league_embed_dim,
				"num_season_stages": new_config.cat_config.num_season_stages,
			}
		
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
			"task_type": task_type,
			"output_dim": 1 if task_type == "binary" else 3,
			"cat_config": cat_config_dict,
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
	task_type: TaskType,
	cat_config: CategoricalConfig = None,
):
	"""Phase 4: Train final model with proper early stopping, evaluate on held-out test."""
	cfg = TASK_CONFIG[task_type]
	prepare_fn = cfg["prepare_fn"]
	prefix = cfg["run_prefix"]
	task_label = "Result Prediction" if task_type == "multiclass" else "Over/Under"
	
	print_header(f"PHASE 4: FINAL MODEL TRAINING ({task_label})")
	
	all_cv_seasons = set()
	for train_seasons, val_season in folds:
		all_cv_seasons.update(train_seasons)
		all_cv_seasons.add(val_season)
	all_cv_seasons = sorted(all_cv_seasons)
	
	final_val_season = all_cv_seasons[-1]
	initial_train_seasons = all_cv_seasons[:-1]
	
	print(f"Step 1: Train on {initial_train_seasons[0]}..{initial_train_seasons[-1]} ({len(initial_train_seasons)} seasons)")
	print(f"        Validate on {final_val_season} (early stopping)")
	print(f"Step 2: Retrain on {all_cv_seasons[0]}..{all_cv_seasons[-1]} ({len(all_cv_seasons)} seasons) for best_epoch")
	print(f"Test season: {test_season}")
	if cat_config:
		print(f"Categorical: {cat_config.num_leagues} leagues (embed_dim={cat_config.league_embed_dim})")
	
	set_seed(42, deterministic=True)
	
	with mlflow.start_run(run_name=f"{prefix}phase4_final_model"):
		# Step 1: Train with early stopping
		print("\n--- Step 1: Finding best epoch with early stopping ---")
		
		data_initial_train = prepare_fn(df, feature_cols, initial_train_seasons, fit_scaler=True)
		data_final_val = prepare_fn(df, feature_cols, [final_val_season], scaler=data_initial_train["scaler"])
		
		initial_train_loader = to_loader(data_initial_train, config.batch_size, device=DEVICE, task_type=task_type)
		final_val_loader = to_loader(data_final_val, config.batch_size, shuffle=False, device=DEVICE, task_type=task_type)
		
		early_stop_config = TrainConfig(
			input_dim=data_initial_train["X"].shape[1],
			hidden_layers=config.hidden_layers,
			dropout=config.dropout,
			norm=config.norm,
			lr=config.lr,
			weight_decay=config.weight_decay,
			lambda_repulsion=config.lambda_repulsion,
			lambda_corr=config.lambda_corr,
			activation=config.activation,
			scheduler_type=config.scheduler_type,
			epochs=REFINE_EPOCHS,
			patience=REFINE_PATIENCE,
			batch_size=config.batch_size,
			task_type=task_type,
			cat_config=cat_config,
		)
		
		_, early_stop_history, best_val_loss = train_model(
			early_stop_config, initial_train_loader, final_val_loader, device=DEVICE, verbose=True
		)
		
		best_epoch = early_stop_history["val_loss"].index(min(early_stop_history["val_loss"])) + 1
		total_epochs_trained = len(early_stop_history["val_loss"])
		print(f"\nEarly stopping: trained for {total_epochs_trained} epochs, best was epoch {best_epoch} (val_loss = {best_val_loss:.5f})")
		
		# Step 2: Retrain on all CV seasons
		print(f"\n--- Step 2: Retraining on all data for {best_epoch} epochs (fixed) ---")
		
		data_train = prepare_fn(df, feature_cols, all_cv_seasons, fit_scaler=True)
		data_test = prepare_fn(df, feature_cols, [test_season], scaler=data_train["scaler"])
		
		train_loader = to_loader(data_train, config.batch_size, device=DEVICE, task_type=task_type)
		
		config.input_dim = data_train["X"].shape[1]
		config.epochs = best_epoch
		config.patience = best_epoch + 1
		config.cat_config = cat_config
		
		mlflow.log_params({
			"phase": "4_final",
			"task": task_type,
			"hidden_layers": str(config.hidden_layers),
			"activation": config.activation,
			"norm": config.norm,
			"dropout": config.dropout,
			"lr": config.lr,
			"weight_decay": config.weight_decay,
			"scheduler_type": config.scheduler_type,
			"batch_size": config.batch_size,
			"best_epoch": best_epoch,
			"early_stop_val_loss": best_val_loss,
			"initial_train_seasons": f"{initial_train_seasons[0]}..{initial_train_seasons[-1]}",
			"final_val_season": final_val_season,
			"retrain_seasons": f"{all_cv_seasons[0]}..{all_cv_seasons[-1]}",
			"test_season": test_season,
			"has_categorical": cat_config is not None,
		})
		
		# Evaluate baseline for multiclass
		if task_type == "multiclass":
			print("\n--- Baseline (Bookmaker Implied Probabilities) ---")
			baseline_metrics = evaluate_implied_baseline(data_test, task_type=task_type)
			print(f"Accuracy: {baseline_metrics['accuracy']:.4f}, Brier: {baseline_metrics['brier']:.4f}, "
				  f"RPS: {baseline_metrics['rps']:.4f}, LogLoss: {baseline_metrics['log_loss']:.4f}")
			
			mlflow.log_metrics({
				"baseline_accuracy": baseline_metrics["accuracy"],
				"baseline_brier": baseline_metrics["brier"],
				"baseline_rps": baseline_metrics["rps"],
				"baseline_log_loss": baseline_metrics["log_loss"],
			})
		
		print("\n--- Training Final Model ---")
		
		dummy_val_loader = to_loader(data_train, config.batch_size, shuffle=False, device=DEVICE, task_type=task_type)
		
		model, history, _ = train_model(
			config, train_loader, dummy_val_loader, device=DEVICE, verbose=True
		)
		
		print("\n--- Model Performance on Held-out Test Set ---")
		metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True, task_type=task_type)
		
		log_metrics = {
			"test_accuracy": metrics["accuracy"],
			"test_brier": metrics["brier"],
			"test_log_loss": metrics["log_loss"],
			"test_corr": metrics["corr_with_implied"],
		}
		if task_type == "multiclass":
			log_metrics["test_rps"] = metrics["rps"]
		mlflow.log_metrics(log_metrics)
		
		# Print comparison with baseline for multiclass
		if task_type == "multiclass":
			print("\n--- Comparison vs Baseline ---")
			print(f"{'Metric':<15} {'Baseline':>12} {'Model':>12} {'Diff':>12}")
			print("-" * 55)
			for metric in ["accuracy", "brier", "rps", "log_loss"]:
				baseline_val = baseline_metrics[metric]
				model_val = metrics[metric]
				diff = model_val - baseline_val
				sign = "+" if (diff > 0 and metric == "accuracy") or (diff < 0 and metric != "accuracy") else ""
				print(f"{metric:<15} {baseline_val:>12.4f} {model_val:>12.4f} {sign}{diff:>11.4f}")
		
		model_saved = compare_and_save_model(
			new_model=model,
			new_metrics=metrics,
			new_config=config,
			feature_cols=feature_cols,
			final_epochs=best_epoch,
			data_train=data_train,
			df=df,
			test_season=test_season,
			task_type=task_type,
		)
		
		mlflow.log_metric("model_saved", int(model_saved))
	
	return model, data_train["scaler"], metrics


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_pipeline(task_type: TaskType):
	"""Main entry point for architecture search pipeline."""
	cfg = TASK_CONFIG[task_type]
	task_label = "RESULT PREDICTION (Home/Draw/Away)" if task_type == "multiclass" else "OVER/UNDER 2.5 GOALS"
	
	print_header(f"ARCHITECTURE SEARCH PIPELINE: {task_label}")
	
	print(f"\nDevice: {DEVICE}")
	if DEVICE.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(0)}")
	
	set_seed(42, deterministic=False)
	
	# Load and prepare data
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = cfg["add_targets_fn"](df)
	
	df = df.drop_nulls(subset=cfg["odds_cols"])
	print(f"Total rows with odds: {len(df)}")
	
	feature_cols = select_feature_columns(df)
	print(f"Features: {len(feature_cols)} columns")
	
	num_leagues = get_num_leagues(df)
	cat_config = CategoricalConfig(
		num_leagues=num_leagues,
		league_embed_dim=3,
		num_season_stages=3,
	)
	print(f"Categorical config: {num_leagues} leagues, embed_dim=3, stages=3")
	
	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Test season (held out): {test_season}")
	
	print("\nPrecomputing scaled data for CV folds...")
	fold_data = precompute_fold_data(df, feature_cols, folds, task_type=task_type)
	input_dim = fold_data[0]["X_train"].shape[1]
	print(f"Input dimension: {input_dim}")
	
	mlflow.set_experiment(cfg["experiment_name"])
	
	# Phase 1
	coarse_study = run_coarse_search(fold_data, input_dim, task_type, cat_config)
	
	# Phase 2
	refine_study = run_refinement_search(fold_data, input_dim, coarse_study, task_type, cat_config)
	
	# Phase 3
	best_config, best_result, _ = run_multi_seed_evaluation(
		refine_study, fold_data, input_dim, task_type, cat_config
	)
	
	# Phase 4
	model, scaler, test_metrics = train_final_model(
		best_config, df, feature_cols, folds, test_season, task_type, cat_config
	)
	
	print_header("PIPELINE COMPLETE")
	print(f"\nFinal architecture: {best_config.hidden_layers}")
	print(f"Activation: {best_config.activation}, Norm: {best_config.norm}")
	print(f"Test metrics: {test_metrics}")
	print(f"\nArtifacts saved to {MODELS_DIR}")
