"""
Fixed-Architecture Optuna Sweep for Gated Residual NN

Goal:
	- Keep MLP architecture fixed (loaded from JSON)
	- Sweep training + gate hyperparameters with Optuna
	- Keep lambdas at 0 (no betting-regularization tuning here)

Usage:
	# Result prediction (default)
	uv run training/fixed_arch_sweep.py

	# Over/Under
	$env:TASK_TYPE = "binary"; uv run training/fixed_arch_sweep.py

	# Custom fixed-arch config
	$env:ARCH_CONFIG_PATH = "training/fixed_arch_model_config.json"; uv run training/fixed_arch_sweep.py
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
from training.models import CategoricalConfig, TrainConfig
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
	load_frame,
	precompute_fold_data,
	prepare_data,
	prepare_data_result,
	select_feature_columns,
	to_loader,
	train_model,
)

TaskType = Literal["binary", "multiclass"]

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ARCH_CONFIG_PATH = Path(os.environ.get("ARCH_CONFIG_PATH", "training/fixed_arch_model_config.json"))
TASK_TYPE: TaskType = os.environ.get("TASK_TYPE", "multiclass")  # "binary" or "multiclass"

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep settings
N_TRIALS = int(os.environ.get("N_TRIALS", "120"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "35"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))

TOP_K_CONFIGS = int(os.environ.get("TOP_K_CONFIGS", "8"))
SEEDS_PER_CONFIG = int(os.environ.get("SEEDS_PER_CONFIG", "5"))
N_CV_FOLDS = int(os.environ.get("N_CV_FOLDS", "3"))

PRUNER_MIN_RESOURCE = int(os.environ.get("PRUNER_MIN_RESOURCE", "5"))
PRUNER_REDUCTION_FACTOR = int(os.environ.get("PRUNER_REDUCTION_FACTOR", "3"))


TASK_CONFIG = {
	"binary": {
		"add_targets_fn": add_targets_and_implied,
		"prepare_fn": prepare_data,
		"odds_cols": ["odds_over", "odds_under"],
		"comparison_metric": "brier",
		"experiment_name": "over_under_fixed_arch_sweep",
		"run_prefix": "",
		"model_path": "over_under_fixed_arch_sweep.pt",
		"config_path": "over_under_fixed_arch_sweep_config.json",
		"scaler_path": "over_under_fixed_arch_sweep_scaler.joblib",
	},
	"multiclass": {
		"add_targets_fn": add_targets_and_implied_result,
		"prepare_fn": prepare_data_result,
		"odds_cols": ["odds_home", "odds_draw", "odds_away"],
		"comparison_metric": "log_loss",
		"experiment_name": "result_fixed_arch_sweep",
		"run_prefix": "result_",
		"model_path": "result_fixed_arch_sweep.pt",
		"config_path": "result_fixed_arch_sweep_config.json",
		"scaler_path": "result_fixed_arch_sweep_scaler.joblib",
	},
}


def set_seed(seed: int = 42, deterministic: bool = False):
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
	print("\n" + "=" * 60)
	print(text)
	print("=" * 60)


def load_arch_config(config_path: Path) -> Dict[str, Any]:
	if not config_path.exists():
		raise FileNotFoundError(f"Architecture config not found: {config_path}")
	with open(config_path, "r") as f:
		cfg = json.load(f)
	
	need = ["base_width", "n_layers", "shape", "activation", "norm"]
	missing = [k for k in need if k not in cfg]
	if missing:
		raise ValueError(f"Missing keys in {config_path}: {missing}")
	
	return cfg


def create_fixed_objective(
	fold_data: List[Dict[str, Any]],
	input_dim: int,
	hidden_layers: List[int],
	activation: str,
	norm: str,
	task_type: TaskType,
	cat_config: CategoricalConfig,
):
	def objective(trial: optuna.Trial) -> float:
		set_seed(42 + trial.number, deterministic=False)
		
		lr = trial.suggest_float("lr", 1e-5, 3e-2, log=True)
		weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
		dropout = trial.suggest_float("dropout", 0.05, 0.40)
		batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
		beta1 = trial.suggest_float("beta1", 0.75, 0.95)
		
		# Gate hyperparameters
		gate_hidden_dim = trial.suggest_categorical("gate_hidden_dim", [16, 32, 64, 128])
		gate_target_budget = trial.suggest_float("gate_target_budget", 0.05, 0.50)
		gate_mean_weight = trial.suggest_float("gate_mean_weight", 1e-4, 5e-2, log=True)
		gate_sat_weight = trial.suggest_float("gate_sat_weight", 1e-5, 1e-2, log=True)
		
		fold_losses = []
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
				activation=activation,
				beta1=beta1,
				epochs=MAX_EPOCHS,
				patience=PATIENCE,
				batch_size=batch_size,
				task_type=task_type,
				cat_config=cat_config,
				gate_hidden_dim=gate_hidden_dim,
				gate_target_budget=gate_target_budget,
				gate_mean_weight=gate_mean_weight,
				gate_sat_weight=gate_sat_weight,
				lambda_repulsion=0.0,
				lambda_corr=0.0,
			)
			
			try:
				trial_for_fold = trial if fold_idx == 0 else None
				_, _, best_val_loss = train_model(
					config,
					train_loader,
					val_loader,
					device=DEVICE,
					trial=trial_for_fold,
					verbose=False,
				)
				fold_losses.append(best_val_loss)
			except optuna.TrialPruned:
				raise
			except Exception as e:
				print(f"Fold {fold_idx} failed: {e}")
				return float("inf")
		
		mean_loss = float(np.mean(fold_losses))
		trial.set_user_attr("fold_losses", fold_losses)
		return mean_loss
	
	return objective


def extract_config_from_trial(
	trial: optuna.trial.FrozenTrial,
	input_dim: int,
	hidden_layers: List[int],
	activation: str,
	norm: str,
	task_type: TaskType,
	cat_config: CategoricalConfig,
) -> TrainConfig:
	p = trial.params
	return TrainConfig(
		input_dim=input_dim,
		hidden_layers=hidden_layers,
		dropout=p["dropout"],
		norm=norm,
		lr=p["lr"],
		weight_decay=p["weight_decay"],
		activation=activation,
		beta1=p["beta1"],
		epochs=MAX_EPOCHS,
		patience=PATIENCE,
		batch_size=p["batch_size"],
		task_type=task_type,
		cat_config=cat_config,
		gate_hidden_dim=p["gate_hidden_dim"],
		gate_target_budget=p["gate_target_budget"],
		gate_mean_weight=p["gate_mean_weight"],
		gate_sat_weight=p["gate_sat_weight"],
		lambda_repulsion=0.0,
		lambda_corr=0.0,
	)


def retrain_with_seeds(
	config: TrainConfig,
	fold_data: List[Dict[str, Any]],
	seeds: List[int],
	task_type: TaskType,
) -> Tuple[float, float, List[float]]:
	all_losses = []
	
	for seed in seeds:
		set_seed(seed, deterministic=False)
		fold_losses = []
		for fold in fold_data:
			train_loader, val_loader = fold_data_to_loaders(fold, config.batch_size, task_type=task_type)
			_, _, best_val_loss = train_model(config, train_loader, val_loader, device=DEVICE, verbose=False)
			fold_losses.append(best_val_loss)
		all_losses.append(float(np.mean(fold_losses)))
	
	return float(np.mean(all_losses)), float(np.std(all_losses)), all_losses


def compare_and_save_model(
	model: Any,
	config: TrainConfig,
	feature_cols: List[str],
	final_epochs: int,
	data_train: Dict[str, Any],
	task_type: TaskType,
	arch_cfg: Dict[str, Any],
):
	cfg = TASK_CONFIG[task_type]
	model_path = MODELS_DIR / cfg["model_path"]
	config_path = MODELS_DIR / cfg["config_path"]
	scaler_path = MODELS_DIR / cfg["scaler_path"]
	
	torch.save(model.state_dict(), model_path)
	joblib.dump(data_train["scaler"], scaler_path)
	
	cat_cfg = None
	if config.cat_config is not None:
		cat_cfg = {
			"num_leagues": config.cat_config.num_leagues,
			"league_embed_dim": config.cat_config.league_embed_dim,
		}
	
	out = {
		"task_type": task_type,
		"input_dim": config.input_dim,
		"hidden_layers": config.hidden_layers,
		"activation": config.activation,
		"norm": config.norm,
		"dropout": config.dropout,
		"lr": config.lr,
		"weight_decay": config.weight_decay,
		"beta1": config.beta1,
		"batch_size": config.batch_size,
		"final_epochs": final_epochs,
		"feature_cols": feature_cols,
		"output_dim": 1 if task_type == "binary" else 3,
		"cat_config": cat_cfg,
		"gate_hidden_dim": config.gate_hidden_dim,
		"gate_target_budget": config.gate_target_budget,
		"gate_mean_weight": config.gate_mean_weight,
		"gate_sat_weight": config.gate_sat_weight,
		"lambda_repulsion": 0.0,
		"lambda_corr": 0.0,
		"arch_config_path": str(ARCH_CONFIG_PATH),
		"arch_config": arch_cfg,
	}
	with open(config_path, "w") as f:
		json.dump(out, f, indent=2)
	
	mlflow.log_artifact(str(model_path))
	mlflow.log_artifact(str(scaler_path))
	mlflow.log_artifact(str(config_path))


def train_final_model(
	best_config: TrainConfig,
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
	test_season: str,
	task_type: TaskType,
	cat_config: CategoricalConfig,
	arch_cfg: Dict[str, Any],
):
	cfg = TASK_CONFIG[task_type]
	prepare_fn = cfg["prepare_fn"]
	prefix = cfg["run_prefix"]
	
	all_cv_seasons = set()
	for train_seasons, val_season in folds:
		all_cv_seasons.update(train_seasons)
		all_cv_seasons.add(val_season)
	all_cv_seasons = sorted(all_cv_seasons)
	
	final_val_season = all_cv_seasons[-1]
	initial_train_seasons = all_cv_seasons[:-1]
	
	print_header("FINAL TRAINING")
	print(f"Step 1: Train on {initial_train_seasons[0]}..{initial_train_seasons[-1]} | Val on {final_val_season}")
	print(f"Step 2: Retrain on {all_cv_seasons[0]}..{all_cv_seasons[-1]} for best_epoch")
	print(f"Test season: {test_season}")
	
	set_seed(42, deterministic=True)
	
	with mlflow.start_run(run_name=f"{prefix}final_model"):
		# Step 1: early stopping to pick epoch
		data_initial_train = prepare_fn(df, feature_cols, initial_train_seasons, fit_scaler=True)
		data_final_val = prepare_fn(df, feature_cols, [final_val_season], scaler=data_initial_train["scaler"])
		
		initial_train_loader = to_loader(data_initial_train, best_config.batch_size, device=DEVICE, task_type=task_type)
		final_val_loader = to_loader(data_final_val, best_config.batch_size, shuffle=False, device=DEVICE, task_type=task_type)
		
		early_stop_cfg = TrainConfig(
			input_dim=data_initial_train["X"].shape[1],
			hidden_layers=best_config.hidden_layers,
			dropout=best_config.dropout,
			norm=best_config.norm,
			lr=best_config.lr,
			weight_decay=best_config.weight_decay,
			activation=best_config.activation,
			beta1=best_config.beta1,
			epochs=MAX_EPOCHS,
			patience=PATIENCE,
			batch_size=best_config.batch_size,
			task_type=task_type,
			cat_config=cat_config,
			gate_hidden_dim=best_config.gate_hidden_dim,
			gate_target_budget=best_config.gate_target_budget,
			gate_mean_weight=best_config.gate_mean_weight,
			gate_sat_weight=best_config.gate_sat_weight,
			lambda_repulsion=0.0,
			lambda_corr=0.0,
		)
		
		_, history, best_val_loss = train_model(
			early_stop_cfg,
			initial_train_loader,
			final_val_loader,
			device=DEVICE,
			verbose=True,
		)
		best_epoch = history["val_loss"].index(min(history["val_loss"])) + 1
		print(f"Early stopping best epoch: {best_epoch} (val_loss={best_val_loss:.5f})")
		
		# Step 2: retrain on all CV seasons, fixed epochs
		data_train = prepare_fn(df, feature_cols, all_cv_seasons, fit_scaler=True)
		data_test = prepare_fn(df, feature_cols, [test_season], scaler=data_train["scaler"])
		train_loader = to_loader(data_train, best_config.batch_size, device=DEVICE, task_type=task_type)
		
		final_cfg = TrainConfig(
			input_dim=data_train["X"].shape[1],
			hidden_layers=best_config.hidden_layers,
			dropout=best_config.dropout,
			norm=best_config.norm,
			lr=best_config.lr,
			weight_decay=best_config.weight_decay,
			activation=best_config.activation,
			beta1=best_config.beta1,
			epochs=best_epoch,
			patience=PATIENCE,
			batch_size=best_config.batch_size,
			task_type=task_type,
			cat_config=cat_config,
			gate_hidden_dim=best_config.gate_hidden_dim,
			gate_target_budget=best_config.gate_target_budget,
			gate_mean_weight=best_config.gate_mean_weight,
			gate_sat_weight=best_config.gate_sat_weight,
			lambda_repulsion=0.0,
			lambda_corr=0.0,
		)
		
		mlflow.log_params({
			"phase": "final",
			"task": task_type,
			"arch": json.dumps(arch_cfg),
			"hidden_layers": str(final_cfg.hidden_layers),
			"activation": final_cfg.activation,
			"norm": final_cfg.norm,
			"dropout": final_cfg.dropout,
			"lr": final_cfg.lr,
			"weight_decay": final_cfg.weight_decay,
			"beta1": final_cfg.beta1,
			"batch_size": final_cfg.batch_size,
			"gate_hidden_dim": final_cfg.gate_hidden_dim,
			"gate_target_budget": final_cfg.gate_target_budget,
			"gate_mean_weight": final_cfg.gate_mean_weight,
			"gate_sat_weight": final_cfg.gate_sat_weight,
			"best_epoch": best_epoch,
			"early_stop_val_loss": best_val_loss,
			"test_season": test_season,
		})
		
		if task_type == "multiclass":
			print("\n--- Baseline (Bookmaker Implied Probabilities) ---")
			baseline_metrics = evaluate_implied_baseline(data_test, task_type=task_type)
			print(
				f"Accuracy: {baseline_metrics['accuracy']:.4f}, Brier: {baseline_metrics['brier']:.4f}, "
				f"RPS: {baseline_metrics['rps']:.4f}, LogLoss: {baseline_metrics['log_loss']:.4f}"
			)
			mlflow.log_metrics({
				"baseline_accuracy": baseline_metrics["accuracy"],
				"baseline_brier": baseline_metrics["brier"],
				"baseline_rps": baseline_metrics["rps"],
				"baseline_log_loss": baseline_metrics["log_loss"],
			})
		
		print("\n--- Training Final Model ---")
		model, _, _ = train_model(final_cfg, train_loader, val_loader=None, device=DEVICE, verbose=True)
		
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
		
		compare_and_save_model(
			model=model,
			config=final_cfg,
			feature_cols=feature_cols,
			final_epochs=best_epoch,
			data_train=data_train,
			task_type=task_type,
			arch_cfg=arch_cfg,
		)
	
	return metrics


def run():
	if TASK_TYPE not in TASK_CONFIG:
		raise ValueError(f"Unknown TASK_TYPE={TASK_TYPE}. Use 'binary' or 'multiclass'.")
	
	cfg = TASK_CONFIG[TASK_TYPE]
	
	print_header("FIXED ARCHITECTURE SWEEP")
	print(f"Task: {TASK_TYPE}")
	print(f"Device: {DEVICE}")
	print(f"Arch config: {ARCH_CONFIG_PATH}")
	
	arch_cfg = load_arch_config(ARCH_CONFIG_PATH)
	hidden_layers = build_hidden_layers(arch_cfg["base_width"], arch_cfg["n_layers"], arch_cfg["shape"])
	activation = arch_cfg["activation"]
	norm = arch_cfg["norm"]
	print(f"Fixed hidden_layers: {hidden_layers} | activation={activation} | norm={norm}")
	
	set_seed(42, deterministic=False)
	
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = cfg["add_targets_fn"](df)
	df = df.drop_nulls(subset=cfg["odds_cols"])
	print(f"Total rows with odds: {len(df)}")
	
	feature_cols = select_feature_columns(df)
	print(f"Features: {len(feature_cols)}")
	
	num_leagues = get_num_leagues(df)
	cat_config = CategoricalConfig(num_leagues=num_leagues, league_embed_dim=3)
	print(f"Categorical: {num_leagues} leagues (embed_dim=3)")
	
	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Test season (held out): {test_season}")
	
	print("\nPrecomputing scaled data for CV folds...")
	fold_data = precompute_fold_data(df, feature_cols, folds, task_type=TASK_TYPE)
	input_dim = fold_data[0]["X_train"].shape[1]
	print(f"Input dim: {input_dim}")
	
	mlflow.set_experiment(cfg["experiment_name"])
	
	print_header("STAGE A: OPTUNA SWEEP")
	print(f"Trials: {N_TRIALS} | Epochs: {MAX_EPOCHS} | Patience: {PATIENCE}")
	
	objective = create_fixed_objective(
		fold_data=fold_data,
		input_dim=input_dim,
		hidden_layers=hidden_layers,
		activation=activation,
		norm=norm,
		task_type=TASK_TYPE,
		cat_config=cat_config,
	)
	
	pruner = optuna.pruners.HyperbandPruner(
		min_resource=PRUNER_MIN_RESOURCE,
		max_resource=MAX_EPOCHS,
		reduction_factor=PRUNER_REDUCTION_FACTOR,
	)
	sampler = optuna.samplers.TPESampler(seed=42)
	study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler, study_name=f"{cfg['run_prefix']}fixed_arch")
	
	with mlflow.start_run(run_name=f"{cfg['run_prefix']}stageA_optuna"):
		mlflow.log_params({
			"stage": "A_optuna",
			"task": TASK_TYPE,
			"arch": json.dumps(arch_cfg),
			"n_trials": N_TRIALS,
			"max_epochs": MAX_EPOCHS,
			"patience": PATIENCE,
			"n_folds": len(fold_data),
			"pruner": "hyperband",
		})
		study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
		mlflow.log_params({f"best_{k}": str(v) for k, v in study.best_params.items()})
		mlflow.log_metric("best_val_loss", float(study.best_value))
	
	print(f"\nBest trial: {study.best_trial.number} | mean CV loss: {study.best_value:.5f}")
	print(f"Best params: {study.best_params}")
	
	print_header("STAGE B: MULTI-SEED CONFIRM")
	completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
	top_trials = sorted(completed, key=lambda t: t.value)[:TOP_K_CONFIGS]
	seeds = list(range(42, 42 + SEEDS_PER_CONFIG))
	
	best_mean = float("inf")
	best_trial = None
	best_cfg = None
	
	with mlflow.start_run(run_name=f"{cfg['run_prefix']}stageB_multi_seed"):
		mlflow.log_params({
			"stage": "B_multi_seed",
			"top_k": TOP_K_CONFIGS,
			"seeds_per_config": SEEDS_PER_CONFIG,
		})
		for i, tr in enumerate(top_trials):
			trial_cfg = extract_config_from_trial(tr, input_dim, hidden_layers, activation, norm, TASK_TYPE, cat_config)
			mean_loss, std_loss, _ = retrain_with_seeds(trial_cfg, fold_data, seeds, TASK_TYPE)
			print(f"Config {i+1}/{len(top_trials)} trial={tr.number} | mean={mean_loss:.5f} ± {std_loss:.5f}")
			
			if mean_loss < best_mean:
				best_mean = mean_loss
				best_trial = tr
				best_cfg = trial_cfg
		
		mlflow.log_metric("best_mean_val_loss", best_mean)
		if best_trial is not None:
			mlflow.log_params({f"best_seed_{k}": str(v) for k, v in best_trial.params.items()})
	
	print(f"\nBest confirmed trial: {best_trial.number} | mean={best_mean:.5f}")
	
	print_header("STAGE C: FINAL TRAIN + TEST")
	metrics = train_final_model(
		best_config=best_cfg,
		df=df,
		feature_cols=feature_cols,
		folds=folds,
		test_season=test_season,
		task_type=TASK_TYPE,
		cat_config=cat_config,
		arch_cfg=arch_cfg,
	)
	
	print_header("DONE")
	print(f"Test metrics: {metrics}")


if __name__ == "__main__":
	run()
