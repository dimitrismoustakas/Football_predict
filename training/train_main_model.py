"""
Canonical training entry point for the two production models.

This script keeps the evaluation loop fixed:
- frozen rolling CV folds for model selection
- fixed epoch-selection season for epoch selection
- fixed held-out latest season for acceptance

Use search scripts only when you explicitly want to sweep hyperparameters.
"""

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import mlflow
import numpy as np
import torch

from training.evaluation import evaluate_model
from training.models import CategoricalConfig, TrainConfig
from training.train_utils import (
	add_targets_and_implied,
	add_targets_and_implied_result,
	evaluate_implied_baseline,
	filter_min_history,
	generate_rolling_cv_folds,
	get_num_leagues,
	get_test_season,
	load_frame,
	prepare_data,
	prepare_data_result,
	select_feature_columns,
	to_loader,
	train_model,
)
from utils.paths import MODELS_DIR, PROJECT_ROOT

TaskType = Literal["binary", "multiclass"]

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
MODEL_CONFIGS_DIR = PROJECT_ROOT / "training" / "configs" / "main_models"
LATEST_MAIN_METRICS_PATH = MODELS_DIR / "latest_main_model_metrics.json"
TASK_TYPE: TaskType = os.environ.get("TASK_TYPE", "binary")
N_CV_FOLDS = int(os.environ.get("N_CV_FOLDS", "3"))
TRAINING_SEED = int(os.environ.get("TRAINING_SEED", "42"))

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_CONFIG = {
	"binary": {
		"label": "over_under",
		"display_name": "Over/Under 2.5 Goals",
		"config_path": MODEL_CONFIGS_DIR / "over_under.json",
		"add_targets_fn": add_targets_and_implied,
		"prepare_fn": prepare_data,
		"odds_cols": ["odds_over", "odds_under"],
		"comparison_metric": "log_loss",
		"experiment_name": "over_under_main_model",
		"artifact_model_path": MODELS_DIR / "over_under_model.pt",
		"artifact_config_path": MODELS_DIR / "over_under_model_config.json",
		"artifact_scaler_path": MODELS_DIR / "over_under_model_scaler.joblib",
	},
	"multiclass": {
		"label": "result",
		"display_name": "Match Result",
		"config_path": MODEL_CONFIGS_DIR / "result.json",
		"add_targets_fn": add_targets_and_implied_result,
		"prepare_fn": prepare_data_result,
		"odds_cols": ["odds_home", "odds_draw", "odds_away"],
		"comparison_metric": "log_loss",
		"experiment_name": "result_main_model",
		"artifact_model_path": MODELS_DIR / "result_model.pt",
		"artifact_config_path": MODELS_DIR / "result_model_config.json",
		"artifact_scaler_path": MODELS_DIR / "result_model_scaler.joblib",
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


def load_training_config(task_type: TaskType) -> Dict[str, Any]:
	config_path = TASK_CONFIG[task_type]["config_path"]
	with open(config_path, "r", encoding="utf-8") as f:
		config = json.load(f)
	return config


def to_builtin(value: Any) -> Any:
	if isinstance(value, dict):
		return {key: to_builtin(val) for key, val in value.items()}
	if isinstance(value, list):
		return [to_builtin(item) for item in value]
	if isinstance(value, tuple):
		return [to_builtin(item) for item in value]
	if isinstance(value, np.generic):
		return value.item()
	return value


def summarize_metrics(task_type: TaskType, metrics: Dict[str, Any]) -> Dict[str, Any]:
	if task_type == "binary":
		keys = [
			"accuracy",
			"brier",
			"log_loss",
			"total_profit",
			"daily_total_profit",
			"daily_roi",
			"n_bets",
		]
	else:
		keys = [
			"accuracy",
			"brier",
			"rps",
			"log_loss",
			"total_profit",
			"avg_profit",
			"n_bets",
		]

	return to_builtin({key: metrics[key] for key in keys if key in metrics})


def write_json(path: Path, payload: Dict[str, Any]):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def update_latest_run_record(task_type: TaskType, run_record: Dict[str, Any]):
	if LATEST_MAIN_METRICS_PATH.exists():
		with open(LATEST_MAIN_METRICS_PATH, "r", encoding="utf-8") as f:
			payload = json.load(f)
	else:
		payload = {
			"schema_version": 1,
			"description": "Latest evaluated main-model candidates. Runtime-generated; compare with training/configs/main_models/baselines.json.",
			"models": {},
		}

	payload["models"][task_type] = run_record
	write_json(LATEST_MAIN_METRICS_PATH, payload)


def log_metric_group(prefix: str, metrics: Dict[str, Any]):
	for metric_name, metric_value in metrics.items():
		if isinstance(metric_value, (int, float)):
			mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))


def build_latest_run_record(
	task_type: TaskType,
	epoch_selection_season: str,
	test_season: str,
	best_epoch: int,
	best_val_loss: float,
	validation_metrics: Dict[str, Any],
	test_metrics: Dict[str, Any],
) -> Dict[str, Any]:
	task_cfg = TASK_CONFIG[task_type]
	return to_builtin({
		"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
		"task_type": task_type,
		"display_name": task_cfg["display_name"],
		"description": "Latest evaluated candidate. If accepted, copy the numbers into training/configs/main_models/baselines.json with a short description of the change.",
		"comparison_metric": task_cfg["comparison_metric"],
		"training_config_source": str(task_cfg["config_path"].relative_to(PROJECT_ROOT)),
		"epoch_selection_season": epoch_selection_season,
		"held_out_test_season": test_season,
		"best_epoch": best_epoch,
		"best_val_loss": float(best_val_loss),
		"val_metrics": summarize_metrics(task_type, validation_metrics),
		"test_metrics": summarize_metrics(task_type, test_metrics),
	})


def build_train_config(
	training_config: Dict[str, Any],
	input_dim: int,
	cat_config: CategoricalConfig,
	epochs: int,
) -> TrainConfig:
	return TrainConfig(
		input_dim=input_dim,
		hidden_layers=training_config["hidden_layers"],
		dropout=training_config["dropout"],
		norm=training_config["norm"],
		lr=training_config["lr"],
		weight_decay=training_config["weight_decay"],
		activation=training_config["activation"],
		beta1=training_config["beta1"],
		epochs=epochs,
		patience=training_config["patience"],
		batch_size=training_config["batch_size"],
		task_type=training_config["task_type"],
		cat_config=cat_config,
		gate_hidden_dim=training_config["gate_hidden_dim"],
		gate_target_budget=training_config["gate_target_budget"],
		gate_mean_weight=training_config["gate_mean_weight"],
		gate_sat_weight=training_config["gate_sat_weight"],
		lambda_repulsion=training_config.get("lambda_repulsion", 0.0),
		lambda_corr=training_config.get("lambda_corr", 0.0),
	)


def save_model_bundle(
	task_type: TaskType,
	model: Any,
	scaler: Any,
	train_config: TrainConfig,
	training_config: Dict[str, Any],
	feature_cols: list[str],
	validation_metrics: Dict[str, Any],
	validation_baseline_metrics: Dict[str, Any],
	test_metrics: Dict[str, Any],
	test_baseline_metrics: Dict[str, Any],
	all_cv_seasons: list[str],
	final_val_season: str,
	test_season: str,
	best_epoch: int,
	best_val_loss: float,
):
	task_cfg = TASK_CONFIG[task_type]
	artifact_model_path = task_cfg["artifact_model_path"]
	artifact_config_path = task_cfg["artifact_config_path"]
	artifact_scaler_path = task_cfg["artifact_scaler_path"]
	artifact_model_path.parent.mkdir(parents=True, exist_ok=True)

	torch.save(model.state_dict(), artifact_model_path)
	joblib.dump(scaler, artifact_scaler_path)

	cat_cfg = None
	if train_config.cat_config is not None:
		cat_cfg = {
			"num_leagues": train_config.cat_config.num_leagues,
			"league_embed_dim": train_config.cat_config.league_embed_dim,
		}

	metadata = {
		"task_type": task_type,
		"model_family": "main",
		"display_name": task_cfg["display_name"],
		"comparison_metric": task_cfg["comparison_metric"],
		"training_config_source": str(task_cfg["config_path"].relative_to(PROJECT_ROOT)),
		"input_dim": train_config.input_dim,
		"hidden_layers": train_config.hidden_layers,
		"activation": train_config.activation,
		"norm": train_config.norm,
		"dropout": train_config.dropout,
		"lr": train_config.lr,
		"weight_decay": train_config.weight_decay,
		"beta1": train_config.beta1,
		"batch_size": train_config.batch_size,
		"final_epochs": best_epoch,
		"feature_cols": feature_cols,
		"output_dim": 1 if task_type == "binary" else 3,
		"cat_config": cat_cfg,
		"gate_hidden_dim": train_config.gate_hidden_dim,
		"gate_target_budget": train_config.gate_target_budget,
		"gate_mean_weight": train_config.gate_mean_weight,
		"gate_sat_weight": train_config.gate_sat_weight,
		"lambda_repulsion": train_config.lambda_repulsion,
		"lambda_corr": train_config.lambda_corr,
		"evaluation_protocol": {
			"cv_strategy": "rolling_origin_expanding_window",
			"n_cv_folds": N_CV_FOLDS,
			"selection_metric": task_cfg["comparison_metric"],
			"cv_seasons": all_cv_seasons,
			"epoch_selection_season": final_val_season,
			"held_out_test_season": test_season,
			"training_seed": TRAINING_SEED,
		},
		"selection_summary": {
			"best_epoch": best_epoch,
			"best_val_loss": float(best_val_loss),
			"epoch_selection_season": final_val_season,
		},
		"validation_metrics": validation_metrics,
		"validation_baseline_metrics": validation_baseline_metrics,
		"test_metrics": test_metrics,
		"test_baseline_metrics": test_baseline_metrics,
		"frozen_training_config": training_config,
	}

	with open(artifact_config_path, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)

	mlflow.log_artifact(str(artifact_model_path))
	mlflow.log_artifact(str(artifact_scaler_path))
	mlflow.log_artifact(str(artifact_config_path))



def train_task(task_type: TaskType) -> Dict[str, Any]:
	if task_type not in TASK_CONFIG:
		raise ValueError(f"Unknown TASK_TYPE={task_type}")

	task_cfg = TASK_CONFIG[task_type]
	training_config = load_training_config(task_type)
	prepare_fn = task_cfg["prepare_fn"]

	print_header(f"TRAIN MAIN MODEL: {task_cfg['display_name']}")
	print(f"Device: {DEVICE}")
	print(f"Training config: {task_cfg['config_path']}")

	set_seed(TRAINING_SEED, deterministic=False)

	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = task_cfg["add_targets_fn"](df)
	df = df.drop_nulls(subset=task_cfg["odds_cols"])
	print(f"Total rows with odds: {len(df)}")

	feature_cols = select_feature_columns(df)
	print(f"Features: {len(feature_cols)}")

	num_leagues = get_num_leagues(df)
	cat_config = CategoricalConfig(num_leagues=num_leagues, league_embed_dim=3)
	print(f"Categorical: {num_leagues} leagues (embed_dim=3)")

	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Held-out test season: {test_season}")

	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})
	epoch_selection_season = all_cv_seasons[-1]
	initial_train_seasons = all_cv_seasons[:-1]

	print(f"Step 1: train on {initial_train_seasons[0]}..{initial_train_seasons[-1]} | epoch selection on {epoch_selection_season}")
	print(f"Step 2: retrain on {all_cv_seasons[0]}..{all_cv_seasons[-1]} | test on {test_season}")

	mlflow.set_experiment(task_cfg["experiment_name"])
	with mlflow.start_run(run_name=f"{task_cfg['label']}_main"):
		mlflow.log_params({
			"task": task_type,
			"training_config": str(task_cfg["config_path"].relative_to(PROJECT_ROOT)),
			"parquet_path": str(DEFAULT_PARQUET),
			"n_cv_folds": N_CV_FOLDS,
			"held_out_test_season": test_season,
		})

		set_seed(TRAINING_SEED, deterministic=True)
		data_initial_train = prepare_fn(df, feature_cols, initial_train_seasons, fit_scaler=True)
		data_final_val = prepare_fn(df, feature_cols, [epoch_selection_season], scaler=data_initial_train["scaler"])

		initial_train_loader = to_loader(
			data_initial_train,
			training_config["batch_size"],
			device=DEVICE,
			task_type=task_type,
		)
		final_val_loader = to_loader(
			data_final_val,
			training_config["batch_size"],
			shuffle=False,
			device=DEVICE,
			task_type=task_type,
		)

		early_stop_config = build_train_config(
			training_config,
			input_dim=data_initial_train["X"].shape[1],
			cat_config=cat_config,
			epochs=training_config["max_epochs"],
		)
		early_stop_model, early_stop_history, best_val_loss = train_model(
			early_stop_config,
			initial_train_loader,
			final_val_loader,
			device=DEVICE,
			verbose=True,
		)
		best_epoch = early_stop_history["val_loss"].index(min(early_stop_history["val_loss"])) + 1
		print(f"Early stopping best epoch: {best_epoch} (val_loss={best_val_loss:.5f})")

		print("\n--- Early-stop Model Performance on Epoch-selection Season ---")
		validation_baseline_metrics = evaluate_implied_baseline(data_final_val, task_type=task_type)
		log_metric_group("val_baseline", validation_baseline_metrics)
		validation_metrics = evaluate_model(
			early_stop_model,
			data_final_val,
			device=DEVICE,
			verbose=True,
			task_type=task_type,
		)
		log_metric_group("val", validation_metrics)
		mlflow.log_metric("best_val_loss", float(best_val_loss))
		mlflow.log_metric("best_epoch", int(best_epoch))

		data_train = prepare_fn(df, feature_cols, all_cv_seasons, fit_scaler=True)
		data_test = prepare_fn(df, feature_cols, [test_season], scaler=data_train["scaler"])
		train_loader = to_loader(
			data_train,
			training_config["batch_size"],
			device=DEVICE,
			task_type=task_type,
		)
		final_config = build_train_config(
			training_config,
			input_dim=data_train["X"].shape[1],
			cat_config=cat_config,
			epochs=best_epoch,
		)

		test_baseline_metrics = evaluate_implied_baseline(data_test, task_type=task_type)
		log_metric_group("test_baseline", test_baseline_metrics)

		print("\n--- Training Final Model ---")
		model, _, _ = train_model(final_config, train_loader, val_loader=None, device=DEVICE, verbose=True)

		print("\n--- Model Performance on Held-out Test Set ---")
		test_metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True, task_type=task_type)
		log_metric_group("test", test_metrics)

		run_record = build_latest_run_record(
			task_type=task_type,
			epoch_selection_season=epoch_selection_season,
			test_season=test_season,
			best_epoch=best_epoch,
			best_val_loss=best_val_loss,
			validation_metrics=validation_metrics,
			test_metrics=test_metrics,
		)

		update_latest_run_record(task_type, run_record)

		save_model_bundle(
			task_type=task_type,
			model=model,
			scaler=data_train["scaler"],
			train_config=final_config,
			training_config=training_config,
			feature_cols=feature_cols,
			validation_metrics=validation_metrics,
			validation_baseline_metrics=validation_baseline_metrics,
			test_metrics=test_metrics,
			test_baseline_metrics=test_baseline_metrics,
			all_cv_seasons=all_cv_seasons,
			final_val_season=epoch_selection_season,
			test_season=test_season,
			best_epoch=best_epoch,
			best_val_loss=best_val_loss,
		)

	print_header("DONE")
	print(f"Best validation loss: {best_val_loss:.5f}")
	print(f"Validation metrics: {run_record['val_metrics']}")
	print(f"Test metrics: {run_record['test_metrics']}")
	return run_record



def main():
	train_task(TASK_TYPE)


if __name__ == "__main__":
	main()
