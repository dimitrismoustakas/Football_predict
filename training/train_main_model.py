"""
Canonical training entry point for the production match-result model.

The evaluation loop is fixed:
- frozen rolling CV folds for model selection
- fixed epoch-selection season for epoch selection
- fixed held-out latest season for acceptance
"""

import json
import os
import random
import subprocess
import sys
from csv import DictWriter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from training.evaluation import evaluate_model
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, save_model_bundle as save_bundle
from training.models import CategoricalConfig, TrainConfig
from training.training_loop import train_fixed_epochs, train_with_early_stopping
from training.train_utils import (
	add_targets_and_implied,
	evaluate_implied_baseline,
	filter_min_history,
	generate_rolling_cv_folds,
	get_num_leagues,
	get_test_season,
	load_frame,
	prepare_data,
	select_feature_columns,
	to_loader,
)
from utils.paths import EXPERIMENT_METRICS_DIR, MODELS_DIR, PROJECT_ROOT

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
LATEST_MAIN_METRICS_PATH = MODELS_DIR / "latest_main_model_metrics.json"
EXPERIMENT_LOG_PATH = EXPERIMENT_METRICS_DIR / "result_main_runs.tsv"
N_CV_FOLDS = int(os.environ.get("N_CV_FOLDS", "3"))
TRAINING_SEED = int(os.environ.get("TRAINING_SEED", "42"))

DISPLAY_NAME = "Match Result"
COMPARISON_METRIC = "log_loss"
MODEL_NAME = "gated_residual"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "result.json"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_LOG_COLUMNS = [
	"recorded_at_utc",
	"display_name",
	"recipe_label",
	"model_name",
	"comparison_metric",
	"objective_name",
	"objective_value",
	"objective_fold_count",
	"objective_val_seasons",
	"best_epoch",
	"final_train_epochs",
	"epoch_selection_season",
	"val_log_loss",
	"val_rps",
	"val_brier",
	"held_out_test_season",
	"test_log_loss",
	"test_rps",
	"test_brier",
	"git_branch",
	"git_commit",
	"training_config_source",
	"feature_manifest_source",
	"cv_metrics_json",
	"cv_fold_metrics_json",
	"val_metrics_json",
	"test_metrics_json",
]


def set_seed(seed: int = 42, deterministic: bool = False):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(deterministic)
	if deterministic:
		os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:
		os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True


def print_header(text: str):
	print("\n" + "=" * 60)
	print(text)
	print("=" * 60)


def summarize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
	keys = ["accuracy", "brier", "rps", "log_loss", "total_profit", "avg_profit", "n_bets"]
	return {key: metrics[key] for key in keys if key in metrics}


def write_json(path: Path, payload: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2)


def append_tsv_row(path: Path, row: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	serialized = {}
	for key in EXPERIMENT_LOG_COLUMNS:
		value = row.get(key, "")
		if isinstance(value, (dict, list)):
			serialized[key] = json.dumps(value, separators=(",", ":"), sort_keys=True)
		elif value is None:
			serialized[key] = ""
		else:
			serialized[key] = value
	write_header = not path.exists() or path.stat().st_size == 0
	with open(path, "a", encoding="utf-8", newline="") as file:
		writer = DictWriter(file, fieldnames=EXPERIMENT_LOG_COLUMNS, delimiter="\t")
		if write_header:
			writer.writeheader()
		writer.writerow(serialized)


def load_training_config() -> dict:
	with open(TRAINING_CONFIG_PATH, "r", encoding="utf-8") as file:
		return json.load(file)


def build_train_config(
	training_config: dict,
	input_dim: int,
	cat_config: CategoricalConfig,
	epochs: int,
) -> TrainConfig:
	model_kwargs = {
		"hidden_layers": training_config["hidden_layers"],
		"dropout": training_config["dropout"],
		"norm": training_config["norm"],
		"activation": training_config["activation"],
		"gate_hidden_dim": training_config["gate_hidden_dim"],
		"gate_target_budget": training_config["gate_target_budget"],
	}
	return TrainConfig(
		input_dim=input_dim,
		model_kwargs=model_kwargs,
		lr=training_config["lr"],
		weight_decay=training_config["weight_decay"],
		beta1=training_config["beta1"],
		beta2=0.999,
		optimizer_eps=1e-8,
		epochs=epochs,
		patience=training_config["patience"],
		batch_size=training_config["batch_size"],
		cat_config=cat_config,
		scheduler_min_lr_ratio=0.01,
		gate_mean_weight=training_config["gate_mean_weight"],
		gate_sat_weight=training_config["gate_sat_weight"],
		lambda_repulsion=training_config.get("lambda_repulsion", 0.0),
		lambda_corr=training_config.get("lambda_corr", 0.0),
	)


def build_bundle_metadata(
	training_config: dict,
	cat_config: CategoricalConfig,
	feature_cols: list[str],
	objective_metrics: dict,
	objective_baseline_metrics: dict,
	objective_fold_metrics: list[dict],
	validation_metrics: dict,
	validation_baseline_metrics: dict,
	test_metrics: dict,
	test_baseline_metrics: dict,
	all_cv_seasons: list[str],
	objective_val_seasons: list[str],
	final_val_season: str,
	test_season: str,
	n_cv_folds: int,
	training_seed: int,
	best_epoch: int,
	final_train_epochs: int,
	best_val_loss: float,
) -> dict:
	return {
		"display_name": DISPLAY_NAME,
		"model_name": MODEL_NAME,
		"model_kwargs": {
			"hidden_layers": training_config["hidden_layers"],
			"dropout": training_config["dropout"],
			"norm": training_config["norm"],
			"activation": training_config["activation"],
			"gate_hidden_dim": training_config["gate_hidden_dim"],
			"gate_target_budget": training_config["gate_target_budget"],
		},
		"feature_cols": feature_cols,
		"cat_config": {
			"num_leagues": cat_config.num_leagues,
			"league_embed_dim": cat_config.league_embed_dim,
		},
		"final_epochs": final_train_epochs,
		"final_epoch_mode": "best",
		"evaluation_protocol": {
			"cv_strategy": "rolling_origin_expanding_window",
			"n_cv_folds": n_cv_folds,
			"objective_fold_count": len(objective_fold_metrics),
			"selection_metric": COMPARISON_METRIC,
			"cv_seasons": all_cv_seasons,
			"objective_val_seasons": objective_val_seasons,
			"epoch_selection_season": final_val_season,
			"held_out_test_season": test_season,
			"training_seed": training_seed,
		},
		"selection_summary": {
			"objective_metrics": objective_metrics,
			"objective_baseline_metrics": objective_baseline_metrics,
			"objective_fold_metrics": objective_fold_metrics,
			"best_epoch": best_epoch,
			"final_train_epochs": final_train_epochs,
			"best_val_loss": float(best_val_loss),
			"epoch_selection_season": final_val_season,
		},
		"validation_metrics": validation_metrics,
		"validation_baseline_metrics": validation_baseline_metrics,
		"test_metrics": test_metrics,
		"test_baseline_metrics": test_baseline_metrics,
	}


def mean_metric(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
	if not metrics_list:
		return {}
	keys = metrics_list[0].keys()
	return {
		key: float(np.mean([metrics[key] for metrics in metrics_list]))
		for key in keys
		if all(isinstance(metrics.get(key), (int, float)) for metrics in metrics_list)
	}


def get_git_value(*args: str) -> str:
	try:
		result = subprocess.run(
			["git", *args],
			cwd=PROJECT_ROOT,
			capture_output=True,
			text=True,
			check=True,
		)
		return result.stdout.strip()
	except Exception:
		return ""


def get_git_metadata() -> dict:
	return {
		"git_branch": get_git_value("rev-parse", "--abbrev-ref", "HEAD"),
		"git_commit": get_git_value("rev-parse", "HEAD"),
	}


def split_selection_folds(folds: list[tuple[list[str], str]]) -> tuple[list[tuple[list[str], str]], tuple[list[str], str]]:
	if len(folds) < 2:
		raise ValueError("Need at least 2 rolling folds: one for CV objective and one for epoch selection.")
	return folds[:-1], folds[-1]


def describe_training_split(
	objective_folds: list[tuple[list[str], str]],
	epoch_train_seasons: list[str],
	epoch_selection_season: str,
	all_cv_seasons: list[str],
	test_season: str,
):
	for fold_idx, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(f"Objective fold {fold_idx}: train on {train_seasons[0]}..{train_seasons[-1]} | validate on {val_season}")
	print(
		f"Epoch selection: train on {epoch_train_seasons[0]}..{epoch_train_seasons[-1]} | validate on {epoch_selection_season}"
	)
	print(f"Final acceptance: train on {all_cv_seasons[0]}..{all_cv_seasons[-1]} | test on {test_season}")


def prepare_phase_loaders(
	df,
	feature_cols: list[str],
	batch_size: int,
	train_seasons: list[str],
	eval_seasons: list[str],
	train_seed: int,
):
	train_data = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	eval_data = prepare_data(df, feature_cols, eval_seasons, scaler=train_data["scaler"])
	train_loader = to_loader(train_data, batch_size, device=DEVICE, seed=train_seed)
	eval_loader = to_loader(eval_data, batch_size, shuffle=False, device=DEVICE)
	return train_data, eval_data, train_loader, eval_loader


def resolve_final_training_epochs(max_epochs: int, best_epoch: int) -> int:
	return max(1, min(max_epochs, int(best_epoch)))


def evaluate_cv_objective(
	df,
	feature_cols: list[str],
	training_config: dict,
	cat_config: CategoricalConfig,
	objective_folds: list[tuple[list[str], str]],
	final_train_epochs: int,
) -> tuple[list[dict], Dict[str, float], Dict[str, float]]:
	fold_metrics = []
	fold_baseline_metrics = []
	for fold_idx, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(
			f"\n--- CV Objective Fold {fold_idx}/{len(objective_folds)}: {train_seasons[0]}..{train_seasons[-1]} -> {val_season} ---"
		)
		set_seed(TRAINING_SEED, deterministic=True)
		data_train, data_val, train_loader, _ = prepare_phase_loaders(
			df,
			feature_cols,
			training_config["batch_size"],
			train_seasons,
			[val_season],
			TRAINING_SEED + fold_idx,
		)
		fold_config = build_train_config(training_config, data_train["X"].shape[1], cat_config, final_train_epochs)
		fold_model, _, _ = train_fixed_epochs(fold_config, train_loader, device=DEVICE, verbose=True)
		baseline_metrics = summarize_metrics(evaluate_implied_baseline(data_val))
		metrics = summarize_metrics(evaluate_model(fold_model, data_val, device=DEVICE, verbose=True))
		fold_baseline_metrics.append(baseline_metrics)
		fold_metrics.append({
			"fold_index": fold_idx,
			"train_start_season": train_seasons[0],
			"train_end_season": train_seasons[-1],
			"val_season": val_season,
			"baseline_metrics": baseline_metrics,
			"metrics": metrics,
		})
	mean_fold_metrics = mean_metric([fold["metrics"] for fold in fold_metrics])
	mean_baseline_metrics = mean_metric(fold_baseline_metrics)
	return fold_metrics, mean_fold_metrics, mean_baseline_metrics


def train_main_model() -> dict:
	training_config = load_training_config()

	print_header(f"TRAIN MAIN MODEL: {DISPLAY_NAME}")
	print(f"Device: {DEVICE}")
	print(f"Training config: {TRAINING_CONFIG_PATH}")

	set_seed(TRAINING_SEED, deterministic=True)
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	print(f"Total rows with odds: {len(df)}")

	feature_cols = select_feature_columns(df, FEATURE_MANIFEST_PATH)
	print(f"Features: {len(feature_cols)}")
	cat_config = CategoricalConfig(num_leagues=get_num_leagues(df), league_embed_dim=3)
	print(f"Categorical: {cat_config.num_leagues} leagues (embed_dim=3)")

	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Held-out test season: {test_season}")

	objective_folds, epoch_fold = split_selection_folds(folds)
	epoch_train_seasons, epoch_selection_season = epoch_fold
	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})
	objective_val_seasons = [val_season for _, val_season in objective_folds]
	describe_training_split(objective_folds, epoch_train_seasons, epoch_selection_season, all_cv_seasons, test_season)
	git_metadata = get_git_metadata()

	set_seed(TRAINING_SEED, deterministic=True)
	data_initial_train, data_final_val, initial_train_loader, final_val_loader = prepare_phase_loaders(
		df,
		feature_cols,
		training_config["batch_size"],
		epoch_train_seasons,
		[epoch_selection_season],
		TRAINING_SEED,
	)

	early_stop_config = build_train_config(training_config, data_initial_train["X"].shape[1], cat_config, training_config["max_epochs"])
	early_stop_model, early_stop_history, best_val_loss = train_with_early_stopping(
		early_stop_config,
		initial_train_loader,
		final_val_loader,
		device=DEVICE,
		verbose=True,
	)
	best_epoch = early_stop_history["val_loss"].index(min(early_stop_history["val_loss"])) + 1
	print(f"Early stopping best epoch: {best_epoch} (val_loss={best_val_loss:.5f})")
	final_train_epochs = resolve_final_training_epochs(training_config["max_epochs"], best_epoch)
	print(f"Final retrain epochs: {final_train_epochs} (mode=best)")

	cv_fold_metrics, cv_metrics, cv_baseline_metrics = evaluate_cv_objective(
		df,
		feature_cols,
		training_config,
		cat_config,
		objective_folds,
		final_train_epochs,
	)
	objective_value = float(cv_metrics[COMPARISON_METRIC])
	print(f"\nCV objective ({COMPARISON_METRIC}): {objective_value:.5f}")

	print("\n--- Early-stop Model Performance on Epoch-selection Season ---")
	validation_baseline_metrics = evaluate_implied_baseline(data_final_val)
	validation_metrics = evaluate_model(early_stop_model, data_final_val, device=DEVICE, verbose=True)

	data_train, data_test, train_loader, _ = prepare_phase_loaders(
		df,
		feature_cols,
		training_config["batch_size"],
		all_cv_seasons,
		[test_season],
		TRAINING_SEED + 10_000,
	)
	final_config = build_train_config(training_config, data_train["X"].shape[1], cat_config, final_train_epochs)

	test_baseline_metrics = evaluate_implied_baseline(data_test)

	print("\n--- Training Final Model ---")
	model, _, _ = train_fixed_epochs(final_config, train_loader, device=DEVICE, verbose=True)

	print("\n--- Model Performance on Held-out Test Set ---")
	test_metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True)

	run_record = {
		"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
		"display_name": DISPLAY_NAME,
		"comparison_metric": COMPARISON_METRIC,
		"objective_name": f"cv_mean_{COMPARISON_METRIC}",
		"objective_value": objective_value,
		"training_config_source": str(TRAINING_CONFIG_PATH.relative_to(PROJECT_ROOT)),
		"feature_manifest_source": str(FEATURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
		"model_name": MODEL_NAME,
		"objective_fold_count": len(objective_folds),
		"objective_val_seasons": objective_val_seasons,
		"objective_metrics": cv_metrics,
		"objective_baseline_metrics": cv_baseline_metrics,
		"objective_fold_metrics": cv_fold_metrics,
		"epoch_selection_season": epoch_selection_season,
		"held_out_test_season": test_season,
		"best_epoch": best_epoch,
		"best_val_loss": float(best_val_loss),
		"val_metrics": summarize_metrics(validation_metrics),
		"test_metrics": summarize_metrics(test_metrics),
		**git_metadata,
	}
	write_json(
		LATEST_MAIN_METRICS_PATH,
		{
			"schema_version": 1,
			"description": "Latest evaluated match-result candidate. Runtime-generated; compare with training/configs/main_models/baselines.json.",
			"model": run_record,
		},
	)

	bundle_metadata = build_bundle_metadata(
		training_config=training_config,
		cat_config=cat_config,
		feature_cols=feature_cols,
		objective_metrics=cv_metrics,
		objective_baseline_metrics=cv_baseline_metrics,
		objective_fold_metrics=cv_fold_metrics,
		validation_metrics=summarize_metrics(validation_metrics),
		validation_baseline_metrics=summarize_metrics(validation_baseline_metrics),
		test_metrics=summarize_metrics(test_metrics),
		test_baseline_metrics=summarize_metrics(test_baseline_metrics),
		all_cv_seasons=all_cv_seasons,
		objective_val_seasons=objective_val_seasons,
		final_val_season=epoch_selection_season,
		test_season=test_season,
		n_cv_folds=N_CV_FOLDS,
		training_seed=TRAINING_SEED,
		best_epoch=best_epoch,
		final_train_epochs=final_train_epochs,
		best_val_loss=best_val_loss,
	)
	save_bundle(RESULT_MODEL_BUNDLE_PATHS, model, data_train["scaler"], bundle_metadata)
	append_tsv_row(
		EXPERIMENT_LOG_PATH,
		{
			"recorded_at_utc": run_record["recorded_at_utc"],
			"display_name": DISPLAY_NAME,
			"recipe_label": "result",
			"model_name": MODEL_NAME,
			"comparison_metric": COMPARISON_METRIC,
			"objective_name": run_record["objective_name"],
			"objective_value": run_record["objective_value"],
			"objective_fold_count": run_record["objective_fold_count"],
			"objective_val_seasons": run_record["objective_val_seasons"],
			"best_epoch": best_epoch,
			"final_train_epochs": final_train_epochs,
			"epoch_selection_season": epoch_selection_season,
			"val_log_loss": run_record["val_metrics"].get("log_loss"),
			"val_rps": run_record["val_metrics"].get("rps"),
			"val_brier": run_record["val_metrics"].get("brier"),
			"held_out_test_season": test_season,
			"test_log_loss": run_record["test_metrics"].get("log_loss"),
			"test_rps": run_record["test_metrics"].get("rps"),
			"test_brier": run_record["test_metrics"].get("brier"),
			"git_branch": git_metadata["git_branch"],
			"git_commit": git_metadata["git_commit"],
			"training_config_source": run_record["training_config_source"],
			"feature_manifest_source": run_record["feature_manifest_source"],
			"cv_metrics_json": run_record["objective_metrics"],
			"cv_fold_metrics_json": run_record["objective_fold_metrics"],
			"val_metrics_json": run_record["val_metrics"],
			"test_metrics_json": run_record["test_metrics"],
		},
	)

	print_header("DONE")
	print(f"CV objective ({COMPARISON_METRIC}): {run_record['objective_value']:.5f}")
	print(f"Best validation loss: {best_val_loss:.5f}")
	print(f"Validation metrics: {run_record['val_metrics']}")
	print(f"Test metrics: {run_record['test_metrics']}")
	print(f"Experiment log: {EXPERIMENT_LOG_PATH}")
	return run_record


def main():
	train_main_model()


if __name__ == "__main__":
	main()
