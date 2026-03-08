"""
Canonical training entry point for the production match-result model.

The evaluation loop is fixed by source-controlled config:
- frozen rolling CV folds for model selection
- fixed epoch-selection season for epoch selection
- fixed held-out watch-only test season for monitoring
"""

import json
import os
import random
import subprocess
import sys
from csv import DictReader, DictWriter
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
	build_data_snapshot,
	evaluate_implied_baseline,
	filter_min_history,
	generate_rolling_cv_folds,
	get_num_leagues,
	load_frame,
	prepare_data,
	resolve_test_season,
	select_feature_columns,
	to_loader,
)
from utils.paths import EXPERIMENT_METRICS_DIR, MODELS_DIR, PROJECT_ROOT

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
LATEST_MAIN_METRICS_PATH = MODELS_DIR / "latest_main_model_metrics.json"
EXPERIMENT_LOG_PATH = EXPERIMENT_METRICS_DIR / "result_main_runs.tsv"
DISPLAY_NAME = "Match Result"
MODEL_NAME = "gated_residual"
TRAINING_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "result.json"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json"
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "evaluation.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LESS_IS_BETTER_METRICS = {"log_loss", "rps", "brier"}

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
	"test_role",
	"min_objective_improvement",
	"reference_status",
	"reference_objective_value",
	"objective_delta",
	"meets_objective_threshold",
	"data_fingerprint",
	"season_row_counts_json",
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
	"evaluation_config_source",
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
	serialized = {key: serialize_log_value(row.get(key, "")) for key in EXPERIMENT_LOG_COLUMNS}
	write_header = not path.exists() or path.stat().st_size == 0
	with open(path, "a", encoding="utf-8", newline="") as file:
		writer = DictWriter(file, fieldnames=EXPERIMENT_LOG_COLUMNS, delimiter="\t")
		if write_header:
			writer.writeheader()
		writer.writerow(serialized)


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def serialize_log_value(value):
	if isinstance(value, (dict, list)):
		return json.dumps(value, separators=(",", ":"), sort_keys=True)
	if value is None:
		return ""
	return value


def load_training_config() -> dict:
	return load_json(TRAINING_CONFIG_PATH)


def load_evaluation_config() -> dict:
	raw = load_json(EVALUATION_CONFIG_PATH)
	required = [
		"comparison_metric",
		"training_seed",
		"rolling_cv_n_folds",
		"min_objective_improvement",
		"test_season",
		"test_role",
	]
	missing = [key for key in required if key not in raw]
	if missing:
		raise ValueError(f"Missing evaluation config keys: {missing}")
	config = dict(raw)
	config["comparison_metric"] = str(config["comparison_metric"])
	config["training_seed"] = int(config["training_seed"])
	config["rolling_cv_n_folds"] = int(config["rolling_cv_n_folds"])
	config["min_objective_improvement"] = float(config["min_objective_improvement"])
	config["test_season"] = str(config["test_season"])
	config["test_role"] = str(config["test_role"])
	if config["test_role"] not in {"watch_only", "acceptance"}:
		raise ValueError(f"Unsupported test_role: {config['test_role']}")
	return config


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
		"shared_gate": training_config.get("shared_gate", False),
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
	evaluation_config: dict,
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
	best_epoch: int,
	final_train_epochs: int,
	best_val_loss: float,
	data_snapshot: dict,
	reference_comparison: dict,
) -> dict:
	return {
		"display_name": DISPLAY_NAME,
		"model_name": MODEL_NAME,
		"comparison_metric": evaluation_config["comparison_metric"],
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
			"comparison_metric": evaluation_config["comparison_metric"],
			"cv_strategy": "rolling_origin_expanding_window",
			"rolling_cv_n_folds": evaluation_config["rolling_cv_n_folds"],
			"objective_fold_count": len(objective_fold_metrics),
			"cv_seasons": all_cv_seasons,
			"objective_val_seasons": objective_val_seasons,
			"epoch_selection_season": final_val_season,
			"test_season": test_season,
			"test_role": evaluation_config["test_role"],
			"min_objective_improvement": evaluation_config["min_objective_improvement"],
			"training_seed": evaluation_config["training_seed"],
		},
		"selection_summary": {
			"objective_metrics": objective_metrics,
			"objective_baseline_metrics": objective_baseline_metrics,
			"objective_fold_metrics": objective_fold_metrics,
			"best_epoch": best_epoch,
			"final_train_epochs": final_train_epochs,
			"best_val_loss": float(best_val_loss),
			"epoch_selection_season": final_val_season,
			"reference_comparison": reference_comparison,
		},
		"data_snapshot": data_snapshot,
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
	test_role: str,
):
	for fold_idx, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(f"Objective fold {fold_idx}: train on {train_seasons[0]}..{train_seasons[-1]} | validate on {val_season}")
	print(
		f"Epoch selection: train on {epoch_train_seasons[0]}..{epoch_train_seasons[-1]} | validate on {epoch_selection_season}"
	)
	print(f"Fixed test season ({test_role}): train on {all_cv_seasons[0]}..{all_cv_seasons[-1]} | test on {test_season}")


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


def metric_improvement(candidate_value: float, reference_value: float, metric_name: str) -> float:
	if metric_name in LESS_IS_BETTER_METRICS:
		return reference_value - candidate_value
	return candidate_value - reference_value


def load_experiment_rows(path: Path | str) -> list[dict]:
	path = Path(path)
	if not path.exists() or path.stat().st_size == 0:
		return []
	with open(path, "r", encoding="utf-8", newline="") as file:
		reader = DictReader(file, delimiter="\t")
		return [row for row in reader]


def get_latest_comparable_reference(
	path: Path,
	comparison_metric: str,
	objective_fold_count: int,
	objective_val_seasons: list[str],
	test_season: str,
	test_role: str,
	data_fingerprint: str,
) -> dict | None:
	expected_val_seasons = serialize_log_value(objective_val_seasons)
	for row in reversed(load_experiment_rows(path)):
		if row.get("comparison_metric") != comparison_metric:
			continue
		if row.get("objective_fold_count") != str(objective_fold_count):
			continue
		if row.get("objective_val_seasons") != expected_val_seasons:
			continue
		if row.get("held_out_test_season") != test_season:
			continue
		if row.get("test_role") != test_role:
			continue
		if row.get("data_fingerprint") != data_fingerprint:
			continue
		return row
	return None


def build_reference_comparison(
	candidate_objective: float,
	evaluation_config: dict,
	reference_row: dict | None,
) -> dict:
	comparison = {
		"status": "no_reference",
		"reference_objective_value": None,
		"objective_delta": None,
		"min_objective_improvement": evaluation_config["min_objective_improvement"],
		"meets_objective_threshold": None,
	}
	if reference_row is None:
		return comparison

	reference_objective = reference_row.get("objective_value")
	if reference_objective in (None, ""):
		comparison["status"] = "reference_missing_objective"
		return comparison

	reference_value = float(reference_objective)
	comparison["reference_objective_value"] = reference_value
	comparison["objective_delta"] = metric_improvement(
		candidate_value=candidate_objective,
		reference_value=reference_value,
		metric_name=evaluation_config["comparison_metric"],
	)
	comparison["meets_objective_threshold"] = comparison["objective_delta"] >= evaluation_config["min_objective_improvement"]
	comparison["status"] = "candidate_clears_threshold" if comparison["meets_objective_threshold"] else "candidate_below_threshold"
	return comparison


def print_data_snapshot(data_snapshot: dict):
	print(f"Data fingerprint: {data_snapshot['data_fingerprint']}")
	print(f"Season rows: {data_snapshot['season_row_counts']}")


def print_reference_comparison(reference_comparison: dict, comparison_metric: str):
	status = reference_comparison["status"]
	delta = reference_comparison.get("objective_delta")
	if delta is None:
		print(f"Reference comparison: {status}")
		return
	print(
		f"Reference comparison: {status} | delta_{comparison_metric}={delta:.6f} | threshold={reference_comparison['min_objective_improvement']:.6f}"
	)


def evaluate_cv_objective(
	df,
	feature_cols: list[str],
	training_config: dict,
	cat_config: CategoricalConfig,
	objective_folds: list[tuple[list[str], str]],
	final_train_epochs: int,
	training_seed: int,
) -> tuple[list[dict], Dict[str, float], Dict[str, float]]:
	fold_metrics = []
	fold_baseline_metrics = []
	for fold_idx, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(
			f"\n--- CV Objective Fold {fold_idx}/{len(objective_folds)}: {train_seasons[0]}..{train_seasons[-1]} -> {val_season} ---"
		)
		set_seed(training_seed, deterministic=True)
		data_train, data_val, train_loader, _ = prepare_phase_loaders(
			df,
			feature_cols,
			training_config["batch_size"],
			train_seasons,
			[val_season],
			training_seed + fold_idx,
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
	evaluation_config = load_evaluation_config()
	comparison_metric = evaluation_config["comparison_metric"]
	training_seed = evaluation_config["training_seed"]

	print_header(f"TRAIN MAIN MODEL: {DISPLAY_NAME}")
	print(f"Device: {DEVICE}")
	print(f"Training config: {TRAINING_CONFIG_PATH}")
	print(f"Evaluation config: {EVALUATION_CONFIG_PATH}")

	set_seed(training_seed, deterministic=True)
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

	test_season = resolve_test_season(df, evaluation_config["test_season"])
	data_snapshot = build_data_snapshot(df, test_season)
	print_data_snapshot(data_snapshot)

	print(f"\nGenerating {evaluation_config['rolling_cv_n_folds']}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=evaluation_config["rolling_cv_n_folds"], test_season=test_season)
	print(f"Fixed held-out test season: {test_season} ({evaluation_config['test_role']})")

	objective_folds, epoch_fold = split_selection_folds(folds)
	epoch_train_seasons, epoch_selection_season = epoch_fold
	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})
	objective_val_seasons = [val_season for _, val_season in objective_folds]
	describe_training_split(
		objective_folds,
		epoch_train_seasons,
		epoch_selection_season,
		all_cv_seasons,
		test_season,
		evaluation_config["test_role"],
	)
	git_metadata = get_git_metadata()

	set_seed(training_seed, deterministic=True)
	data_initial_train, data_final_val, initial_train_loader, final_val_loader = prepare_phase_loaders(
		df,
		feature_cols,
		training_config["batch_size"],
		epoch_train_seasons,
		[epoch_selection_season],
		training_seed,
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
		training_seed,
	)
	objective_value = float(cv_metrics[comparison_metric])
	print(f"\nCV objective ({comparison_metric}): {objective_value:.5f}")

	print("\n--- Early-stop Model Performance on Epoch-selection Season ---")
	validation_baseline_metrics = evaluate_implied_baseline(data_final_val)
	validation_metrics = evaluate_model(early_stop_model, data_final_val, device=DEVICE, verbose=True)

	data_train, data_test, train_loader, _ = prepare_phase_loaders(
		df,
		feature_cols,
		training_config["batch_size"],
		all_cv_seasons,
		[test_season],
		training_seed + 10_000,
	)
	final_config = build_train_config(training_config, data_train["X"].shape[1], cat_config, final_train_epochs)

	test_baseline_metrics = evaluate_implied_baseline(data_test)

	print("\n--- Training Final Model ---")
	model, _, _ = train_fixed_epochs(final_config, train_loader, device=DEVICE, verbose=True)

	print(f"\n--- Model Performance on Fixed Test Set ({evaluation_config['test_role']}) ---")
	test_metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True)

	reference_row = get_latest_comparable_reference(
		path=EXPERIMENT_LOG_PATH,
		comparison_metric=comparison_metric,
		objective_fold_count=len(objective_folds),
		objective_val_seasons=objective_val_seasons,
		test_season=test_season,
		test_role=evaluation_config["test_role"],
		data_fingerprint=data_snapshot["data_fingerprint"],
	)
	reference_comparison = build_reference_comparison(
		candidate_objective=objective_value,
		evaluation_config=evaluation_config,
		reference_row=reference_row,
	)
	print_reference_comparison(reference_comparison, comparison_metric)

	run_record = {
		"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
		"display_name": DISPLAY_NAME,
		"comparison_metric": comparison_metric,
		"objective_name": f"cv_mean_{comparison_metric}",
		"objective_value": objective_value,
		"training_config_source": str(TRAINING_CONFIG_PATH.relative_to(PROJECT_ROOT)),
		"feature_manifest_source": str(FEATURE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
		"evaluation_config_source": str(EVALUATION_CONFIG_PATH.relative_to(PROJECT_ROOT)),
		"model_name": MODEL_NAME,
		"objective_fold_count": len(objective_folds),
		"objective_val_seasons": objective_val_seasons,
		"objective_metrics": cv_metrics,
		"objective_baseline_metrics": cv_baseline_metrics,
		"objective_fold_metrics": cv_fold_metrics,
		"epoch_selection_season": epoch_selection_season,
		"held_out_test_season": test_season,
		"test_role": evaluation_config["test_role"],
		"min_objective_improvement": evaluation_config["min_objective_improvement"],
		"best_epoch": best_epoch,
		"best_val_loss": float(best_val_loss),
		"data_snapshot": data_snapshot,
		"reference_comparison": reference_comparison,
		"val_metrics": summarize_metrics(validation_metrics),
		"test_metrics": summarize_metrics(test_metrics),
		**git_metadata,
	}
	write_json(
		LATEST_MAIN_METRICS_PATH,
		{
			"schema_version": 3,
			"description": "Latest evaluated match-result candidate. Runtime-generated; compare with prior comparable rows in artifacts/experiment_metrics/result_main_runs.tsv.",
			"model": run_record,
		},
	)

	bundle_metadata = build_bundle_metadata(
		training_config=training_config,
		evaluation_config=evaluation_config,
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
		best_epoch=best_epoch,
		final_train_epochs=final_train_epochs,
		best_val_loss=best_val_loss,
		data_snapshot=data_snapshot,
		reference_comparison=reference_comparison,
	)
	save_bundle(RESULT_MODEL_BUNDLE_PATHS, model, data_train["scaler"], bundle_metadata)
	append_tsv_row(
		EXPERIMENT_LOG_PATH,
		{
			"recorded_at_utc": run_record["recorded_at_utc"],
			"display_name": DISPLAY_NAME,
			"recipe_label": "result",
			"model_name": MODEL_NAME,
			"comparison_metric": comparison_metric,
			"objective_name": run_record["objective_name"],
			"objective_value": run_record["objective_value"],
			"objective_fold_count": run_record["objective_fold_count"],
			"objective_val_seasons": run_record["objective_val_seasons"],
			"best_epoch": best_epoch,
			"final_train_epochs": final_train_epochs,
			"epoch_selection_season": epoch_selection_season,
			"test_role": evaluation_config["test_role"],
			"min_objective_improvement": evaluation_config["min_objective_improvement"],
			"reference_status": reference_comparison["status"],
			"reference_objective_value": reference_comparison["reference_objective_value"],
			"objective_delta": reference_comparison["objective_delta"],
			"meets_objective_threshold": reference_comparison["meets_objective_threshold"],
			"data_fingerprint": data_snapshot["data_fingerprint"],
			"season_row_counts_json": data_snapshot["season_row_counts"],
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
			"evaluation_config_source": run_record["evaluation_config_source"],
			"cv_metrics_json": run_record["objective_metrics"],
			"cv_fold_metrics_json": run_record["objective_fold_metrics"],
			"val_metrics_json": run_record["val_metrics"],
			"test_metrics_json": run_record["test_metrics"],
		},
	)

	print_header("DONE")
	print(f"CV objective ({comparison_metric}): {run_record['objective_value']:.5f}")
	print(f"Minimum meaningful improvement: {evaluation_config['min_objective_improvement']:.5f}")
	print(f"Best validation loss: {best_val_loss:.5f}")
	print(f"Validation metrics: {run_record['val_metrics']}")
	print(f"Test metrics ({evaluation_config['test_role']}): {run_record['test_metrics']}")
	print(f"Experiment log: {EXPERIMENT_LOG_PATH}")
	return run_record


def main():
	train_main_model()


if __name__ == "__main__":
	main()
