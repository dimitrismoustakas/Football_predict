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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import torch

from training.evaluation import evaluate_model
from training.experiment_recipe import CANONICAL_RECIPE, build_bundle_metadata, build_train_config, load_training_config
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, save_model_bundle as save_bundle
from training.models import CategoricalConfig
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
from utils.paths import MODELS_DIR, PROJECT_ROOT

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
LATEST_MAIN_METRICS_PATH = MODELS_DIR / "latest_main_model_metrics.json"
N_CV_FOLDS = int(os.environ.get("N_CV_FOLDS", "3"))
TRAINING_SEED = int(os.environ.get("TRAINING_SEED", "42"))

os.environ["MLFLOW_TRACKING_URI"] = "mlruns"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def summarize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
	keys = ["accuracy", "brier", "rps", "log_loss", "total_profit", "avg_profit", "n_bets"]
	return {key: metrics[key] for key in keys if key in metrics}


def write_json(path: Path, payload: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2)


def log_metric_group(prefix: str, metrics: Dict[str, float]):
	for metric_name, metric_value in metrics.items():
		if isinstance(metric_value, (int, float)):
			mlflow.log_metric(f"{prefix}_{metric_name}", float(metric_value))


def describe_training_split(all_cv_seasons: list[str], test_season: str) -> str:
	epoch_selection_season = all_cv_seasons[-1]
	initial_train_seasons = all_cv_seasons[:-1]
	print(
		f"Step 1: train on {initial_train_seasons[0]}..{initial_train_seasons[-1]} | epoch selection on {epoch_selection_season}"
	)
	print(f"Step 2: retrain on {all_cv_seasons[0]}..{all_cv_seasons[-1]} | test on {test_season}")
	return epoch_selection_season


def prepare_phase_loaders(
	df,
	feature_cols: list[str],
	batch_size: int,
	train_seasons: list[str],
	eval_seasons: list[str],
):
	train_data = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
	eval_data = prepare_data(df, feature_cols, eval_seasons, scaler=train_data["scaler"])
	train_loader = to_loader(train_data, batch_size, device=DEVICE)
	eval_loader = to_loader(eval_data, batch_size, shuffle=False, device=DEVICE)
	return train_data, eval_data, train_loader, eval_loader


def resolve_final_training_epochs(max_epochs: int, best_epoch: int) -> int:
	return max(1, min(max_epochs, int(best_epoch)))


def train_main_model() -> dict:
	training_config = load_training_config(CANONICAL_RECIPE)

	print_header(f"TRAIN MAIN MODEL: {CANONICAL_RECIPE.display_name}")
	print(f"Device: {DEVICE}")
	print(f"Training config: {CANONICAL_RECIPE.training_config_path}")

	set_seed(TRAINING_SEED, deterministic=False)
	print(f"\nLoading data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	print(f"Total rows with odds: {len(df)}")

	feature_cols = select_feature_columns(df, CANONICAL_RECIPE.feature_manifest_path)
	print(f"Features: {len(feature_cols)}")
	cat_config = CategoricalConfig(num_leagues=get_num_leagues(df), league_embed_dim=3)
	print(f"Categorical: {cat_config.num_leagues} leagues (embed_dim=3)")

	print(f"\nGenerating {N_CV_FOLDS}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(df, n_folds=N_CV_FOLDS)
	test_season = get_test_season(df)
	print(f"Held-out test season: {test_season}")

	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})
	epoch_selection_season = describe_training_split(all_cv_seasons, test_season)
	initial_train_seasons = all_cv_seasons[:-1]

	mlflow.set_experiment(CANONICAL_RECIPE.experiment_name)
	with mlflow.start_run(run_name=f"{CANONICAL_RECIPE.label}_main"):
		mlflow.log_params({
			"task": "match_result",
			"training_config": str(CANONICAL_RECIPE.training_config_path.relative_to(PROJECT_ROOT)),
			"feature_manifest": str(CANONICAL_RECIPE.feature_manifest_path.relative_to(PROJECT_ROOT)),
			"model_name": CANONICAL_RECIPE.model_name,
			"parquet_path": str(DEFAULT_PARQUET),
			"n_cv_folds": N_CV_FOLDS,
			"held_out_test_season": test_season,
			"optimizer_name": "adamw",
			"scheduler_name": "cosine",
			"final_epoch_mode": "best",
		})

		set_seed(TRAINING_SEED, deterministic=True)
		data_initial_train, data_final_val, initial_train_loader, final_val_loader = prepare_phase_loaders(
			df,
			feature_cols,
			training_config["batch_size"],
			initial_train_seasons,
			[epoch_selection_season],
		)

		early_stop_config = build_train_config(CANONICAL_RECIPE, training_config, input_dim=data_initial_train["X"].shape[1], cat_config=cat_config, epochs=training_config["max_epochs"])
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

		print("\n--- Early-stop Model Performance on Epoch-selection Season ---")
		validation_baseline_metrics = evaluate_implied_baseline(data_final_val)
		log_metric_group("val_baseline", validation_baseline_metrics)
		validation_metrics = evaluate_model(early_stop_model, data_final_val, device=DEVICE, verbose=True)
		log_metric_group("val", validation_metrics)
		mlflow.log_metric("best_val_loss", float(best_val_loss))
		mlflow.log_metric("best_epoch", int(best_epoch))

		data_train, data_test, train_loader, _ = prepare_phase_loaders(
			df,
			feature_cols,
			training_config["batch_size"],
			all_cv_seasons,
			[test_season],
		)
		final_config = build_train_config(CANONICAL_RECIPE, training_config, input_dim=data_train["X"].shape[1], cat_config=cat_config, epochs=final_train_epochs)

		test_baseline_metrics = evaluate_implied_baseline(data_test)
		log_metric_group("test_baseline", test_baseline_metrics)

		print("\n--- Training Final Model ---")
		model, _, _ = train_fixed_epochs(final_config, train_loader, device=DEVICE, verbose=True)

		print("\n--- Model Performance on Held-out Test Set ---")
		test_metrics = evaluate_model(model, data_test, device=DEVICE, verbose=True)
		log_metric_group("test", test_metrics)

		run_record = {
			"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
			"display_name": CANONICAL_RECIPE.display_name,
			"comparison_metric": CANONICAL_RECIPE.comparison_metric,
			"training_config_source": str(CANONICAL_RECIPE.training_config_path.relative_to(PROJECT_ROOT)),
			"feature_manifest_source": str(CANONICAL_RECIPE.feature_manifest_path.relative_to(PROJECT_ROOT)),
			"model_name": CANONICAL_RECIPE.model_name,
			"epoch_selection_season": epoch_selection_season,
			"held_out_test_season": test_season,
			"best_epoch": best_epoch,
			"best_val_loss": float(best_val_loss),
			"val_metrics": summarize_metrics(validation_metrics),
			"test_metrics": summarize_metrics(test_metrics),
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
			recipe=CANONICAL_RECIPE,
			training_config=training_config,
			cat_config=cat_config,
			feature_cols=feature_cols,
			validation_metrics=summarize_metrics(validation_metrics),
			validation_baseline_metrics=summarize_metrics(validation_baseline_metrics),
			test_metrics=summarize_metrics(test_metrics),
			test_baseline_metrics=summarize_metrics(test_baseline_metrics),
			all_cv_seasons=all_cv_seasons,
			final_val_season=epoch_selection_season,
			test_season=test_season,
			n_cv_folds=N_CV_FOLDS,
			training_seed=TRAINING_SEED,
			best_epoch=best_epoch,
			final_train_epochs=final_train_epochs,
			best_val_loss=best_val_loss,
		)
		save_bundle(RESULT_MODEL_BUNDLE_PATHS, model, data_train["scaler"], bundle_metadata)
		mlflow.log_artifact(str(RESULT_MODEL_BUNDLE_PATHS.model_path))
		mlflow.log_artifact(str(RESULT_MODEL_BUNDLE_PATHS.scaler_path))
		mlflow.log_artifact(str(RESULT_MODEL_BUNDLE_PATHS.config_path))

	print_header("DONE")
	print(f"Best validation loss: {best_val_loss:.5f}")
	print(f"Validation metrics: {run_record['val_metrics']}")
	print(f"Test metrics: {run_record['test_metrics']}")
	return run_record


def main():
	train_main_model()


if __name__ == "__main__":
	main()
