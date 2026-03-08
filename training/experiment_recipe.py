"""
Experiment recipe surface for the canonical match-result harness.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from training.models import CategoricalConfig, TrainConfig
from utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class ExperimentRecipe:
	"""Editable experiment surface for the fixed training harness."""

	label: str
	display_name: str
	comparison_metric: str
	experiment_name: str
	training_config_path: Path
	feature_manifest_path: Path
	model_name: str = "gated_residual"


CANONICAL_RECIPE = ExperimentRecipe(
	label="result",
	display_name="Match Result",
	comparison_metric="log_loss",
	experiment_name="result_main_model",
	training_config_path=PROJECT_ROOT / "training" / "configs" / "main_models" / "result.json",
	feature_manifest_path=PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json",
)


def load_training_config(recipe: ExperimentRecipe) -> dict:
	"""Load the frozen training config for a recipe."""

	with open(recipe.training_config_path, "r", encoding="utf-8") as file:
		return json.load(file)


def build_train_config(
	recipe: ExperimentRecipe,
	training_config: dict,
	input_dim: int,
	cat_config: CategoricalConfig,
	epochs: int,
) -> TrainConfig:
	"""Translate recipe config JSON into the runtime training config."""

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
		model_name=recipe.model_name,
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
	recipe: ExperimentRecipe,
	training_config: dict,
	cat_config: CategoricalConfig,
	feature_cols: list[str],
	validation_metrics: dict,
	validation_baseline_metrics: dict,
	test_metrics: dict,
	test_baseline_metrics: dict,
	all_cv_seasons: list[str],
	final_val_season: str,
	test_season: str,
	n_cv_folds: int,
	training_seed: int,
	best_epoch: int,
	final_train_epochs: int,
	best_val_loss: float,
) -> dict:
	"""Build saved model-bundle metadata from the current recipe."""

	return {
		"display_name": recipe.display_name,
		"model_name": recipe.model_name,
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
			"selection_metric": recipe.comparison_metric,
			"cv_seasons": all_cv_seasons,
			"epoch_selection_season": final_val_season,
			"held_out_test_season": test_season,
			"training_seed": training_seed,
		},
		"selection_summary": {
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
