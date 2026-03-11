"""
Shared save/load helpers for the canonical result-model bundle.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from training.models import CategoricalConfig, GatedResidualModel
from utils.paths import MODELS_DIR


@dataclass(frozen=True)
class ModelBundlePaths:
	"""File paths for a saved model bundle."""

	name: str
	model_path: Path
	config_path: Path
	scaler_path: Path


@dataclass
class LoadedModelBundle:
	"""Loaded production-ready model bundle."""

	name: str
	model: GatedResidualModel
	scaler: Any
	feature_cols: list[str]
	cat_config: CategoricalConfig | None
	metadata: dict


RESULT_MODEL_BUNDLE_PATHS = ModelBundlePaths(
	name="result_main",
	model_path=MODELS_DIR / "result_model.pt",
	config_path=MODELS_DIR / "result_model_config.json",
	scaler_path=MODELS_DIR / "result_model_scaler.joblib",
)


def resolve_bundle_paths(paths: ModelBundlePaths) -> ModelBundlePaths:
	"""Validate that all bundle files exist."""

	missing = [
		path
		for path in (paths.model_path, paths.config_path, paths.scaler_path)
		if not path.exists()
	]
	if missing:
		raise FileNotFoundError(
			"Missing production model bundle. Expected: " + ", ".join(str(path) for path in missing)
		)
	return paths


def write_bundle_metadata(config_path: Path, metadata: dict):
	"""Write bundle metadata JSON."""

	config_path.parent.mkdir(parents=True, exist_ok=True)
	with open(config_path, "w", encoding="utf-8") as file:
		json.dump(metadata, file, indent=2)


def read_bundle_metadata(config_path: Path) -> dict:
	"""Read bundle metadata JSON."""

	with open(config_path, "r", encoding="utf-8") as file:
		return json.load(file)


def cat_config_from_metadata(metadata: dict) -> CategoricalConfig | None:
	"""Build categorical config from saved metadata."""

	cat_config_dict = metadata.get("cat_config")
	if cat_config_dict is None:
		return None
	return CategoricalConfig(
		num_leagues=cat_config_dict["num_leagues"],
		league_embed_dim=cat_config_dict.get("league_embed_dim", 3),
	)


def infer_num_leagues_from_state_dict(state_dict: dict[str, Any]) -> int | None:
	"""Recover league count for older bundles that omitted it from metadata."""

	for key in ("league_market_bias.weight", "league_gate_bias.weight", "league_residual_bias.weight"):
		weights = state_dict.get(key)
		if weights is not None and getattr(weights, "ndim", 0) >= 1:
			return int(weights.shape[0])
	return None


def build_result_model(
	feature_cols: list[str],
	metadata: dict,
	state_dict: dict[str, Any] | None = None,
) -> tuple[GatedResidualModel, CategoricalConfig | None]:
	"""Build the result model from saved metadata."""

	cat_config = cat_config_from_metadata(metadata)
	model_kwargs = dict(metadata.get("model_kwargs", {}))
	if (
		state_dict is not None
		and model_kwargs.get("num_leagues", 0) <= 0
		and (model_kwargs.get("learn_league_market_bias") or model_kwargs.get("learn_league_residual_bias"))
	):
		inferred_num_leagues = infer_num_leagues_from_state_dict(state_dict)
		if inferred_num_leagues is not None:
			model_kwargs["num_leagues"] = inferred_num_leagues
	model = GatedResidualModel(
		input_dim=len(feature_cols),
		n_classes=3,
		cat_config=cat_config,
		**model_kwargs,
	)
	return model, cat_config


def save_model_bundle(paths: ModelBundlePaths, model: GatedResidualModel, scaler: Any, metadata: dict):
	"""Save model weights, scaler, and metadata."""

	paths.model_path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(model.state_dict(), paths.model_path)
	joblib.dump(scaler, paths.scaler_path)
	write_bundle_metadata(paths.config_path, metadata)


def load_model_bundle(paths: ModelBundlePaths, device: torch.device) -> LoadedModelBundle:
	"""Load a production-ready result-model bundle."""

	paths = resolve_bundle_paths(paths)
	metadata = read_bundle_metadata(paths.config_path)
	feature_cols = metadata.get("feature_cols")
	if not feature_cols:
		raise ValueError(f"No feature column list found in {paths.config_path}")
	try:
		state_dict = torch.load(paths.model_path, map_location=device, weights_only=True)
	except TypeError:
		state_dict = torch.load(paths.model_path, map_location=device)

	try:
		model, cat_config = build_result_model(feature_cols, metadata, state_dict=state_dict)
	except TypeError as exc:
		raise ValueError(
			f"Incompatible runtime model bundle at {paths.config_path}. "
			"Regenerate artifacts/models with `uv run python training/train_main_model.py`."
		) from exc
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()

	return LoadedModelBundle(
		name=paths.name,
		model=model,
		scaler=joblib.load(paths.scaler_path),
		feature_cols=feature_cols,
		cat_config=cat_config,
		metadata=metadata,
	)
