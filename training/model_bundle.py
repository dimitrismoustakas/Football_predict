"""
Shared save/load helpers for the canonical result-model bundle.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import torch

from training.models import GatedResidualModel
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


def build_result_model(
	feature_cols: list[str],
	metadata: dict,
) -> GatedResidualModel:
	"""Build the result model from saved metadata."""

	model_kwargs = dict(metadata.get("model_kwargs", {}))
	return GatedResidualModel(
		input_dim=len(feature_cols),
		n_classes=3,
		**model_kwargs,
	)


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
	state_dict = torch.load(paths.model_path, map_location=device, weights_only=True)
	model = build_result_model(feature_cols, metadata)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()

	return LoadedModelBundle(
		name=paths.name,
		model=model,
		scaler=joblib.load(paths.scaler_path),
		feature_cols=feature_cols,
		metadata=metadata,
	)
