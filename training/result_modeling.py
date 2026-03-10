"""
Shared helpers for canonical result-model families.
"""

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LESS_IS_BETTER_METRICS = {"log_loss", "rps", "brier"}
DEFAULT_BLEND_ALPHA_GRID = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]


def resolve_model_family(training_config: dict | None) -> str:
	"""Return the configured canonical model family."""

	if training_config is None:
		return "gated_residual"
	return str(training_config.get("model_family", "gated_residual"))


def resolve_model_name(training_config: dict | None) -> str:
	"""Return the ledger/model-bundle identifier for the configured family."""

	family = resolve_model_family(training_config)
	return {
		"gated_residual": "gated_residual",
		"hist_gradient_boosting_blend": "hist_gradient_boosting_blend",
		"elastic_net_blend": "elastic_net_blend",
	}.get(family, family)


def uses_torch_backend(model_family: str) -> bool:
	"""Return whether the family is the existing Torch path."""

	return model_family == "gated_residual"


def safe_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Convert probabilities to logits safely."""

	probs = np.clip(probs, eps, 1.0 - eps)
	return np.log(probs / (1.0 - probs))


def normalize_probabilities(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
	"""Normalize rows to valid probability distributions."""

	probs = np.clip(probs, eps, None)
	return probs / probs.sum(axis=1, keepdims=True)


def build_input_recipe(model_family: str, training_config: dict | None = None) -> dict:
	"""Describe how a family expects its tabular inputs."""

	if model_family == "elastic_net_blend":
		recipe = {
			"numeric_scaling": "raw",
			"league_encoding": "one_hot",
			"include_promoted_flags": True,
			"market_features": "logit_probs_and_margin",
			"market_interactions": "none",
		}
		recipe.update(dict((training_config or {}).get("elastic_net_input_recipe", {})))
		return recipe
	if model_family == "hist_gradient_boosting_blend":
		recipe = {
			"numeric_scaling": "raw",
			"league_encoding": "one_hot",
			"include_promoted_flags": True,
			"market_features": "raw_probs_and_margin",
		}
		recipe.update(dict((training_config or {}).get("hgb_input_recipe", {})))
		return recipe
	return {
		"numeric_scaling": "external_standard_scaler",
		"league_encoding": "native",
		"include_promoted_flags": True,
		"market_features": "native_torch",
	}


def requires_categorical_inputs(metadata: dict) -> bool:
	"""Return whether bundle inference requires categorical raw inputs."""

	recipe = metadata.get("input_recipe") or {}
	return recipe.get("league_encoding") == "one_hot" or recipe.get("include_promoted_flags", False)


def build_design_matrix(
	X_numeric: np.ndarray,
	cat_features: np.ndarray | None,
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
	input_recipe: dict,
) -> np.ndarray:
	"""Build the design matrix for supported non-torch model families."""

	pieces = [X_numeric.astype(np.float64)]
	league_onehot = None
	promoted_flags = None

	if requires_categorical_inputs({"input_recipe": input_recipe}):
		if cat_features is None:
			raise ValueError("cat_features required for the configured input recipe")
		if input_recipe.get("league_encoding") == "one_hot":
			league_idx = cat_features[:, 0].astype(int)
			num_leagues = int(league_idx.max()) + 1
			league_onehot = np.eye(num_leagues, dtype=np.float64)[league_idx]
			pieces.append(league_onehot)
		if input_recipe.get("include_promoted_flags", False):
			promoted_flags = cat_features[:, 1:].astype(np.float64)
			pieces.append(promoted_flags)

	market_features = input_recipe.get("market_features", "raw_probs_and_margin")
	market_values = None
	if market_features == "logit_probs_and_margin":
		market_values = safe_logit(implied_probs.astype(np.float64))
		pieces.append(market_values)
		pieces.append(raw_margin.reshape(-1, 1).astype(np.float64))
	elif market_features == "raw_probs_and_margin":
		market_values = implied_probs.astype(np.float64)
		pieces.append(market_values)
		pieces.append(raw_margin.reshape(-1, 1).astype(np.float64))
	elif market_features != "none":
		raise ValueError(f"Unsupported market feature mode: {market_features}")

	market_interactions = str(input_recipe.get("market_interactions", "none"))
	if market_interactions != "none":
		if market_values is None:
			raise ValueError("Market interactions require raw or logit market features")
		context_blocks = []
		if market_interactions in {"league_only", "league_and_promoted"}:
			if league_onehot is None:
				raise ValueError("League interactions require one-hot league encoding")
			context_blocks.append(league_onehot)
		if market_interactions in {"promoted_only", "league_and_promoted"}:
			if promoted_flags is None:
				raise ValueError("Promotion interactions require promoted flags in cat_features")
			context_blocks.append(promoted_flags)
		if not context_blocks:
			raise ValueError(f"Unsupported market interaction mode: {market_interactions}")
		context_matrix = np.concatenate(context_blocks, axis=1)
		# Let linear models learn league/context-specific adjustments to bookmaker logits.
		pieces.extend(context_matrix[:, [idx]] * market_values for idx in range(context_matrix.shape[1]))

	return np.concatenate(pieces, axis=1)


def build_design_from_data(data: Dict[str, np.ndarray], input_recipe: dict) -> np.ndarray:
	"""Build a design matrix directly from prepared data."""

	return build_design_matrix(
		X_numeric=data["X"],
		cat_features=data.get("cat_features"),
		implied_probs=data["implied"],
		raw_margin=data["raw_margin"],
		input_recipe=input_recipe,
	)


def get_blend_alpha_grid(training_config: dict) -> list[float]:
	"""Return the convex blend grid used on the selection season."""

	values = training_config.get("blend_alpha_grid", DEFAULT_BLEND_ALPHA_GRID)
	return [float(value) for value in values]


def apply_implied_blend(probs: np.ndarray, implied_probs: np.ndarray, blend_alpha: float | None) -> np.ndarray:
	"""Blend model probabilities back toward market implied probabilities."""

	if blend_alpha is None:
		return normalize_probabilities(probs)
	blended = blend_alpha * probs + (1.0 - blend_alpha) * implied_probs
	return normalize_probabilities(blended)


def tune_blend_alpha(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	y_true: np.ndarray,
	alpha_grid: list[float],
) -> tuple[float | None, float]:
	"""Select the best implied-blend weight on the fixed selection season."""

	best_alpha = None
	best_loss = None
	for alpha in alpha_grid:
		loss = log_loss(y_true, apply_implied_blend(probs, implied_probs, alpha), labels=[0, 1, 2])
		if best_loss is None or loss < best_loss:
			best_alpha = float(alpha)
			best_loss = float(loss)
	return best_alpha, float(best_loss)


def build_non_torch_model(training_config: dict):
	"""Build a supported non-torch result model."""

	model_family = resolve_model_family(training_config)
	if model_family == "elastic_net_blend":
		params = dict(training_config.get("elastic_net_params", {}))
		return Pipeline([
			("scale", StandardScaler()),
			("clf", LogisticRegression(
				solver="saga",
				penalty="elasticnet",
				l1_ratio=float(params.get("l1_ratio", 0.2)),
				C=float(params.get("C", 0.05)),
				max_iter=int(params.get("max_iter", 5000)),
				random_state=int(params.get("random_state", 42)),
			)),
		])
	if model_family == "hist_gradient_boosting_blend":
		params = dict(training_config.get("hgb_params", {}))
		return HistGradientBoostingClassifier(
			loss="log_loss",
			learning_rate=float(params.get("learning_rate", 0.05)),
			max_depth=None if params.get("max_depth") is None else int(params.get("max_depth")),
			max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
			min_samples_leaf=int(params.get("min_samples_leaf", 100)),
			l2_regularization=float(params.get("l2_regularization", 0.1)),
			max_iter=int(params.get("max_iter", 300)),
			random_state=int(params.get("random_state", 42)),
		)
	raise ValueError(f"Unsupported non-torch model family: {model_family}")


def fit_non_torch_selection_model(
	training_config: dict,
	train_data: Dict[str, np.ndarray],
	val_data: Dict[str, np.ndarray],
) -> tuple[Any, dict]:
	"""Fit a non-torch model on the selection split and choose its blend weight."""

	model_family = resolve_model_family(training_config)
	if uses_torch_backend(model_family):
		raise ValueError("Torch models should not use fit_non_torch_selection_model")

	input_recipe = build_input_recipe(model_family, training_config)
	model = build_non_torch_model(training_config)
	X_train = build_design_from_data(train_data, input_recipe)
	X_val = build_design_from_data(val_data, input_recipe)
	model.fit(X_train, train_data["y"])
	val_probs = normalize_probabilities(model.predict_proba(X_val))
	blend_alpha, blend_val_loss = tune_blend_alpha(
		val_probs,
		val_data["implied"],
		val_data["y"],
		get_blend_alpha_grid(training_config),
	)
	return model, {
		"blend_alpha": blend_alpha,
		"blend_val_log_loss": blend_val_loss,
	}


def fit_non_torch_final_model(training_config: dict, train_data: Dict[str, np.ndarray]):
	"""Fit a non-torch model on a full training split."""

	model_family = resolve_model_family(training_config)
	if uses_torch_backend(model_family):
		raise ValueError("Torch models should not use fit_non_torch_final_model")

	input_recipe = build_input_recipe(model_family, training_config)
	model = build_non_torch_model(training_config)
	model.fit(build_design_from_data(train_data, input_recipe), train_data["y"])
	return model


def predict_result_proba(
	model: Any,
	X_numeric: np.ndarray,
	cat_features: np.ndarray | None,
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
	metadata: dict,
	scaler: Any = None,
	device: torch.device | None = None,
) -> np.ndarray:
	"""Predict canonical result probabilities for any supported family."""

	model_family = metadata.get("model_family", "gated_residual")
	if uses_torch_backend(model_family):
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		X_scaled = X_numeric if scaler is None else scaler.transform(X_numeric)
		X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
		cat_tensor = None if cat_features is None else torch.tensor(cat_features, dtype=torch.long, device=device)
		implied_tensor = torch.tensor(implied_probs, dtype=torch.float32, device=device)
		raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32, device=device)
		model = model.to(device)
		model.eval()
		with torch.no_grad():
			if hasattr(model, "gate_head"):
				logits = model(X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
			else:
				logits = model(X_tensor, cat_tensor)
			return F.softmax(logits, dim=-1).cpu().numpy()

	input_recipe = metadata.get("input_recipe") or build_input_recipe(model_family)
	design = build_design_matrix(X_numeric, cat_features, implied_probs, raw_margin, input_recipe)
	probs = normalize_probabilities(model.predict_proba(design))
	blend_alpha = metadata.get("selection_summary", {}).get("blend_alpha")
	return apply_implied_blend(probs, implied_probs, blend_alpha)


def predict_result_proba_from_data(
	model: Any,
	data: Dict[str, np.ndarray],
	metadata: dict,
	scaler: Any = None,
	device: torch.device | None = None,
) -> np.ndarray:
	"""Predict probabilities from prepared training/evaluation data."""

	return predict_result_proba(
		model=model,
		X_numeric=data["X"],
		cat_features=data.get("cat_features"),
		implied_probs=data["implied"],
		raw_margin=data["raw_margin"],
		metadata=metadata,
		scaler=scaler,
		device=device,
	)
