"""
Shared helpers for canonical result-model families.
"""

import warnings
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LESS_IS_BETTER_METRICS = {"log_loss", "rps", "brier"}
DEFAULT_BLEND_ALPHA_GRID = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0]
DEFAULT_ELASTIC_WEIGHT_GRID = [0.8, 0.85, 0.9, 0.925, 0.95]
DEFAULT_COMPONENT_BLEND_MODE = "convex"
DEFAULT_DRAW_COMPONENT_BASE_WEIGHT_GRID = [0.975]
DEFAULT_DRAW_COMPONENT_BLEND_MODE = "convex"


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
		"lightgbm_blend": "lightgbm_blend",
		"elastic_net_blend": "elastic_net_blend",
		"elastic_hgb_blend": "elastic_hgb_blend",
		"elastic_lgbm_blend": "elastic_lgbm_blend",
	}.get(family, family)


def uses_torch_backend(model_family: str) -> bool:
	"""Return whether the family is the existing Torch path."""

	return model_family == "gated_residual"


def build_component_training_config(training_config: dict, model_family: str) -> dict:
	"""Build a component-local view of the top-level training config."""

	component_config = dict(training_config)
	component_config["model_family"] = model_family
	component_feature_cols = dict(training_config.get("hybrid_component_feature_cols", {}))
	if model_family == "elastic_net_blend" and component_feature_cols.get("elastic") is not None:
		component_config["component_feature_cols"] = list(component_feature_cols["elastic"])
	if model_family in {"hist_gradient_boosting_blend", "lightgbm_blend"} and component_feature_cols.get("tree") is not None:
		component_config["component_feature_cols"] = list(component_feature_cols["tree"])
	return component_config


def has_draw_decomp_component(training_config: dict | None) -> bool:
	"""Return whether an auxiliary draw-decomposition expert is configured."""

	if training_config is None:
		return False
	return "draw_decomp_params" in training_config or "draw_decomp_input_recipe" in training_config


def safe_logit(probs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
	"""Convert probabilities to logits safely."""

	probs = np.clip(probs, eps, 1.0 - eps)
	return np.log(probs / (1.0 - probs))


def normalize_probabilities(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
	"""Normalize rows to valid probability distributions."""

	probs = np.clip(probs, eps, None)
	return probs / probs.sum(axis=1, keepdims=True)


def select_numeric_feature_block(
	X_numeric: np.ndarray,
	available_feature_cols: list[str] | None,
	selected_feature_cols: list[str] | None,
) -> np.ndarray:
	"""Select an optional numeric feature subset from the active feature block."""

	X_numeric = np.nan_to_num(X_numeric, nan=0.0)
	if selected_feature_cols is None:
		return X_numeric
	if available_feature_cols is None:
		raise ValueError("feature_cols are required to select a component-specific design block")
	feature_index = {name: idx for idx, name in enumerate(available_feature_cols)}
	try:
		selected_indices = [feature_index[name] for name in selected_feature_cols]
	except KeyError as exc:
		raise ValueError(f"Selected feature column missing from metadata: {exc.args[0]}") from exc
	return X_numeric[:, selected_indices]


def build_market_feature_enrichment(
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
	input_recipe: dict,
) -> np.ndarray | None:
	"""Build optional global bookmaker feature enrichments."""

	enrichment_mode = str(input_recipe.get("market_feature_enrichment", "none"))
	if enrichment_mode == "none":
		return None

	logit_values = safe_logit(implied_probs.astype(np.float64))
	pieces = []
	if enrichment_mode in {"quadratic", "quadratic_plus_margin"}:
		pieces.append(logit_values ** 2)
		pieces.append(np.stack([
			logit_values[:, 0] * logit_values[:, 1],
			logit_values[:, 0] * logit_values[:, 2],
			logit_values[:, 1] * logit_values[:, 2],
		], axis=1))
	if enrichment_mode == "quadratic_plus_margin":
		pieces.append(logit_values * raw_margin.reshape(-1, 1).astype(np.float64))
	if not pieces:
		raise ValueError(f"Unsupported market feature enrichment mode: {enrichment_mode}")
	return np.concatenate(pieces, axis=1)


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
		if (training_config or {}).get("component_feature_cols") is not None:
			recipe["feature_cols"] = list((training_config or {})["component_feature_cols"])
		return recipe
	if model_family == "hist_gradient_boosting_blend":
		recipe = {
			"numeric_scaling": "raw",
			"league_encoding": "one_hot",
			"include_promoted_flags": True,
			"market_features": "raw_probs_and_margin",
		}
		recipe.update(dict((training_config or {}).get("hgb_input_recipe", {})))
		if (training_config or {}).get("component_feature_cols") is not None:
			recipe["feature_cols"] = list((training_config or {})["component_feature_cols"])
		return recipe
	if model_family == "lightgbm_blend":
		recipe = {
			"numeric_scaling": "raw",
			"league_encoding": "one_hot",
			"include_promoted_flags": True,
			"market_features": "raw_probs_and_margin",
		}
		recipe.update(dict((training_config or {}).get("lightgbm_input_recipe", {})))
		if (training_config or {}).get("component_feature_cols") is not None:
			recipe["feature_cols"] = list((training_config or {})["component_feature_cols"])
		return recipe
	if model_family in {"elastic_hgb_blend", "elastic_lgbm_blend"}:
		tree_component_family = (
			"hist_gradient_boosting_blend"
			if model_family == "elastic_hgb_blend"
			else "lightgbm_blend"
		)
		elastic_recipe = build_input_recipe(
			"elastic_net_blend",
			build_component_training_config(training_config or {}, "elastic_net_blend"),
		)
		tree_recipe = build_input_recipe(
			tree_component_family,
			build_component_training_config(training_config or {}, tree_component_family),
		)
		component_recipes = {
			"elastic": elastic_recipe,
			"tree": tree_recipe,
		}
		if has_draw_decomp_component(training_config):
			draw_recipe = {
				"numeric_scaling": "raw",
				"league_encoding": "one_hot",
				"include_promoted_flags": True,
				"market_features": "raw_probs_and_margin",
			}
			draw_recipe.update(dict((training_config or {}).get("draw_decomp_input_recipe", {})))
			if (training_config or {}).get("draw_decomp_feature_cols") is not None:
				draw_recipe["feature_cols"] = list((training_config or {})["draw_decomp_feature_cols"])
			component_recipes["draw"] = draw_recipe
		return {
			"ensemble_type": "probability_blend",
			"component_recipes": component_recipes,
		}
	return {
		"numeric_scaling": "external_standard_scaler",
		"league_encoding": "native",
		"include_promoted_flags": True,
		"market_features": "native_torch",
	}


def requires_categorical_inputs(metadata: dict) -> bool:
	"""Return whether bundle inference requires categorical raw inputs."""

	recipe = metadata.get("input_recipe") or {}
	component_recipes = recipe.get("component_recipes") or {}
	if component_recipes:
		return any(
			requires_categorical_inputs({"input_recipe": component_recipe})
			for component_recipe in component_recipes.values()
		)
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

	market_enrichment = build_market_feature_enrichment(implied_probs, raw_margin, input_recipe)
	if market_enrichment is not None:
		pieces.append(market_enrichment)

	return np.concatenate(pieces, axis=1)


def build_design_from_data(data: Dict[str, np.ndarray], input_recipe: dict) -> np.ndarray:
	"""Build a design matrix directly from prepared data."""

	X_numeric = data["X"]
	selected_feature_cols = input_recipe.get("feature_cols")
	available_feature_cols = data.get("feature_cols")
	if selected_feature_cols is not None:
		if available_feature_cols is None:
			raise ValueError("feature_cols are required to select a component-specific design block")
		index_by_name = {name: idx for idx, name in enumerate(available_feature_cols)}
		try:
			selected_indices = [index_by_name[name] for name in selected_feature_cols]
		except KeyError as exc:
			raise ValueError(f"Selected feature column missing from prepared data: {exc.args[0]}") from exc
		X_numeric = X_numeric[:, selected_indices]
	return build_design_matrix(
		X_numeric=X_numeric,
		cat_features=data.get("cat_features"),
		implied_probs=data["implied"],
		raw_margin=data["raw_margin"],
		input_recipe=input_recipe,
	)


def get_blend_alpha_grid(training_config: dict) -> list[float]:
	"""Return the convex blend grid used on the selection season."""

	values = training_config.get("blend_alpha_grid", DEFAULT_BLEND_ALPHA_GRID)
	return [float(value) for value in values]


def get_blend_mode(training_config: dict) -> str:
	"""Return the configured bookmaker-blend parameterization."""

	return str(training_config.get("blend_mode", "global"))


def get_class_blend_alpha_grid(training_config: dict) -> list[float]:
	"""Return the per-class alpha grid used on the selection season."""

	values = training_config.get("class_blend_alpha_grid")
	if values is None:
		values = training_config.get("blend_alpha_grid", DEFAULT_BLEND_ALPHA_GRID)
	return [float(value) for value in values]


def get_class_blend_alpha_vector_grid(training_config: dict) -> list[list[float]] | None:
	"""Return optional explicit per-class alpha candidates for market blending."""

	values = training_config.get("class_blend_alpha_vector_grid")
	if values is None:
		return None
	vector_grid = []
	for value in values:
		parsed = parse_component_weight_value(value)
		if not isinstance(parsed, list):
			raise ValueError("Class blend alpha vector candidates must be explicit per-class lists")
		vector_grid.append(parsed)
	return vector_grid


def get_class_blend_alpha_regime_grid(training_config: dict) -> list[dict] | None:
	"""Return optional regime-aware classwise alpha candidates."""

	values = training_config.get("class_blend_alpha_regime_grid")
	if values is None:
		return None
	regime_grid = []
	for value in values:
		if not isinstance(value, dict):
			raise ValueError("Regime-aware class blend candidates must be objects")
		feature = str(value.get("feature", "")).strip()
		if feature not in {"draw_implied", "entropy"}:
			raise ValueError(f"Unsupported class blend regime feature: {feature}")
		if "threshold" not in value:
			raise ValueError("Regime-aware class blend candidates require a threshold")
		regime_grid.append({
			"feature": feature,
			"threshold": float(value["threshold"]),
			"low_alpha": parse_component_weight_value(value.get("low_alpha", 1.0)),
			"high_alpha": parse_component_weight_value(value.get("high_alpha", 1.0)),
		})
	return regime_grid


def get_elastic_weight_grid(training_config: dict) -> list[float | list[float]]:
	"""Return the elastic/tree probability-mix grid for hybrid blends."""

	params = dict(training_config.get("hybrid_blend_params", {}))
	values = params.get("elastic_weight_grid", DEFAULT_ELASTIC_WEIGHT_GRID)
	return [parse_component_weight_value(value) for value in values]


def get_component_blend_mode(training_config: dict) -> str:
	"""Return how hybrid components are combined before market blending."""

	params = dict(training_config.get("hybrid_blend_params", {}))
	return str(params.get("component_blend_mode", DEFAULT_COMPONENT_BLEND_MODE))


def parse_component_weight_value(
	value: float | list[float] | tuple[float, ...] | np.ndarray,
) -> float | list[float]:
	"""Normalize a scalar or vector component-weight value from config."""

	if isinstance(value, np.ndarray):
		if value.ndim == 0:
			return float(value)
		return [float(item) for item in value.tolist()]
	if isinstance(value, (list, tuple)):
		return [float(item) for item in value]
	return float(value)


def get_draw_component_base_weight_grid(training_config: dict) -> list[float | list[float]]:
	"""Return the base/draw blend grid for the auxiliary draw component."""

	params = dict(training_config.get("hybrid_blend_params", {}))
	if "draw_component_base_weight_grid" not in params and "draw_component_base_weight_regime_grid" in params:
		return []
	values = params.get(
		"draw_component_base_weight_grid",
		DEFAULT_DRAW_COMPONENT_BASE_WEIGHT_GRID,
	)
	return [parse_component_weight_value(value) for value in values]


def get_draw_component_base_weight_regime_grid(training_config: dict) -> list[dict]:
	"""Return optional regime-aware base/draw blend candidates."""

	params = dict(training_config.get("hybrid_blend_params", {}))
	values = params.get("draw_component_base_weight_regime_grid", [])
	regime_grid = []
	for value in values:
		if not isinstance(value, dict):
			raise ValueError("Regime-aware component-weight candidates must be objects")
		feature = str(value.get("feature", "")).strip()
		if feature not in {"draw_implied", "entropy"}:
			raise ValueError(f"Unsupported component-weight regime feature: {feature}")
		if "threshold" not in value:
			raise ValueError("Regime-aware component-weight candidates require a threshold")
		regime_grid.append({
			"feature": feature,
			"threshold": float(value["threshold"]),
			"low_weight": parse_component_weight_value(value.get("low_weight", 1.0)),
			"high_weight": parse_component_weight_value(value.get("high_weight", 1.0)),
		})
	return regime_grid


def get_draw_component_blend_mode(training_config: dict) -> str:
	"""Return how the base ensemble and auxiliary draw expert are combined."""

	params = dict(training_config.get("hybrid_blend_params", {}))
	return str(params.get("draw_component_blend_mode", DEFAULT_DRAW_COMPONENT_BLEND_MODE))


def get_draw_binary_params(training_config: dict) -> dict:
	"""Return the parameter block for the draw-vs-non-draw auxiliary head."""

	if "draw_binary_params" in training_config:
		return dict(training_config.get("draw_binary_params", {}))
	return dict(training_config.get("draw_decomp_params", {}))


def get_home_non_draw_binary_params(training_config: dict) -> dict:
	"""Return the parameter block for the home-vs-away-given-non-draw head."""

	if "home_non_draw_binary_params" in training_config:
		return dict(training_config.get("home_non_draw_binary_params", {}))
	return dict(training_config.get("draw_decomp_params", {}))


def apply_implied_blend(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	blend_alpha: float | list[float] | tuple[float, ...] | np.ndarray | dict | None,
) -> np.ndarray:
	"""Blend model probabilities back toward market implied probabilities."""

	if blend_alpha is None:
		return normalize_probabilities(probs)
	if isinstance(blend_alpha, dict):
		return apply_implied_blend_by_regime(probs, implied_probs, blend_alpha)
	alpha = np.asarray(blend_alpha, dtype=np.float64)
	if alpha.ndim == 0:
		alpha = np.full((1, probs.shape[1]), float(alpha))
	elif alpha.ndim == 1:
		if alpha.shape[0] != probs.shape[1]:
			raise ValueError("Per-class blend alpha must match the number of outcomes")
		alpha = alpha.reshape(1, -1)
	else:
		raise ValueError("Blend alpha must be a scalar or a length-K vector")
	blended = alpha * probs + (1.0 - alpha) * implied_probs
	return normalize_probabilities(blended)


def compute_market_regime_signal(implied_probs: np.ndarray, feature: str) -> np.ndarray:
	"""Compute the regime signal used for subgroup-aware market blending."""

	if feature == "draw_implied":
		return implied_probs[:, 1]
	if feature == "entropy":
		clipped = np.clip(implied_probs, 1e-12, 1.0)
		return -(clipped * np.log(clipped)).sum(axis=1)
	raise ValueError(f"Unsupported class blend regime feature: {feature}")


def apply_implied_blend_by_regime(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	blend_alpha_regime: dict,
) -> np.ndarray:
	"""Blend toward market implied probabilities with separate alpha vectors by regime."""

	feature = str(blend_alpha_regime.get("feature", "")).strip()
	threshold = float(blend_alpha_regime["threshold"])
	low_alpha = blend_alpha_regime.get("low_alpha")
	high_alpha = blend_alpha_regime.get("high_alpha")
	mask = compute_market_regime_signal(implied_probs, feature) >= threshold
	blended = np.empty_like(probs)
	if np.any(~mask):
		blended[~mask] = apply_implied_blend(probs[~mask], implied_probs[~mask], low_alpha)
	if np.any(mask):
		blended[mask] = apply_implied_blend(probs[mask], implied_probs[mask], high_alpha)
	return normalize_probabilities(blended)


def blend_component_probabilities(
	elastic_probs: np.ndarray,
	tree_probs: np.ndarray,
	elastic_weight: float | list[float] | tuple[float, ...] | np.ndarray,
	mode: str = DEFAULT_COMPONENT_BLEND_MODE,
) -> np.ndarray:
	"""Blend elastic and tree probabilities into a single prediction."""

	elastic_weight = np.asarray(elastic_weight, dtype=np.float64)
	if elastic_weight.ndim == 0:
		elastic_weight = np.full((1, elastic_probs.shape[1]), float(elastic_weight))
	elif elastic_weight.ndim == 1:
		if elastic_weight.shape[0] != elastic_probs.shape[1]:
			raise ValueError("Per-class component weight must match the number of outcomes")
		elastic_weight = elastic_weight.reshape(1, -1)
	else:
		raise ValueError("Component weight must be a scalar or a length-K vector")
	if mode == "convex":
		return normalize_probabilities(
			elastic_weight * elastic_probs + (1.0 - elastic_weight) * tree_probs
		)
	if mode == "logit":
		mixed = np.exp(
			elastic_weight * np.log(np.clip(elastic_probs, 1e-9, 1.0))
			+ (1.0 - elastic_weight) * np.log(np.clip(tree_probs, 1e-9, 1.0))
		)
		return normalize_probabilities(mixed)
	raise ValueError(f"Unsupported hybrid component blend mode: {mode}")


def blend_component_probabilities_by_regime(
	component_a_probs: np.ndarray,
	component_b_probs: np.ndarray,
	implied_probs: np.ndarray,
	weight_regime: dict,
	mode: str = DEFAULT_COMPONENT_BLEND_MODE,
) -> np.ndarray:
	"""Blend two components with regime-specific weights driven by bookmaker features."""

	feature = str(weight_regime.get("feature", "")).strip()
	threshold = float(weight_regime["threshold"])
	low_weight = weight_regime.get("low_weight", 1.0)
	high_weight = weight_regime.get("high_weight", 1.0)
	mask = compute_market_regime_signal(implied_probs, feature) >= threshold
	blended = np.empty_like(component_a_probs)
	if np.any(~mask):
		blended[~mask] = blend_component_probabilities(
			component_a_probs[~mask],
			component_b_probs[~mask],
			low_weight,
			mode=mode,
		)
	if np.any(mask):
		blended[mask] = blend_component_probabilities(
			component_a_probs[mask],
			component_b_probs[mask],
			high_weight,
			mode=mode,
		)
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


def tune_class_blend_alpha(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	y_true: np.ndarray,
	alpha_grid: list[float],
) -> tuple[list[float], float]:
	"""Select the best per-outcome implied-blend weights on the selection season."""

	best_alpha = None
	best_loss = None
	for alpha_values in np.array(np.meshgrid(*([alpha_grid] * probs.shape[1]))).T.reshape(-1, probs.shape[1]):
		alpha_vector = [float(value) for value in alpha_values]
		loss = log_loss(y_true, apply_implied_blend(probs, implied_probs, alpha_vector), labels=[0, 1, 2])
		if best_loss is None or loss < best_loss:
			best_alpha = alpha_vector
			best_loss = float(loss)
	return best_alpha, float(best_loss)


def tune_class_blend_alpha_vectors(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	y_true: np.ndarray,
	alpha_vector_grid: list[list[float]],
) -> tuple[list[float], float]:
	"""Select the best explicit per-outcome implied-blend vector on the selection season."""

	best_alpha = None
	best_loss = None
	for alpha_vector in alpha_vector_grid:
		if len(alpha_vector) != probs.shape[1]:
			raise ValueError("Explicit class blend alpha candidates must match the number of outcomes")
		alpha_vector = [float(value) for value in alpha_vector]
		loss = log_loss(y_true, apply_implied_blend(probs, implied_probs, alpha_vector), labels=[0, 1, 2])
		if best_loss is None or loss < best_loss:
			best_alpha = alpha_vector
			best_loss = float(loss)
	return best_alpha, float(best_loss)


def tune_class_blend_alpha_regimes(
	probs: np.ndarray,
	implied_probs: np.ndarray,
	y_true: np.ndarray,
	regime_grid: list[dict],
) -> tuple[dict, float]:
	"""Select the best regime-aware classwise alpha candidate on the selection season."""

	best_regime = None
	best_loss = None
	for regime in regime_grid:
		blended = apply_implied_blend_by_regime(probs, implied_probs, regime)
		loss = log_loss(y_true, blended, labels=[0, 1, 2])
		if best_loss is None or loss < best_loss:
			best_regime = {
				"feature": str(regime["feature"]),
				"threshold": float(regime["threshold"]),
				"low_alpha": parse_component_weight_value(regime["low_alpha"]),
				"high_alpha": parse_component_weight_value(regime["high_alpha"]),
			}
			best_loss = float(loss)
	return best_regime, float(best_loss)


def tune_market_blend(
	training_config: dict,
	probs: np.ndarray,
	implied_probs: np.ndarray,
	y_true: np.ndarray,
) -> tuple[str, float | list[float] | dict, float]:
	"""Tune the market-implied blend for the active configuration."""

	blend_mode = get_blend_mode(training_config)
	if blend_mode == "classwise":
		alpha_regime_grid = get_class_blend_alpha_regime_grid(training_config)
		alpha_vector_grid = get_class_blend_alpha_vector_grid(training_config)
		if alpha_regime_grid is not None:
			blend_alpha, blend_val_loss = tune_class_blend_alpha_regimes(
				probs,
				implied_probs,
				y_true,
				alpha_regime_grid,
			)
		elif alpha_vector_grid is not None:
			blend_alpha, blend_val_loss = tune_class_blend_alpha_vectors(
				probs,
				implied_probs,
				y_true,
				alpha_vector_grid,
			)
		else:
			blend_alpha, blend_val_loss = tune_class_blend_alpha(
				probs,
				implied_probs,
				y_true,
				get_class_blend_alpha_grid(training_config),
			)
	else:
		blend_alpha, blend_val_loss = tune_blend_alpha(
			probs,
			implied_probs,
			y_true,
			get_blend_alpha_grid(training_config),
		)
	return blend_mode, blend_alpha, blend_val_loss


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
	if model_family == "lightgbm_blend":
		params = dict(training_config.get("lightgbm_params", {}))
		return LGBMClassifier(
			objective="multiclass",
			num_class=3,
			n_estimators=int(params.get("n_estimators", 300)),
			learning_rate=float(params.get("learning_rate", 0.05)),
			num_leaves=int(params.get("num_leaves", 31)),
			min_child_samples=int(params.get("min_child_samples", 20)),
			reg_lambda=float(params.get("reg_lambda", 0.0)),
			random_state=int(params.get("random_state", 42)),
			n_jobs=int(params.get("n_jobs", -1)),
			verbosity=int(params.get("verbosity", -1)),
		)
	raise ValueError(f"Unsupported non-torch model family: {model_family}")


def build_binary_lightgbm_model(params: dict) -> LGBMClassifier:
	"""Build a binary LightGBM classifier for auxiliary decomposition tasks."""

	return LGBMClassifier(
		objective="binary",
		n_estimators=int(params.get("n_estimators", 300)),
		learning_rate=float(params.get("learning_rate", 0.05)),
		num_leaves=int(params.get("num_leaves", 31)),
		min_child_samples=int(params.get("min_child_samples", 20)),
		reg_lambda=float(params.get("reg_lambda", 0.0)),
		random_state=int(params.get("random_state", 42)),
		n_jobs=int(params.get("n_jobs", -1)),
		verbosity=int(params.get("verbosity", -1)),
	)


def predict_non_torch_model_proba(
	model: Any,
	design: np.ndarray,
) -> np.ndarray:
	"""Predict calibrated probabilities for a non-torch estimator."""

	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
			category=UserWarning,
		)
		return normalize_probabilities(model.predict_proba(design))


def fit_binary_model(
	design: np.ndarray,
	y_binary: np.ndarray,
	params: dict,
) -> Any:
	"""Fit a binary auxiliary model, with a constant fallback for degenerate splits."""

	unique_labels = np.unique(y_binary)
	if unique_labels.size == 1:
		return {
			"constant_positive_prob": float(unique_labels[0]),
		}
	model = build_binary_lightgbm_model(params)
	model.fit(design, y_binary)
	return model


def predict_binary_positive_proba(
	model: Any,
	design: np.ndarray,
	positive_label: int = 1,
) -> np.ndarray:
	"""Predict the positive-class probability for a fitted binary auxiliary model."""

	if isinstance(model, dict) and "constant_positive_prob" in model:
		return np.full(design.shape[0], float(model["constant_positive_prob"]), dtype=np.float64)
	probs = predict_non_torch_model_proba(model, design)
	classes = np.asarray(getattr(model, "classes_", np.arange(probs.shape[1])))
	match_index = np.flatnonzero(classes == positive_label)
	if match_index.size == 0:
		raise ValueError(f"Positive label {positive_label} missing from binary model classes: {classes.tolist()}")
	return np.clip(probs[:, int(match_index[0])].astype(np.float64), 1e-9, 1.0 - 1e-9)


def build_draw_decomp_component(training_config: dict, train_data: Dict[str, np.ndarray]) -> dict:
	"""Fit the auxiliary draw-vs-non-draw and home-vs-away heads."""

	input_recipe = (build_input_recipe(resolve_model_family(training_config), training_config).get("component_recipes") or {}).get("draw")
	if input_recipe is None:
		raise ValueError("Draw-decomposition component requested without an input recipe")
	design = build_design_from_data(train_data, input_recipe)
	draw_target = (train_data["y"] == 1).astype(int)
	non_draw_mask = train_data["y"] != 1
	home_given_non_draw_target = (train_data["y"][non_draw_mask] == 0).astype(int)
	return {
		"draw_binary": fit_binary_model(
			design,
			draw_target,
			get_draw_binary_params(training_config),
		),
		"home_non_draw_binary": fit_binary_model(
			design[non_draw_mask],
			home_given_non_draw_target,
			get_home_non_draw_binary_params(training_config),
		),
	}


def predict_draw_decomp_component_proba(
	model: dict,
	X_numeric: np.ndarray,
	cat_features: np.ndarray | None,
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
	input_recipe: dict,
	available_feature_cols: list[str] | None,
) -> np.ndarray:
	"""Predict 1X2 probabilities from the auxiliary draw-decomposition expert."""

	design = build_design_matrix(
		select_numeric_feature_block(
			X_numeric,
			available_feature_cols,
			input_recipe.get("feature_cols"),
		),
		cat_features,
		implied_probs,
		raw_margin,
		input_recipe,
	)
	draw_prob = predict_binary_positive_proba(model["draw_binary"], design, positive_label=1)
	home_non_draw_prob = predict_binary_positive_proba(
		model["home_non_draw_binary"],
		design,
		positive_label=1,
	)
	probs = np.stack([
		(1.0 - draw_prob) * home_non_draw_prob,
		draw_prob,
		(1.0 - draw_prob) * (1.0 - home_non_draw_prob),
	], axis=1)
	return normalize_probabilities(probs)


def fit_non_torch_selection_model(
	training_config: dict,
	train_data: Dict[str, np.ndarray],
	val_data: Dict[str, np.ndarray],
) -> tuple[Any, dict]:
	"""Fit a non-torch model on the selection split and choose its blend weight."""

	model_family = resolve_model_family(training_config)
	if uses_torch_backend(model_family):
		raise ValueError("Torch models should not use fit_non_torch_selection_model")
	if model_family in {"elastic_hgb_blend", "elastic_lgbm_blend"}:
		tree_component_family = (
			"hist_gradient_boosting_blend"
			if model_family == "elastic_hgb_blend"
			else "lightgbm_blend"
		)
		elastic_config = build_component_training_config(training_config, "elastic_net_blend")
		tree_config = build_component_training_config(training_config, tree_component_family)
		elastic_recipe = build_input_recipe("elastic_net_blend", elastic_config)
		tree_recipe = build_input_recipe(tree_component_family, tree_config)
		elastic_model = build_non_torch_model(elastic_config)
		tree_model = build_non_torch_model(tree_config)
		X_train_elastic = build_design_from_data(train_data, elastic_recipe)
		X_val_elastic = build_design_from_data(val_data, elastic_recipe)
		X_train_tree = build_design_from_data(train_data, tree_recipe)
		X_val_tree = build_design_from_data(val_data, tree_recipe)
		elastic_model.fit(X_train_elastic, train_data["y"])
		tree_model.fit(X_train_tree, train_data["y"])
		elastic_probs = predict_non_torch_model_proba(elastic_model, X_val_elastic)
		tree_probs = predict_non_torch_model_proba(tree_model, X_val_tree)
		component_blend_mode = get_component_blend_mode(training_config)
		draw_component = None
		draw_probs = None
		draw_component_blend_mode = get_draw_component_blend_mode(training_config)
		draw_component_base_weight_grid = [1.0]
		draw_component_base_weight_regime_grid = []
		if has_draw_decomp_component(training_config):
			draw_component = build_draw_decomp_component(training_config, train_data)
			draw_recipe = (build_input_recipe(model_family, training_config).get("component_recipes") or {}).get("draw")
			draw_probs = predict_draw_decomp_component_proba(
				draw_component,
				val_data["X"],
				val_data.get("cat_features"),
				val_data["implied"],
				val_data["raw_margin"],
				draw_recipe,
				val_data.get("feature_cols"),
			)
			draw_component_base_weight_grid = get_draw_component_base_weight_grid(training_config)
			draw_component_base_weight_regime_grid = get_draw_component_base_weight_regime_grid(training_config)
		best_summary = None
		for elastic_weight in get_elastic_weight_grid(training_config):
			base_probs = blend_component_probabilities(
				elastic_probs,
				tree_probs,
				elastic_weight,
				mode=component_blend_mode,
			)
			for draw_component_base_weight in draw_component_base_weight_grid:
				blended_probs = base_probs
				if draw_probs is not None:
					blended_probs = blend_component_probabilities(
						base_probs,
						draw_probs,
						draw_component_base_weight,
						mode=draw_component_blend_mode,
					)
				blend_mode, blend_alpha, blend_val_loss = tune_market_blend(
					training_config,
					blended_probs,
					val_data["implied"],
					val_data["y"],
				)
				if best_summary is None or blend_val_loss < best_summary["blend_val_log_loss"]:
					best_summary = {
						"blend_mode": blend_mode,
						"blend_alpha": blend_alpha,
						"blend_val_log_loss": float(blend_val_loss),
						"elastic_weight": parse_component_weight_value(elastic_weight),
						"component_blend_mode": component_blend_mode,
					}
					if draw_probs is not None:
						best_summary["draw_component_base_weight"] = parse_component_weight_value(draw_component_base_weight)
						best_summary["draw_component_blend_mode"] = draw_component_blend_mode
			for draw_component_base_weight_regime in draw_component_base_weight_regime_grid:
				blended_probs = blend_component_probabilities_by_regime(
					base_probs,
					draw_probs,
					val_data["implied"],
					draw_component_base_weight_regime,
					mode=draw_component_blend_mode,
				)
				blend_mode, blend_alpha, blend_val_loss = tune_market_blend(
					training_config,
					blended_probs,
					val_data["implied"],
					val_data["y"],
				)
				if best_summary is None or blend_val_loss < best_summary["blend_val_log_loss"]:
					best_summary = {
						"blend_mode": blend_mode,
						"blend_alpha": blend_alpha,
						"blend_val_log_loss": float(blend_val_loss),
						"elastic_weight": parse_component_weight_value(elastic_weight),
						"component_blend_mode": component_blend_mode,
						"draw_component_base_weight_regime": {
							"feature": str(draw_component_base_weight_regime["feature"]),
							"threshold": float(draw_component_base_weight_regime["threshold"]),
							"low_weight": parse_component_weight_value(draw_component_base_weight_regime["low_weight"]),
							"high_weight": parse_component_weight_value(draw_component_base_weight_regime["high_weight"]),
						},
						"draw_component_blend_mode": draw_component_blend_mode,
					}
		models = {
			"elastic": elastic_model,
			"tree": tree_model,
		}
		if draw_component is not None:
			models["draw_component"] = draw_component
		return models, best_summary

	input_recipe = build_input_recipe(model_family, training_config)
	model = build_non_torch_model(training_config)
	X_train = build_design_from_data(train_data, input_recipe)
	X_val = build_design_from_data(val_data, input_recipe)
	model.fit(X_train, train_data["y"])
	val_probs = predict_non_torch_model_proba(model, X_val)
	blend_mode, blend_alpha, blend_val_loss = tune_market_blend(
		training_config,
		val_probs,
		val_data["implied"],
		val_data["y"],
	)
	return model, {
		"blend_mode": blend_mode,
		"blend_alpha": blend_alpha,
		"blend_val_log_loss": blend_val_loss,
	}


def fit_non_torch_final_model(training_config: dict, train_data: Dict[str, np.ndarray]):
	"""Fit a non-torch model on a full training split."""

	model_family = resolve_model_family(training_config)
	if uses_torch_backend(model_family):
		raise ValueError("Torch models should not use fit_non_torch_final_model")
	if model_family in {"elastic_hgb_blend", "elastic_lgbm_blend"}:
		tree_component_family = (
			"hist_gradient_boosting_blend"
			if model_family == "elastic_hgb_blend"
			else "lightgbm_blend"
		)
		elastic_config = build_component_training_config(training_config, "elastic_net_blend")
		tree_config = build_component_training_config(training_config, tree_component_family)
		models = {
			"elastic": fit_non_torch_final_model(elastic_config, train_data),
			"tree": fit_non_torch_final_model(tree_config, train_data),
		}
		if has_draw_decomp_component(training_config):
			models["draw_component"] = build_draw_decomp_component(training_config, train_data)
		return models

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
		X_scaled = np.nan_to_num(X_scaled, nan=0.0)
		X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
		cat_tensor = None if cat_features is None else torch.tensor(cat_features, dtype=torch.long, device=device)
		implied_tensor = torch.tensor(implied_probs, dtype=torch.float32, device=device)
		raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32, device=device)
		if hasattr(model, "to"):
			model = model.to(device)
		if hasattr(model, "eval"):
			model.eval()
		with torch.no_grad():
			try:
				logits = model(X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
			except TypeError:
				logits = model(X_tensor, cat_tensor)
			return F.softmax(logits, dim=-1).cpu().numpy()

	input_recipe = metadata.get("input_recipe") or build_input_recipe(model_family)
	if model_family in {"elastic_hgb_blend", "elastic_lgbm_blend"}:
		component_recipes = input_recipe.get("component_recipes") or {}
		elastic_recipe = component_recipes.get("elastic")
		tree_recipe = component_recipes.get("tree")
		if elastic_recipe is None or tree_recipe is None:
			raise ValueError("Hybrid blend metadata is missing component recipes")
		available_feature_cols = metadata.get("feature_cols")
		elastic_X_numeric = select_numeric_feature_block(
			X_numeric,
			available_feature_cols,
			elastic_recipe.get("feature_cols"),
		)
		tree_X_numeric = select_numeric_feature_block(
			X_numeric,
			available_feature_cols,
			tree_recipe.get("feature_cols"),
		)
		elastic_design = build_design_matrix(
			elastic_X_numeric,
			cat_features,
			implied_probs,
			raw_margin,
			elastic_recipe,
		)
		tree_design = build_design_matrix(
			tree_X_numeric,
			cat_features,
			implied_probs,
			raw_margin,
			tree_recipe,
		)
		elastic_probs = predict_non_torch_model_proba(model["elastic"], elastic_design)
		tree_probs = predict_non_torch_model_proba(model["tree"], tree_design)
		elastic_weight = metadata.get("selection_summary", {}).get("elastic_weight", 1.0)
		component_blend_mode = metadata.get("selection_summary", {}).get(
			"component_blend_mode",
			DEFAULT_COMPONENT_BLEND_MODE,
		)
		probs = blend_component_probabilities(
			elastic_probs,
			tree_probs,
			elastic_weight,
			mode=component_blend_mode,
		)
		draw_recipe = component_recipes.get("draw")
		draw_component = model.get("draw_component") if isinstance(model, dict) else None
		draw_component_base_weight = metadata.get("selection_summary", {}).get("draw_component_base_weight")
		draw_component_base_weight_regime = metadata.get("selection_summary", {}).get("draw_component_base_weight_regime")
		if draw_recipe is not None and draw_component is not None and draw_component_base_weight is not None:
			draw_probs = predict_draw_decomp_component_proba(
				draw_component,
				X_numeric,
				cat_features,
				implied_probs,
				raw_margin,
				draw_recipe,
				available_feature_cols,
			)
			draw_component_blend_mode = metadata.get("selection_summary", {}).get(
				"draw_component_blend_mode",
				DEFAULT_DRAW_COMPONENT_BLEND_MODE,
			)
			probs = blend_component_probabilities(
				probs,
				draw_probs,
				draw_component_base_weight,
				mode=draw_component_blend_mode,
			)
		elif draw_recipe is not None and draw_component is not None and draw_component_base_weight_regime is not None:
			draw_probs = predict_draw_decomp_component_proba(
				draw_component,
				X_numeric,
				cat_features,
				implied_probs,
				raw_margin,
				draw_recipe,
				available_feature_cols,
			)
			draw_component_blend_mode = metadata.get("selection_summary", {}).get(
				"draw_component_blend_mode",
				DEFAULT_DRAW_COMPONENT_BLEND_MODE,
			)
			probs = blend_component_probabilities_by_regime(
				probs,
				draw_probs,
				implied_probs,
				draw_component_base_weight_regime,
				mode=draw_component_blend_mode,
			)
	else:
		design = build_design_matrix(np.nan_to_num(X_numeric, nan=0.0), cat_features, implied_probs, raw_margin, input_recipe)
		probs = predict_non_torch_model_proba(model, design)
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
