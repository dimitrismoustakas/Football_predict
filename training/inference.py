"""Shared runtime inference helpers for training evaluation and production scoring."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def model_requires_cat_features(model: Any, cat_config: Any | None) -> bool:
	"""Return whether the model requires categorical side inputs at inference time."""

	return (
		cat_config is not None
		or getattr(model, "learn_league_market_bias", False)
		or getattr(model, "learn_league_market_scale", False)
		or getattr(model, "learn_league_market_class_scale", False)
		or getattr(model, "learn_league_market_logit_mixer", False)
		or getattr(model, "learn_league_gate_bias", False)
		or getattr(model, "learn_league_residual_bias", False)
	)


def forward_model(
	model,
	X: torch.Tensor,
	cat_features: torch.Tensor | None,
	implied: torch.Tensor | None,
	raw_margin: torch.Tensor | None,
) -> torch.Tensor:
	"""Forward pass that works for gated and plain tabular models."""

	use_market = implied is not None and raw_margin is not None
	if use_market:
		try:
			return model(X, cat_features, implied, raw_margin)
		except TypeError:
			pass
	return model(X, cat_features)


def predict_probabilities(
	model,
	scaler,
	X_raw: np.ndarray,
	device: torch.device,
	cat_features: np.ndarray | None = None,
	implied_probs: np.ndarray | None = None,
	raw_margin: np.ndarray | None = None,
) -> np.ndarray:
	"""Scale raw features, run the model, and return class probabilities."""

	if not np.isfinite(X_raw).all():
		raise ValueError("Inference received non-finite model feature values; filter incomplete rows before scoring")
	X_scaled = scaler.transform(X_raw)
	if not np.isfinite(X_scaled).all():
		raise ValueError("Inference produced non-finite scaled features; check feature preparation and scaler inputs")
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
	cat_tensor = None
	if cat_features is not None:
		cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(device)
	implied_tensor = None
	if implied_probs is not None:
		implied_tensor = torch.tensor(implied_probs, dtype=torch.float32).to(device)
	raw_margin_tensor = None
	if raw_margin is not None:
		raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32).to(device)
	with torch.no_grad():
		pred_logits = forward_model(model, X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
		return F.softmax(pred_logits, dim=-1).cpu().numpy()
