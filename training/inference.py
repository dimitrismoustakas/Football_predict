"""Shared runtime inference helpers for the canonical result model."""

import numpy as np
import torch
import torch.nn.functional as F


def predict_probabilities(
	model,
	scaler,
	X_raw: np.ndarray,
	device: torch.device,
	cat_features: np.ndarray,
	implied_probs: np.ndarray,
	raw_margin: np.ndarray,
) -> np.ndarray:
	"""Scale raw features, run the model, and return class probabilities."""

	if not np.isfinite(X_raw).all():
		raise ValueError("Inference received non-finite model feature values; filter incomplete rows before scoring")
	if cat_features is None or implied_probs is None or raw_margin is None:
		raise ValueError("Canonical result inference requires cat_features, implied_probs, and raw_margin")
	X_scaled = scaler.transform(X_raw)
	if not np.isfinite(X_scaled).all():
		raise ValueError("Inference produced non-finite scaled features; check feature preparation and scaler inputs")
	X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
	cat_tensor = torch.tensor(cat_features, dtype=torch.long).to(device)
	implied_tensor = torch.tensor(implied_probs, dtype=torch.float32).to(device)
	raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32).to(device)
	with torch.no_grad():
		pred_logits = model(X_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
		return F.softmax(pred_logits, dim=-1).cpu().numpy()
