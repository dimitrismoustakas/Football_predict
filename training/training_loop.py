"""
Core training loop for the canonical match-result model.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.models.neural_net import GatedResidualModel, TrainConfig, _normalized_market_entropy, gated_loss


def build_model(config: TrainConfig, device: torch.device) -> GatedResidualModel:
	"""Build the configured result model."""

	model = GatedResidualModel(
		input_dim=config.input_dim,
		n_classes=3,
		**config.model_kwargs,
	)
	return model.to(device)


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
	"""Create the fixed optimizer used by the canonical model."""

	return torch.optim.AdamW(
		model.parameters(),
		lr=config.lr,
		weight_decay=config.weight_decay,
		betas=(config.beta1, config.beta2),
		eps=config.optimizer_eps,
	)


def _entropy_curriculum_weights(
	implied_probs: torch.Tensor,
	batch_idx: int,
	total_batches: int,
	config: TrainConfig,
	eps: float = 1e-6,
) -> torch.Tensor | None:
	mode = getattr(config, "entropy_curriculum_mode", "none")
	strength = float(getattr(config, "entropy_curriculum_strength", 0.0))
	if mode == "none" or abs(strength) <= eps:
		return None

	normalized_entropy = _normalized_market_entropy(implied_probs, eps=eps)
	edge_signal = 2.0 * torch.abs(normalized_entropy - 0.5)
	center_signal = 1.0 - edge_signal
	progress = batch_idx / max(1, total_batches - 1)

	if mode == "edge_to_center":
		focus = (1.0 - progress) * edge_signal + progress * center_signal
	elif mode == "center_to_edge":
		focus = (1.0 - progress) * center_signal + progress * edge_signal
	elif mode == "edge_only":
		focus = edge_signal
	elif mode == "center_only":
		focus = center_signal
	else:
		raise ValueError(f"Unsupported entropy_curriculum_mode: {mode}")

	weights = torch.exp(strength * (focus - focus.mean()))
	return weights / weights.mean().clamp_min(eps)


def _compute_training_loss(
	model: GatedResidualModel,
	batch_x: torch.Tensor,
	cat_in: torch.Tensor | None,
	batch_implied: torch.Tensor,
	batch_y: torch.Tensor,
	batch_raw_margin: torch.Tensor,
	sample_weights: torch.Tensor | None,
	config: TrainConfig,
) -> torch.Tensor:
	return gated_loss(
		model,
		batch_x,
		cat_in,
		batch_implied,
		batch_y,
		batch_raw_margin,
		gate_mean_weight=config.gate_mean_weight,
		gate_sat_weight=config.gate_sat_weight,
		lambda_repulsion=config.lambda_repulsion,
		lambda_corr=config.lambda_corr,
		lambda_logit_delta=config.lambda_logit_delta,
		logit_delta_home_weight=config.logit_delta_home_weight,
		logit_delta_draw_weight=config.logit_delta_draw_weight,
		logit_delta_away_weight=config.logit_delta_away_weight,
		market_target_mix=config.market_target_mix,
		market_target_surprise_scale=config.market_target_surprise_scale,
		market_target_surprise_power=config.market_target_surprise_power,
		market_target_surprise_floor=config.market_target_surprise_floor,
		market_target_draw_surprise_scale=config.market_target_draw_surprise_scale,
		market_target_away_surprise_scale=config.market_target_away_surprise_scale,
		market_target_draw_surprise_floor=config.market_target_draw_surprise_floor,
		market_target_away_surprise_floor=config.market_target_away_surprise_floor,
		market_target_surprise_mode=config.market_target_surprise_mode,
		market_target_surprise_center=config.market_target_surprise_center,
		market_target_surprise_width=config.market_target_surprise_width,
		market_target_surprise_slope=config.market_target_surprise_slope,
		market_target_draw_weight=config.market_target_draw_weight,
		market_target_away_weight=config.market_target_away_weight,
		market_target_entropy_scale=config.market_target_entropy_scale,
		market_target_entropy_mode=config.market_target_entropy_mode,
		sample_weights=sample_weights,
		confidence_penalty_weight=config.confidence_penalty_weight,
		brier_aux_weight=config.brier_aux_weight,
		symmetric_ce_weight=config.symmetric_ce_weight,
		symmetric_ce_label_floor=config.symmetric_ce_label_floor,
		gce_mix_weight=config.gce_mix_weight,
		gce_q=config.gce_q,
		bi_tempered_mix_weight=config.bi_tempered_mix_weight,
		bi_tempered_t1=config.bi_tempered_t1,
		bi_tempered_t2=config.bi_tempered_t2,
		bi_tempered_num_iters=config.bi_tempered_num_iters,
		anchor_regret_weight=config.anchor_regret_weight,
		anchor_regret_margin=config.anchor_regret_margin,
		anchor_regret_power=config.anchor_regret_power,
	)


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
	return {name: value.detach().clone() for name, value in model.state_dict().items()}


def run_train_epoch(
	model: GatedResidualModel,
	train_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	config: TrainConfig,
) -> float:
	"""Run one training epoch."""

	model.train()
	total_loss = 0.0
	total_batches = max(1, len(train_loader))
	for batch_idx, (batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin) in enumerate(train_loader):
		batch_x = batch_x.to(device)
		batch_cat = batch_cat.to(device)
		batch_implied = batch_implied.to(device)
		batch_y = batch_y.to(device)
		batch_raw_margin = batch_raw_margin.to(device)
		sample_weights = _entropy_curriculum_weights(batch_implied, batch_idx, total_batches, config)

		optimizer.zero_grad(set_to_none=True)
		loss = _compute_training_loss(
			model,
			batch_x,
			batch_cat,
			batch_implied,
			batch_y,
			batch_raw_margin,
			sample_weights,
			config,
		)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * len(batch_x)
	return total_loss / len(train_loader.dataset)


def run_validation_epoch(
	model: GatedResidualModel,
	val_loader: DataLoader,
	device: torch.device,
) -> Tuple[float, list[float], list[float]]:
	"""Run one validation epoch."""

	model.eval()
	val_loss = 0.0
	all_gates = []
	with torch.no_grad():
		for batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin in val_loader:
			batch_x = batch_x.to(device)
			batch_cat = batch_cat.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)
			batch_raw_margin = batch_raw_margin.to(device)
			pred_logits = model(batch_x, batch_cat, batch_implied, batch_raw_margin)
			loss = F.cross_entropy(pred_logits, batch_y.view(-1).long())
			val_loss += loss.item() * len(batch_x)
			all_gates.append(model.get_gate_stats(batch_x, batch_cat, batch_implied, batch_raw_margin)["gate_values"])

	all_gates = np.concatenate(all_gates, axis=0)
	return (
		val_loss / len(val_loader.dataset),
		all_gates.mean(axis=0).tolist(),
		all_gates.std(axis=0).tolist(),
	)


def _fit_model(
	config: TrainConfig,
	train_loader: DataLoader,
	val_loader: DataLoader | None,
	device: torch.device = None,
	verbose: bool = True,
) -> Tuple[GatedResidualModel, Dict[str, list], float]:
	"""Train the canonical model with optional validation-based early stopping."""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = build_model(config, device)
	optimizer = create_optimizer(model, config)
	use_validation = val_loader is not None
	history = {"train_loss": [], "val_loss": [], "gate_mean": [], "gate_std": []}
	best_val_loss = float("inf")
	best_model_state = None
	patience_loss = float("inf")
	stalled_epochs = 0

	for epoch in range(1, config.epochs + 1):
		avg_train_loss = run_train_epoch(model, train_loader, optimizer, device, config)
		history["train_loss"].append(avg_train_loss)

		if not use_validation:
			if verbose and (epoch % 10 == 0 or epoch == 1):
				print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f}")
			continue

		avg_val_loss, gate_mean, gate_std = run_validation_epoch(model, val_loader, device)
		history["val_loss"].append(avg_val_loss)
		history["gate_mean"].append(gate_mean)
		history["gate_std"].append(gate_std)

		if verbose and (epoch % 10 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | Gate: [{gate_mean[0]:.3f}, {gate_mean[1]:.3f}, {gate_mean[2]:.3f}]"
			)

		if avg_val_loss < best_val_loss:
			best_val_loss = avg_val_loss
			best_model_state = _clone_state_dict(model)

		if avg_val_loss < patience_loss - 1e-4:
			patience_loss = avg_val_loss
			stalled_epochs = 0
		else:
			stalled_epochs += 1

		if stalled_epochs >= config.patience:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break

	if use_validation and best_model_state is not None:
		model.load_state_dict(best_model_state)
	best_loss = best_val_loss if use_validation else history["train_loss"][-1]
	return model, history, best_loss


def train_with_early_stopping(
	config: TrainConfig,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device = None,
	verbose: bool = True,
) -> Tuple[GatedResidualModel, Dict[str, list], float]:
	"""Train with validation-based early stopping."""

	return _fit_model(config, train_loader, val_loader=val_loader, device=device, verbose=verbose)


def train_fixed_epochs(
	config: TrainConfig,
	train_loader: DataLoader,
	device: torch.device = None,
	verbose: bool = True,
) -> Tuple[GatedResidualModel, Dict[str, list], float]:
	"""Train for a fixed number of epochs without validation."""

	return _fit_model(config, train_loader, val_loader=None, device=device, verbose=verbose)
