"""
Core training loop for the canonical match-result model.
"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.inference import model_requires_cat_features
from training.models.neural_net import GatedResidualModel, TrainConfig, gated_loss


def build_model(config: TrainConfig, device: torch.device) -> GatedResidualModel:
	"""Build the configured result model."""

	model = GatedResidualModel(
		input_dim=config.input_dim,
		n_classes=3,
		cat_config=getattr(config, "cat_config", None),
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



def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> torch.optim.lr_scheduler.CosineAnnealingLR:
	"""Create the fixed scheduler used by the canonical model."""

	min_lr = config.lr * config.scheduler_min_lr_ratio
	return torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=max(1, config.epochs),
		eta_min=min_lr,
	)



def run_train_epoch(
	model: GatedResidualModel,
	train_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	config: TrainConfig,
) -> float:
	"""Run one training epoch."""

	cat_config = getattr(config, "cat_config", None)
	needs_cat = model_requires_cat_features(model, cat_config)
	model.train()
	total_loss = 0.0
	for batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin in train_loader:
		batch_x = batch_x.to(device)
		batch_cat = batch_cat.to(device)
		batch_implied = batch_implied.to(device)
		batch_y = batch_y.to(device)
		batch_raw_margin = batch_raw_margin.to(device)
		cat_in = batch_cat if needs_cat else None

		optimizer.zero_grad(set_to_none=True)
		loss = gated_loss(
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
			market_target_mix=config.market_target_mix,
			gce_mix_weight=config.gce_mix_weight,
			gce_q=config.gce_q,
		)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * len(batch_x)
	return total_loss / len(train_loader.dataset)


def run_validation_epoch(
	model: GatedResidualModel,
	val_loader: DataLoader,
	device: torch.device,
	cat_config,
	needs_cat: bool,
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
			cat_in = batch_cat if needs_cat else None
			pred_logits = model(batch_x, cat_in, batch_implied, batch_raw_margin)
			loss = F.cross_entropy(pred_logits, batch_y.view(-1).long())
			val_loss += loss.item() * len(batch_x)
			all_gates.append(model.get_gate_stats(batch_x, cat_in, batch_implied, batch_raw_margin)["gate_values"])

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
	scheduler = create_scheduler(optimizer, config)
	cat_config = getattr(config, "cat_config", None)
	needs_cat = model_requires_cat_features(model, cat_config)
	use_validation = val_loader is not None
	history = {"train_loss": [], "val_loss": [], "gate_mean": [], "gate_std": []}
	best_val_loss = float("inf")
	best_model_state = None
	stalled_epochs = 0

	for epoch in range(1, config.epochs + 1):
		avg_train_loss = run_train_epoch(model, train_loader, optimizer, device, config)
		history["train_loss"].append(avg_train_loss)

		if not use_validation:
			scheduler.step()
			if verbose and (epoch % 10 == 0 or epoch == 1):
				print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f}")
			continue

		avg_val_loss, gate_mean, gate_std = run_validation_epoch(model, val_loader, device, cat_config, needs_cat)
		history["val_loss"].append(avg_val_loss)
		history["gate_mean"].append(gate_mean)
		history["gate_std"].append(gate_std)
		scheduler.step()

		if verbose and (epoch % 10 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | Gate: [{gate_mean[0]:.3f}, {gate_mean[1]:.3f}, {gate_mean[2]:.3f}]"
			)

		if avg_val_loss < best_val_loss - 1e-4:
			best_val_loss = avg_val_loss
			best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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
