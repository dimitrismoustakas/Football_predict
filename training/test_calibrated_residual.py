"""
Test: Learnable Calibration/Shrinkage in Log-Space for Multiclass Result Prediction.

Current implementation:
    p̂(y|x) = softmax(log p_mkt(y) + r(x))

This test implements:
    pred_logits = α · log p_mkt + β · r(x) + b

Where:
- α: temperature on the market distribution (α<1 softens, α>1 sharpens)
- β: scales the residual corrections from the neural network
- b: per-class bias (global systematic tilt)

The key difference from the original is that instead of fixed "+ log p_mkt",
we learn how much to trust the market vs our residual corrections.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_utils import (
	load_frame,
	filter_min_history,
	add_targets_and_implied_result,
	prepare_data_result,
	get_test_season,
	get_num_leagues,
	to_loader,
	EarlyStopping,
	create_scheduler,
)
from training.models.neural_net import (
	MLP,
	CategoricalConfig,
	_log_softmax_from_implied,
)
from training.analyze_residuals_by_decile import (
	analyze_residuals_by_realized_outcome,
	analyze_calibration_by_predicted_prob,
	print_residual_table,
	print_calibration_table,
	plot_residual_analysis,
	plot_calibration_analysis,
)
from training.evaluation.metrics import evaluate_profit_result


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# CALIBRATED RESIDUAL MODEL
# ============================================================================

class CalibratedResidualModel(nn.Module):
	"""
	Neural network with learnable calibration parameters for market-residual blending.
	
	Instead of: pred_logits = log p_mkt + r(x)
	We learn:   pred_logits = α · log p_mkt + β · r(x) + b
	
	Args:
		base_model: The underlying MLP that produces residual logits r(x)
		n_classes: Number of output classes (3 for home/draw/away)
		init_alpha: Initial value for α (market temperature)
		init_beta: Initial value for β (residual scaling)
	"""
	
	def __init__(
		self,
		base_model: MLP,
		n_classes: int = 3,
		init_alpha: float = 1.0,
		init_beta: float = 1.0,
	):
		super().__init__()
		self.base_model = base_model
		self.n_classes = n_classes
		
		# Learnable calibration parameters
		# α: temperature on market log-probs
		self.alpha = nn.Parameter(torch.tensor(init_alpha))
		# β: scaling for residual corrections
		self.beta = nn.Parameter(torch.tensor(init_beta))
		# b: per-class bias
		self.class_bias = nn.Parameter(torch.zeros(n_classes))
	
	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_probs: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Forward pass.
		
		Args:
			x: Continuous features (batch, input_dim)
			cat_features: Optional categorical features (batch, n_cat)
			implied_probs: Market implied probabilities (batch, n_classes)
			                If None, returns raw residual logits (for inference without odds)
		
		Returns:
			Final logits: α · log(p_mkt) + β · r(x) + b
		"""
		# Get residual logits from base model
		residual_logits = self.base_model(x, cat_features)
		
		if implied_probs is None:
			# Without market probs, just return scaled residuals + bias
			return self.beta * residual_logits + self.class_bias
		
		# Convert implied probs to log space
		implied_log = _log_softmax_from_implied(implied_probs)
		
		# Apply learnable calibration: α · log(p_mkt) + β · r(x) + b
		pred_logits = self.alpha * implied_log + self.beta * residual_logits + self.class_bias
		
		return pred_logits
	
	def get_calibration_params(self) -> Dict[str, float]:
		"""Return current calibration parameter values."""
		return {
			"alpha": self.alpha.item(),
			"beta": self.beta.item(),
			"class_bias": self.class_bias.detach().cpu().numpy().tolist(),
		}


# ============================================================================
# CONTEXTUAL GATED RESIDUAL MODEL
# ============================================================================

class MLPWithHiddenAccess(nn.Module):
	"""
	MLP that provides access to the last hidden layer representation.
	
	This wraps the base MLP to extract h(x) before the final linear layer.
	"""
	
	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
		output_dim: int = 3,
		cat_config: Optional[CategoricalConfig] = None,
	):
		super().__init__()
		
		# Categorical embedder (optional)
		self.cat_embedder = None
		total_input_dim = input_dim
		if cat_config is not None:
			from training.models.neural_net import CategoricalEmbedder
			self.cat_embedder = CategoricalEmbedder(cat_config)
			total_input_dim = input_dim + self.cat_embedder.output_dim
		
		# Build hidden layers (all but final)
		layers = []
		prev = total_input_dim
		NormClass = {"none": None, "bn": nn.BatchNorm1d, "ln": nn.LayerNorm}.get(norm)
		
		def get_activation():
			if activation == "relu":
				return nn.ReLU()
			elif activation == "silu":
				return nn.SiLU()
			elif activation == "gelu":
				return nn.GELU()
			else:
				raise ValueError(f"Unknown activation: {activation}")
		
		for h in hidden_layers:
			layers.append(nn.Linear(prev, h))
			if NormClass is not None:
				layers.append(NormClass(h))
			layers.append(get_activation())
			layers.append(nn.Dropout(dropout))
			prev = h
		
		self.hidden_net = nn.Sequential(*layers)
		self.final_layer = nn.Linear(prev, output_dim)
		self.hidden_dim = prev  # Last hidden layer dimension
		self.cat_config = cat_config
	
	def forward(self, x: torch.Tensor, cat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Standard forward pass returning output logits."""
		h = self.get_hidden(x, cat_features)
		return self.final_layer(h)
	
	def get_hidden(self, x: torch.Tensor, cat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Get the last hidden layer representation h(x)."""
		if self.cat_embedder is not None:
			if cat_features is None:
				raise ValueError("cat_features required when model has cat_config")
			cat_emb = self.cat_embedder(cat_features)
			x = torch.cat([x, cat_emb], dim=-1)
		return self.hidden_net(x)


class ContextualGatedResidualModel(nn.Module):
	"""
	Neural network with contextual gating that learns WHEN to deviate from market.
	
	The gate g is computed from both hidden features and market features:
		g = sigmoid(g_head([h(x), market_features]))
		pred_logits = log p_mkt + g · r(x)
	
	Where:
		- h(x): hidden representation from the base network
		- market_features: implied probabilities (and optionally derived features)
		- g: per-sample, per-class gate (3-vector) controlling how much to trust model
		- r(x): residual logits from base model
	
	This allows the model to learn contexts where it should trust itself more
	(e.g., when market is uncertain, specific leagues, etc.)
	
	Args:
		input_dim: Number of continuous input features
		hidden_layers: List of hidden layer sizes
		n_classes: Number of output classes (3 for home/draw/away)
		cat_config: Optional categorical feature config
		gate_hidden_dim: Hidden dimension for the gate network
		market_feature_dim: Dimension of market features (default 3 for implied probs)
		dropout: Dropout rate
		norm: Normalization type
		activation: Activation function
	"""
	
	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		n_classes: int = 3,
		cat_config: Optional[CategoricalConfig] = None,
		gate_hidden_dim: int = 32,
		market_feature_dim: int = 3,
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
	):
		super().__init__()
		self.n_classes = n_classes
		
		# Base model that produces residual logits and hidden representation
		self.base_model = MLPWithHiddenAccess(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			norm=norm,
			activation=activation,
			output_dim=n_classes,
			cat_config=cat_config,
		)
		
		# Gate head: takes [h(x), market_features] -> g (3-vector)
		# Market features: implied probs + derived features (entropy, max_prob, etc.)
		self.market_feature_dim = market_feature_dim + 3  # +3 for derived features
		gate_input_dim = self.base_model.hidden_dim + self.market_feature_dim
		
		# Gate head outputs logits (no sigmoid here - applied in forward with bias)
		self.gate_head = nn.Sequential(
			nn.Linear(gate_input_dim, gate_hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 0.5),  # Less dropout in gate
			nn.Linear(gate_hidden_dim, n_classes),
		)
		
		# Learnable bias in logit space (0 = neutral gate of 0.5 after sigmoid)
		self.gate_bias = nn.Parameter(torch.zeros(n_classes))
	
	def _compute_market_features(self, implied_probs: torch.Tensor) -> torch.Tensor:
		"""
		Compute market features from implied probabilities.
		
		Features:
			- implied_probs: raw probabilities (3)
			- entropy: market uncertainty (1)
			- max_prob: market confidence (1)
			- margin: overround indicator (1)
		"""
		eps = 1e-6
		
		# Normalize implied probs
		implied_norm = implied_probs / (implied_probs.sum(dim=-1, keepdim=True) + eps)
		
		# Entropy: -sum(p * log(p))
		entropy = -torch.sum(implied_norm * torch.log(implied_norm + eps), dim=-1, keepdim=True)
		# Normalize entropy to [0, 1] range (max entropy for 3 classes is log(3) ≈ 1.1)
		entropy = entropy / 1.1
		
		# Max probability (market confidence)
		max_prob = implied_probs.max(dim=-1, keepdim=True)[0]
		
		# Sum of implied probs (overround indicator, typically > 1)
		margin = implied_probs.sum(dim=-1, keepdim=True)
		
		return torch.cat([implied_probs, entropy, max_prob, margin], dim=-1)
	
	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_probs: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Forward pass with contextual gating.
		
		Args:
			x: Continuous features (batch, input_dim)
			cat_features: Optional categorical features (batch, n_cat)
			implied_probs: Market implied probabilities (batch, n_classes)
		
		Returns:
			Final logits: log(p_mkt) + g · r(x)
		"""
		# Get hidden representation and residual logits
		h = self.base_model.get_hidden(x, cat_features)
		residual_logits = self.base_model.final_layer(h)
		
		if implied_probs is None:
			# Without market probs, return raw residuals (gate = 1)
			return residual_logits
		
		# Compute market features
		market_features = self._compute_market_features(implied_probs)
		
		# Compute gate: g = sigmoid(logits + bias)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = self.gate_head(gate_input)  # (batch, n_classes)
		
		# Apply bias in logit space, then sigmoid (bias=0 gives neutral gate of 0.5)
		g = torch.sigmoid(gate_logits + self.gate_bias)
		
		# Convert implied probs to log space
		implied_log = _log_softmax_from_implied(implied_probs)
		
		# Final prediction: log p_mkt + g · r(x)
		pred_logits = implied_log + g * residual_logits
		
		return pred_logits
	
	def get_gate_stats(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor],
		implied_probs: torch.Tensor,
	) -> Dict[str, np.ndarray]:
		"""Get gate statistics for analysis."""
		self.eval()
		with torch.no_grad():
			h = self.base_model.get_hidden(x, cat_features)
			market_features = self._compute_market_features(implied_probs)
			gate_input = torch.cat([h, market_features], dim=-1)
			gate_logits = self.gate_head(gate_input)
			g = torch.sigmoid(gate_logits + self.gate_bias)
		
		return {
			"gate_values": g.cpu().numpy(),
			"gate_mean": g.mean(dim=0).cpu().numpy(),
			"gate_std": g.std(dim=0).cpu().numpy(),
			"gate_min": g.min(dim=0)[0].cpu().numpy(),
			"gate_max": g.max(dim=0)[0].cpu().numpy(),
		}


def contextual_gated_loss_multiclass(
	model: ContextualGatedResidualModel,
	x: torch.Tensor,
	cat_features: Optional[torch.Tensor],
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	gate_reg_weight: float = 0.01,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Loss function for contextual gated model.
	
	Includes optional regularization to prevent gate from collapsing to 0 or 1.
	
	Args:
		model: ContextualGatedResidualModel instance
		x: Input features
		cat_features: Optional categorical features
		implied_probs: Market implied probabilities
		target: Class labels
		gate_reg_weight: Weight for gate regularization (encourages gate diversity)
		eps: Numerical stability epsilon
	
	Returns:
		Loss value
	"""
	# Get predictions (model handles gating internally)
	pred_logits = model(x, cat_features, implied_probs)
	
	# Cross-entropy loss
	target = target.view(-1).long()
	loss = F.cross_entropy(pred_logits, target)
	
	# Optional: gate regularization to encourage diverse gate values
	if gate_reg_weight > 0:
		h = model.base_model.get_hidden(x, cat_features)
		market_features = model._compute_market_features(implied_probs)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = model.gate_head(gate_input)
		# Use the same gate computation as forward (with bias)
		g = torch.sigmoid(gate_logits + model.gate_bias)
		
		# Penalize gates that are too close to 0 or 1 (encourage exploration)
		# Using entropy-like regularization: -g*log(g) - (1-g)*log(1-g)
		gate_entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))
		gate_reg = -gate_entropy.mean()  # Negative because we want to maximize entropy
		
		loss = loss + gate_reg_weight * gate_reg
	
	return loss


def calibrated_residual_loss_multiclass(
	model: CalibratedResidualModel,
	x: torch.Tensor,
	cat_features: Optional[torch.Tensor],
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Loss function for calibrated residual model (multiclass).
	
	The model internally computes: α · log(p_mkt) + β · r(x) + b
	We just apply cross-entropy on top.
	
	Args:
		model: CalibratedResidualModel instance
		x: Input features (batch, input_dim)
		cat_features: Optional categorical features (batch, n_cat)
		implied_probs: Market implied probabilities (batch, 3)
		target: Class labels (batch,) with values 0, 1, 2
		eps: Numerical stability epsilon
	
	Returns:
		Cross-entropy loss
	"""
	# Get calibrated logits from model
	pred_logits = model(x, cat_features, implied_probs)
	
	# Cross-entropy loss
	target = target.view(-1).long()
	loss = F.cross_entropy(pred_logits, target)
	
	return loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_calibrated_model(
	model: CalibratedResidualModel,
	train_loader,
	val_loader,
	lr: float,
	weight_decay: float,
	beta1: float,
	epochs: int,
	patience: int,
	device: torch.device,
	verbose: bool = True,
) -> Tuple[CalibratedResidualModel, Dict, float]:
	"""
	Train the calibrated residual model.
	
	Returns: (model, history, best_val_loss)
	"""
	model = model.to(device)
	
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay,
		betas=(beta1, 0.999),
	)
	
	scheduler = create_scheduler(optimizer, epochs=epochs, lr=lr)
	early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
	
	history = {"train_loss": [], "val_loss": [], "alpha": [], "beta": [], "bias": []}
	
	for epoch in range(1, epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		
		for batch_x, batch_cat, batch_implied, batch_y in train_loader:
			batch_x = batch_x.to(device)
			batch_cat = batch_cat.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)
			
			optimizer.zero_grad()
			loss = calibrated_residual_loss_multiclass(
				model, batch_x, batch_cat, batch_implied, batch_y
			)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(batch_x)
		
		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)
		
		# Validation phase
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for bx, bc, bi, by in val_loader:
				bx = bx.to(device)
				bc = bc.to(device)
				bi = bi.to(device)
				by = by.to(device)
				
				loss = calibrated_residual_loss_multiclass(model, bx, bc, bi, by)
				val_loss += loss.item() * len(bx)
		
		avg_val_loss = val_loss / len(val_loader.dataset)
		history["val_loss"].append(avg_val_loss)
		
		# Track calibration parameters
		params = model.get_calibration_params()
		history["alpha"].append(params["alpha"])
		history["beta"].append(params["beta"])
		history["bias"].append(params["class_bias"])
		
		scheduler.step()
		early_stopping(avg_val_loss, model)
		
		if verbose and (epoch % 5 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
				f"α={params['alpha']:.4f} | β={params['beta']:.4f} | "
				f"b=[{params['class_bias'][0]:.3f}, {params['class_bias'][1]:.3f}, {params['class_bias'][2]:.3f}]"
			)
		
		if early_stopping.early_stop:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break
	
	early_stopping.load_best_weights(model)
	return model, history, early_stopping.best_loss


def train_contextual_gated_model(
	model: ContextualGatedResidualModel,
	train_loader,
	val_loader,
	lr: float,
	weight_decay: float,
	beta1: float,
	epochs: int,
	patience: int,
	device: torch.device,
	gate_reg_weight: float = 0.01,
	verbose: bool = True,
) -> Tuple[ContextualGatedResidualModel, Dict, float]:
	"""
	Train the contextual gated residual model.
	
	Returns: (model, history, best_val_loss)
	"""
	model = model.to(device)
	
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay,
		betas=(beta1, 0.999),
	)
	
	scheduler = create_scheduler(optimizer, epochs=epochs, lr=lr)
	early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
	
	history = {"train_loss": [], "val_loss": [], "gate_mean": [], "gate_std": []}
	
	for epoch in range(1, epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		
		for batch_x, batch_cat, batch_implied, batch_y in train_loader:
			batch_x = batch_x.to(device)
			batch_cat = batch_cat.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)
			
			optimizer.zero_grad()
			loss = contextual_gated_loss_multiclass(
				model, batch_x, batch_cat, batch_implied, batch_y,
				gate_reg_weight=gate_reg_weight
			)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(batch_x)
		
		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)
		
		# Validation phase
		model.eval()
		val_loss = 0.0
		all_gates = []
		with torch.no_grad():
			for bx, bc, bi, by in val_loader:
				bx = bx.to(device)
				bc = bc.to(device)
				bi = bi.to(device)
				by = by.to(device)
				
				# Get loss without gate regularization for fair comparison
				pred_logits = model(bx, bc, bi)
				loss = F.cross_entropy(pred_logits, by.view(-1).long())
				val_loss += loss.item() * len(bx)
				
				# Collect gate values
				gate_stats = model.get_gate_stats(bx, bc, bi)
				all_gates.append(gate_stats["gate_values"])
		
		avg_val_loss = val_loss / len(val_loader.dataset)
		history["val_loss"].append(avg_val_loss)
		
		# Track gate statistics
		all_gates = np.concatenate(all_gates, axis=0)
		gate_mean = all_gates.mean(axis=0)
		gate_std = all_gates.std(axis=0)
		history["gate_mean"].append(gate_mean.tolist())
		history["gate_std"].append(gate_std.tolist())
		
		scheduler.step()
		early_stopping(avg_val_loss, model)
		
		if verbose and (epoch % 5 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
				f"Gate μ=[{gate_mean[0]:.3f}, {gate_mean[1]:.3f}, {gate_mean[2]:.3f}] | "
				f"Gate σ=[{gate_std[0]:.3f}, {gate_std[1]:.3f}, {gate_std[2]:.3f}]"
			)
		
		if early_stopping.early_stop:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break
	
	early_stopping.load_best_weights(model)
	return model, history, early_stopping.best_loss


# ============================================================================
# EVALUATION
# ============================================================================

def get_calibrated_model_predictions(
	model: CalibratedResidualModel,
	X: np.ndarray,
	cat_features: np.ndarray,
	implied_probs: np.ndarray,
	device: torch.device,
) -> np.ndarray:
	"""Get predictions from calibrated model."""
	model.eval()
	with torch.no_grad():
		X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
		cat_tensor = torch.tensor(cat_features, dtype=torch.long, device=device)
		implied_tensor = torch.tensor(implied_probs, dtype=torch.float32, device=device)
		
		logits = model(X_tensor, cat_tensor, implied_tensor)
		probs = F.softmax(logits, dim=1).cpu().numpy()
	
	return probs


def compare_residual_analysis(
	y_true: np.ndarray,
	baseline_probs: np.ndarray,
	calibrated_probs: np.ndarray,
	implied_probs: np.ndarray,
	n_bins: int = 10,
) -> Dict:
	"""Compare residual analysis between baseline and calibrated models."""
	baseline_results = analyze_residuals_by_realized_outcome(
		y_true, baseline_probs, implied_probs, "multiclass", n_bins
	)
	calibrated_results = analyze_residuals_by_realized_outcome(
		y_true, calibrated_probs, implied_probs, "multiclass", n_bins
	)
	
	return {
		"baseline": baseline_results,
		"calibrated": calibrated_results,
	}


def print_comparison_table(comparison: Dict):
	"""Print side-by-side comparison of baseline vs calibrated model."""
	baseline = comparison["baseline"]["bins"]
	calibrated = comparison["calibrated"]["bins"]
	
	print(f"\n{'='*120}")
	print("RESIDUAL ANALYSIS COMPARISON: BASELINE vs CALIBRATED")
	print(f"{'='*120}\n")
	
	header = (
		f"{'Bin':^6} | {'Market Prob':^12} | {'N':^6} | "
		f"{'Base Resid':^11} | {'Cal Resid':^11} | {'Δ Resid':^10} | "
		f"{'Base ΔLL':^10} | {'Cal ΔLL':^10} | {'Improvement':^12}"
	)
	print(header)
	print("-" * len(header))
	
	total_base_ll_delta = 0.0
	total_cal_ll_delta = 0.0
	total_samples = 0
	
	for i, (b, c) in enumerate(zip(baseline, calibrated)):
		n_samples = b["n_samples"]
		prob_range = f"{b['bin_range'][0]:.3f}-{b['bin_range'][1]:.3f}"
		
		resid_improvement = abs(c["mean_residual"]) - abs(b["mean_residual"])
		ll_improvement = b["mean_log_loss_delta"] - c["mean_log_loss_delta"]
		
		total_base_ll_delta += b["mean_log_loss_delta"] * n_samples
		total_cal_ll_delta += c["mean_log_loss_delta"] * n_samples
		total_samples += n_samples
		
		row = (
			f"D{i+1:2d}    | "
			f"{prob_range:^12} | "
			f"{n_samples:6d} | "
			f"{b['mean_residual']:+10.4f} | "
			f"{c['mean_residual']:+10.4f} | "
			f"{resid_improvement:+9.4f} | "
			f"{b['mean_log_loss_delta']:+9.4f} | "
			f"{c['mean_log_loss_delta']:+9.4f} | "
			f"{ll_improvement:+11.4f}"
		)
		print(row)
	
	avg_base_ll = total_base_ll_delta / total_samples
	avg_cal_ll = total_cal_ll_delta / total_samples
	
	print("-" * len(header))
	print(f"\nOverall ΔLog Loss (vs Market):")
	print(f"  Baseline:   {avg_base_ll:+.5f}")
	print(f"  Calibrated: {avg_cal_ll:+.5f}")
	print(f"  Improvement:{avg_base_ll - avg_cal_ll:+.5f}")
	print(f"\n{'='*120}\n")


def print_per_class_calibration_comparison(
	baseline_calib: Dict,
	calibrated_calib: Dict,
):
	"""Print side-by-side per-class calibration comparison."""
	print(f"\n{'='*140}")
	print("PER-CLASS CALIBRATION COMPARISON: BASELINE vs CALIBRATED")
	print(f"{'='*140}\n")
	
	class_names = ["Home", "Draw", "Away"]
	
	for class_name in class_names:
		baseline_bins = baseline_calib["results_by_class"][class_name]
		calibrated_bins = calibrated_calib["results_by_class"][class_name]
		
		print(f"\n--- {class_name.upper()} Outcome ---")
		header = (
			f"{'Bin':^6} | {'Prob Range':^12} | {'N':^5} | "
			f"{'Actual':^7} | "
			f"{'Base P':^8} | {'Cal P':^8} | "
			f"{'Base Err':^9} | {'Cal Err':^9} | {'Δ|Err|':^8} | "
			f"{'Base ΔLL':^9} | {'Cal ΔLL':^9} | {'LL Impr':^8}"
		)
		print(header)
		print("-" * len(header))
		
		total_base_err = 0.0
		total_cal_err = 0.0
		total_base_ll = 0.0
		total_cal_ll = 0.0
		total_samples = 0
		
		for b, c in zip(baseline_bins, calibrated_bins):
			n_samples = b["n_samples"]
			prob_range = f"{b['bin_range'][0]:.3f}-{b['bin_range'][1]:.3f}"
			
			# Calibration error improvement (negative = calibrated better)
			err_improvement = abs(b["calibration_error_model"]) - abs(c["calibration_error_model"])
			ll_improvement = b["log_loss_delta"] - c["log_loss_delta"]
			
			total_base_err += abs(b["calibration_error_model"]) * n_samples
			total_cal_err += abs(c["calibration_error_model"]) * n_samples
			total_base_ll += b["log_loss_delta"] * n_samples
			total_cal_ll += c["log_loss_delta"] * n_samples
			total_samples += n_samples
			
			row = (
				f"D{b['bin_idx']+1:2d}    | "
				f"{prob_range:^12} | "
				f"{n_samples:5d} | "
				f"{b['empirical_freq']:7.4f} | "
				f"{b['mean_model_prob']:8.4f} | "
				f"{c['mean_model_prob']:8.4f} | "
				f"{b['calibration_error_model']:+8.4f} | "
				f"{c['calibration_error_model']:+8.4f} | "
				f"{err_improvement:+7.4f} | "
				f"{b['log_loss_delta']:+8.4f} | "
				f"{c['log_loss_delta']:+8.4f} | "
				f"{ll_improvement:+7.4f}"
			)
			print(row)
		
		# Summary for this class
		avg_base_err = total_base_err / total_samples
		avg_cal_err = total_cal_err / total_samples
		avg_base_ll = total_base_ll / total_samples
		avg_cal_ll = total_cal_ll / total_samples
		
		print(f"  {class_name} Summary: MAE Base={avg_base_err:.4f}, MAE Cal={avg_cal_err:.4f}, "
			  f"ΔLL Base={avg_base_ll:+.4f}, ΔLL Cal={avg_cal_ll:+.4f}")
	
	print(f"\n{'='*140}\n")


def plot_per_class_calibration_comparison(
	baseline_calib: Dict,
	calibrated_calib: Dict,
	calibration_params: Dict,
	save_path: Path,
):
	"""Plot per-class calibration comparison."""
	class_names = ["Home", "Draw", "Away"]
	
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle(
		f"Per-Class Calibration Comparison\n"
		f"α={calibration_params['alpha']:.4f}, β={calibration_params['beta']:.4f}",
		fontsize=12, fontweight="bold"
	)
	
	for col_idx, class_name in enumerate(class_names):
		baseline_bins = baseline_calib["results_by_class"][class_name]
		calibrated_bins = calibrated_calib["results_by_class"][class_name]
		
		# Row 1: Calibration curves
		ax = axes[0, col_idx]
		
		market_probs = [b["mean_market_prob"] for b in baseline_bins]
		base_probs = [b["mean_model_prob"] for b in baseline_bins]
		cal_probs = [c["mean_model_prob"] for c in calibrated_bins]
		empirical_freqs = [b["empirical_freq"] for b in baseline_bins]
		
		ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
		ax.plot(market_probs, empirical_freqs, 'o-', linewidth=2, markersize=5, 
				label='Market', color='gray', alpha=0.7)
		ax.plot(base_probs, empirical_freqs, 's-', linewidth=2, markersize=5, 
				label='Baseline', color='blue')
		ax.plot(cal_probs, empirical_freqs, '^-', linewidth=2, markersize=5, 
				label='Calibrated', color='red')
		ax.set_xlabel(f"Predicted P({class_name})")
		ax.set_ylabel(f"Actual Freq ({class_name})")
		ax.set_title(f"{class_name} Calibration")
		ax.legend(fontsize=8)
		ax.grid(True, alpha=0.3)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		
		# Row 2: Calibration error comparison
		ax = axes[1, col_idx]
		
		base_errors = [b["calibration_error_model"] for b in baseline_bins]
		cal_errors = [c["calibration_error_model"] for c in calibrated_bins]
		
		x = np.arange(len(base_errors))
		width = 0.35
		ax.bar(x - width/2, base_errors, width, label='Baseline', color='blue', alpha=0.7)
		ax.bar(x + width/2, cal_errors, width, label='Calibrated', color='red', alpha=0.7)
		ax.axhline(0, color='black', linestyle='--')
		ax.set_xlabel("Decile")
		ax.set_ylabel("Calibration Error (Pred - Actual)")
		ax.set_title(f"{class_name} Calibration Error")
		ax.legend(fontsize=8)
		ax.set_xticks(x)
		ax.set_xticklabels([f"D{i+1}" for i in x], fontsize=8)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Per-class calibration plot saved: {save_path}")


def plot_three_way_calibration_comparison(
	baseline_calib: Dict,
	calibrated_calib: Dict,
	gated_calib: Dict,
	calibration_params: Dict,
	gate_stats: Dict,
	save_path: Path,
):
	"""Plot per-class calibration comparison for all three models."""
	class_names = ["Home", "Draw", "Away"]
	
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle(
		f"Three-Way Calibration Comparison\n"
		f"Calibrated: α={calibration_params['alpha']:.3f}, β={calibration_params['beta']:.3f} | "
		f"Gated: μ(g)=[{gate_stats['gate_mean'][0]:.3f}, {gate_stats['gate_mean'][1]:.3f}, {gate_stats['gate_mean'][2]:.3f}]",
		fontsize=12, fontweight="bold"
	)
	
	for col_idx, class_name in enumerate(class_names):
		baseline_bins = baseline_calib["results_by_class"][class_name]
		calibrated_bins = calibrated_calib["results_by_class"][class_name]
		gated_bins = gated_calib["results_by_class"][class_name]
		
		# Row 1: Calibration curves
		ax = axes[0, col_idx]
		
		market_probs = [b["mean_market_prob"] for b in baseline_bins]
		base_probs = [b["mean_model_prob"] for b in baseline_bins]
		cal_probs = [c["mean_model_prob"] for c in calibrated_bins]
		gated_probs = [g["mean_model_prob"] for g in gated_bins]
		empirical_freqs = [b["empirical_freq"] for b in baseline_bins]
		
		ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Perfect')
		ax.plot(market_probs, empirical_freqs, 'o-', linewidth=2, markersize=5, 
				label='Market', color='gray', alpha=0.7)
		ax.plot(base_probs, empirical_freqs, 's-', linewidth=2, markersize=5, 
				label='Baseline', color='blue')
		ax.plot(cal_probs, empirical_freqs, '^-', linewidth=2, markersize=5, 
				label='Calibrated', color='red')
		ax.plot(gated_probs, empirical_freqs, 'd-', linewidth=2, markersize=5, 
				label='Gated', color='green')
		ax.set_xlabel(f"Predicted P({class_name})")
		ax.set_ylabel(f"Actual Freq ({class_name})")
		ax.set_title(f"{class_name} Calibration")
		ax.legend(fontsize=8)
		ax.grid(True, alpha=0.3)
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		
		# Row 2: Calibration error comparison
		ax = axes[1, col_idx]
		
		base_errors = [b["calibration_error_model"] for b in baseline_bins]
		cal_errors = [c["calibration_error_model"] for c in calibrated_bins]
		gated_errors = [g["calibration_error_model"] for g in gated_bins]
		
		x = np.arange(len(base_errors))
		width = 0.25
		ax.bar(x - width, base_errors, width, label='Baseline', color='blue', alpha=0.7)
		ax.bar(x, cal_errors, width, label='Calibrated', color='red', alpha=0.7)
		ax.bar(x + width, gated_errors, width, label='Gated', color='green', alpha=0.7)
		ax.axhline(0, color='black', linestyle='--')
		ax.set_xlabel("Decile")
		ax.set_ylabel("Calibration Error (Pred - Actual)")
		ax.set_title(f"{class_name} Calibration Error")
		ax.legend(fontsize=8)
		ax.set_xticks(x)
		ax.set_xticklabels([f"D{i+1}" for i in x], fontsize=8)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Three-way calibration plot saved: {save_path}")


# ============================================================================
# BETTING EVALUATION
# ============================================================================

def evaluate_betting_with_sharpe(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_home: np.ndarray,
	odds_draw: np.ndarray,
	odds_away: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
) -> Dict:
	"""
	Evaluate betting performance with Sharpe-weighted portfolio for result prediction.
	
	Args:
		probs: Shape (n, 3) with [home, draw, away] probabilities
		y_true: Shape (n,) with class labels 0=Home, 1=Draw, 2=Away
		odds_*: Shape (n,) with decimal odds for each outcome
		dates: Match dates for daily grouping
		budget_per_day: Daily budget allocation
	
	Returns:
		Dictionary with betting metrics including Sharpe ratio
	"""
	import pandas as pd
	
	odds_matrix = np.stack([odds_home, odds_draw, odds_away], axis=1)  # (n, 3)
	
	# Expected value for each outcome: EV = prob * odds - 1
	ev = probs * odds_matrix - 1  # (n, 3)
	
	# Variance for each outcome: Var = prob * (odds-1)^2 + (1-prob) * 1 - EV^2
	e_x2 = probs * (odds_matrix - 1) ** 2 + (1 - probs) * 1
	var = e_x2 - ev ** 2  # (n, 3)
	
	# For each game, find the outcome with highest EV
	best_outcome = np.argmax(ev, axis=1)  # (n,)
	best_ev = ev[np.arange(len(ev)), best_outcome]  # (n,)
	best_var = var[np.arange(len(var)), best_outcome]  # (n,)
	best_odds = odds_matrix[np.arange(len(odds_matrix)), best_outcome]  # (n,)
	
	# Create DataFrame for grouping
	df = pd.DataFrame({
		"date": dates,
		"best_outcome": best_outcome,
		"y_true": y_true,
		"mu": best_ev,
		"var": best_var,
		"odds": best_odds,
		"eligible": best_ev > 0,
		"won": best_outcome == y_true,
	})
	
	# Simple profit metrics (flat betting)
	simple_metrics = evaluate_profit_result(probs, y_true, odds_home, odds_draw, odds_away)
	
	# Sharpe-weighted daily portfolio
	daily_results = []
	
	for _, group in df.groupby("date"):
		eligible = group[group["eligible"]]
		if len(eligible) == 0:
			daily_results.append(0.0)
			continue
		
		mus = eligible["mu"].values
		vars_ = eligible["var"].values + 1e-6
		raw_weights = np.maximum(0, mus / vars_)
		sum_weights = raw_weights.sum()
		
		if sum_weights > 0:
			norm_weights = raw_weights / sum_weights
			bets = budget_per_day * norm_weights
			profits = np.where(
				eligible["won"].values,
				bets * (eligible["odds"].values - 1),
				-bets,
			)
			daily_results.append(profits.sum())
		else:
			daily_results.append(0.0)
	
	daily = np.array(daily_results)
	
	def sharpe_ratio(x):
		if len(x) == 0:
			return 0.0
		std = x.std()
		return float(x.mean() / std) if std > 0 else 0.0
	
	return {
		**simple_metrics,
		"sharpe_total_profit": float(daily.sum()),
		"sharpe_avg_daily_profit": float(daily.mean()) if len(daily) > 0 else 0.0,
		"sharpe_ratio": sharpe_ratio(daily),
		"n_days": int(len(daily)),
		"win_rate": float(df[df["eligible"]]["won"].mean()) if df["eligible"].sum() > 0 else 0.0,
	}


def print_betting_comparison(
	baseline_metrics: Dict,
	calibrated_metrics: Dict,
	market_metrics: Dict,
):
	"""Print betting comparison table."""
	print(f"\n{'='*100}")
	print("BETTING PERFORMANCE COMPARISON")
	print(f"{'='*100}\n")
	
	metrics_to_show = [
		("n_bets", "Number of Bets", "{:d}"),
		("percent_bets", "% Games Bet", "{:.1f}%"),
		("n_home_bets", "Home Bets", "{:d}"),
		("n_draw_bets", "Draw Bets", "{:d}"),
		("n_away_bets", "Away Bets", "{:d}"),
		("win_rate", "Win Rate", "{:.1%}"),
		("total_profit", "Total Profit (Flat)", "{:+.2f}"),
		("avg_profit", "Avg Profit/Bet", "{:+.4f}"),
		("sharpe_total_profit", "Total Profit (Sharpe)", "{:+.2f}"),
		("sharpe_avg_daily_profit", "Avg Daily Profit", "{:+.4f}"),
		("sharpe_ratio", "Sharpe Ratio", "{:+.4f}"),
		("n_days", "Trading Days", "{:d}"),
	]
	
	header = f"{'Metric':<25} | {'Market':<15} | {'Baseline':<15} | {'Calibrated':<15} | {'Cal vs Base':<15}"
	print(header)
	print("-" * len(header))
	
	for key, name, fmt in metrics_to_show:
		market_val = market_metrics.get(key, 0)
		base_val = baseline_metrics.get(key, 0)
		cal_val = calibrated_metrics.get(key, 0)
		
		# Calculate improvement
		if key in ["sharpe_ratio", "total_profit", "sharpe_total_profit", "avg_profit", "sharpe_avg_daily_profit", "win_rate"]:
			diff = cal_val - base_val
			diff_str = f"{diff:+.4f}" if abs(diff) < 100 else f"{diff:+.2f}"
		else:
			diff = cal_val - base_val
			diff_str = f"{diff:+d}" if isinstance(base_val, int) else f"{diff:+.1f}"
		
		row = (
			f"{name:<25} | "
			f"{fmt.format(market_val):<15} | "
			f"{fmt.format(base_val):<15} | "
			f"{fmt.format(cal_val):<15} | "
			f"{diff_str:<15}"
		)
		print(row)
	
	print(f"\n{'='*100}\n")


def print_betting_comparison_three_way(
	baseline_metrics: Dict,
	calibrated_metrics: Dict,
	gated_metrics: Dict,
	market_metrics: Dict,
):
	"""Print betting comparison table for all three models."""
	print(f"\n{'='*130}")
	print("BETTING PERFORMANCE COMPARISON: BASELINE vs CALIBRATED vs GATED")
	print(f"{'='*130}\n")
	
	metrics_to_show = [
		("n_bets", "Number of Bets", "{:d}"),
		("percent_bets", "% Games Bet", "{:.1f}%"),
		("n_home_bets", "Home Bets", "{:d}"),
		("n_draw_bets", "Draw Bets", "{:d}"),
		("n_away_bets", "Away Bets", "{:d}"),
		("win_rate", "Win Rate", "{:.1%}"),
		("total_profit", "Total Profit (Flat)", "{:+.2f}"),
		("avg_profit", "Avg Profit/Bet", "{:+.4f}"),
		("sharpe_total_profit", "Total Profit (Sharpe)", "{:+.2f}"),
		("sharpe_avg_daily_profit", "Avg Daily Profit", "{:+.4f}"),
		("sharpe_ratio", "Sharpe Ratio", "{:+.4f}"),
		("n_days", "Trading Days", "{:d}"),
	]
	
	header = f"{'Metric':<25} | {'Market':<12} | {'Baseline':<12} | {'Calibrated':<12} | {'Gated':<12} | {'Best Model':<12}"
	print(header)
	print("-" * len(header))
	
	for key, name, fmt in metrics_to_show:
		market_val = market_metrics.get(key, 0)
		base_val = baseline_metrics.get(key, 0)
		cal_val = calibrated_metrics.get(key, 0)
		gated_val = gated_metrics.get(key, 0)
		
		# Determine best model for profit/performance metrics
		if key in ["sharpe_ratio", "total_profit", "sharpe_total_profit", "avg_profit", "sharpe_avg_daily_profit", "win_rate"]:
			vals = {"Baseline": base_val, "Calibrated": cal_val, "Gated": gated_val}
			best = max(vals.keys(), key=lambda k: vals[k])
		else:
			best = "-"
		
		row = (
			f"{name:<25} | "
			f"{fmt.format(market_val):<12} | "
			f"{fmt.format(base_val):<12} | "
			f"{fmt.format(cal_val):<12} | "
			f"{fmt.format(gated_val):<12} | "
			f"{best:<12}"
		)
		print(row)
	
	print(f"\n{'='*130}\n")


def plot_comparison(
	comparison: Dict,
	calibration_params: Dict,
	save_path: Path,
):
	"""Plot comparison of baseline vs calibrated model residuals."""
	baseline = comparison["baseline"]["bins"]
	calibrated = comparison["calibrated"]["bins"]
	
	bin_centers = [b["mean_market_prob"] for b in baseline]
	base_residuals = [b["mean_residual"] for b in baseline]
	cal_residuals = [c["mean_residual"] for c in calibrated]
	base_ll_delta = [b["mean_log_loss_delta"] for b in baseline]
	cal_ll_delta = [c["mean_log_loss_delta"] for c in calibrated]
	
	fig, axes = plt.subplots(2, 2, figsize=(14, 10))
	fig.suptitle(
		f"Calibrated Residual Model Comparison\n"
		f"α={calibration_params['alpha']:.4f}, β={calibration_params['beta']:.4f}, "
		f"b=[{calibration_params['class_bias'][0]:.3f}, {calibration_params['class_bias'][1]:.3f}, {calibration_params['class_bias'][2]:.3f}]",
		fontsize=12, fontweight="bold"
	)
	
	# Plot 1: Mean residuals comparison
	ax = axes[0, 0]
	ax.plot(bin_centers, base_residuals, 'o-', linewidth=2, markersize=6, label='Baseline', color='blue')
	ax.plot(bin_centers, cal_residuals, 's-', linewidth=2, markersize=6, label='Calibrated', color='red')
	ax.axhline(0, color='black', linestyle='--', alpha=0.7)
	ax.set_xlabel("Market Prob (True Outcome)")
	ax.set_ylabel("Mean Residual (Model - Market)")
	ax.set_title("Residual by Decile")
	ax.legend()
	ax.grid(True, alpha=0.3)
	
	# Plot 2: Log loss delta comparison
	ax = axes[0, 1]
	x = np.arange(len(base_ll_delta))
	width = 0.35
	ax.bar(x - width/2, base_ll_delta, width, label='Baseline', color='blue', alpha=0.7)
	ax.bar(x + width/2, cal_ll_delta, width, label='Calibrated', color='red', alpha=0.7)
	ax.axhline(0, color='black', linestyle='--')
	ax.set_xlabel("Decile")
	ax.set_ylabel("ΔLog Loss (Model - Market)")
	ax.set_title("Log Loss Delta by Decile")
	ax.legend()
	ax.set_xticks(x)
	ax.set_xticklabels([f"D{i+1}" for i in x])
	
	# Plot 3: Residual improvement
	ax = axes[1, 0]
	improvements = [abs(b["mean_residual"]) - abs(c["mean_residual"]) for b, c in zip(baseline, calibrated)]
	colors = ['green' if imp > 0 else 'red' for imp in improvements]
	ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.7)
	ax.axhline(0, color='black', linestyle='--')
	ax.set_xlabel("Decile")
	ax.set_ylabel("Improvement in |Residual|")
	ax.set_title("Residual Magnitude Improvement (Green = Better)")
	ax.set_xticks(range(len(improvements)))
	ax.set_xticklabels([f"D{i+1}" for i in range(len(improvements))])
	
	# Plot 4: Log loss improvement
	ax = axes[1, 1]
	ll_improvements = [b["mean_log_loss_delta"] - c["mean_log_loss_delta"] for b, c in zip(baseline, calibrated)]
	colors = ['green' if imp > 0 else 'red' for imp in ll_improvements]
	ax.bar(range(len(ll_improvements)), ll_improvements, color=colors, alpha=0.7)
	ax.axhline(0, color='black', linestyle='--')
	ax.set_xlabel("Decile")
	ax.set_ylabel("Log Loss Improvement")
	ax.set_title("Log Loss Improvement (Green = Calibrated Better)")
	ax.set_xticks(range(len(ll_improvements)))
	ax.set_xticklabels([f"D{i+1}" for i in range(len(ll_improvements))])
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Comparison plot saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
	"""Main test pipeline."""
	print(f"\n{'='*80}")
	print("TEST: LEARNABLE CALIBRATION/SHRINKAGE IN LOG-SPACE")
	print(f"{'='*80}\n")
	
	# Paths
	data_path = project_root / "data" / "training" / "understat_df.parquet"
	models_dir = project_root / "data" / "models"
	plots_dir = project_root / "data" / "plots"
	plots_dir.mkdir(exist_ok=True, parents=True)
	
	# Load existing model config
	config_path = models_dir / "result_architecture_config.json"
	with open(config_path, 'r') as f:
		config = json.load(f)
	
	print(f"Loading config from: {config_path}")
	print(f"  Input dim: {config['input_dim']}")
	print(f"  Hidden layers: {config['hidden_layers']}")
	print(f"  Dropout: {config['dropout']:.4f}")
	print(f"  LR: {config['lr']:.6f}")
	print(f"  Features: {len(config['feature_cols'])}\n")
	
	# Load data
	print("Loading data...")
	df = load_frame(data_path)
	df = filter_min_history(df)
	df = add_targets_and_implied_result(df)
	
	# Get seasons
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	test_season = seasons[-1]
	val_season = seasons[-2]
	train_seasons = seasons[:-2]
	
	print(f"Train seasons: {train_seasons[0]}..{train_seasons[-1]}")
	print(f"Val season: {val_season}")
	print(f"Test season: {test_season}\n")
	
	feature_cols = config["feature_cols"]
	
	# Prepare data
	print("Preparing data...")
	train_data = prepare_data_result(df, feature_cols, train_seasons, fit_scaler=True)
	val_data = prepare_data_result(df, feature_cols, [val_season], scaler=train_data["scaler"])
	test_data = prepare_data_result(df, feature_cols, [test_season], scaler=train_data["scaler"])
	
	print(f"Train samples: {len(train_data['y'])}")
	print(f"Val samples: {len(val_data['y'])}")
	print(f"Test samples: {len(test_data['y'])}\n")
	
	# Create data loaders
	batch_size = config.get("batch_size", 256)
	train_loader = to_loader(train_data, batch_size, shuffle=True, device=DEVICE, task_type="multiclass")
	val_loader = to_loader(val_data, batch_size, shuffle=False, device=DEVICE, task_type="multiclass")
	
	# Build categorical config
	num_leagues = get_num_leagues(df)
	cat_config = CategoricalConfig(
		num_leagues=num_leagues,
		league_embed_dim=config.get("cat_config", {}).get("league_embed_dim", 3),
	)
	
	# ========================================================================
	# TRAIN BASELINE MODEL (original residual model)
	# ========================================================================
	print(f"\n{'='*80}")
	print("TRAINING BASELINE MODEL (original r(x) + log p_mkt)")
	print(f"{'='*80}\n")
	
	from training.train_utils import train_model, TrainConfig
	
	baseline_config = TrainConfig(
		input_dim=len(feature_cols),
		hidden_layers=config["hidden_layers"],
		dropout=config["dropout"],
		norm=config.get("norm", "none"),
		lr=config["lr"],
		weight_decay=config.get("weight_decay", 1e-4),
		lambda_repulsion=config.get("lambda_repulsion", 0.0),
		lambda_corr=config.get("lambda_corr", 0.0),
		activation=config.get("activation", "relu"),
		beta1=config.get("beta1", 0.9),
		epochs=config.get("final_epochs", 50),
		patience=15,
		batch_size=batch_size,
		task_type="multiclass",
		cat_config=cat_config,
	)
	
	baseline_model, baseline_history, baseline_best_loss = train_model(
		baseline_config, train_loader, val_loader, device=DEVICE, verbose=True
	)
	print(f"\nBaseline best val loss: {baseline_best_loss:.5f}")
	
	# ========================================================================
	# TRAIN CALIBRATED MODEL
	# ========================================================================
	print(f"\n{'='*80}")
	print("TRAINING CALIBRATED MODEL (α·log p_mkt + β·r(x) + b)")
	print(f"{'='*80}\n")
	
	# Create base MLP for the calibrated model
	base_mlp = MLP(
		input_dim=len(feature_cols),
		hidden_layers=config["hidden_layers"],
		dropout=config["dropout"],
		norm=config.get("norm", "none"),
		activation=config.get("activation", "relu"),
		output_dim=3,
		cat_config=cat_config,
	)
	
	# Wrap in calibrated model
	calibrated_model = CalibratedResidualModel(
		base_model=base_mlp,
		n_classes=3,
		init_alpha=1.0,  # Start at original behavior
		init_beta=1.0,
	)
	
	calibrated_model, cal_history, cal_best_loss = train_calibrated_model(
		calibrated_model,
		train_loader,
		val_loader,
		lr=config["lr"],
		weight_decay=config.get("weight_decay", 1e-4),
		beta1=config.get("beta1", 0.9),
		epochs=config.get("final_epochs", 50),
		patience=15,
		device=DEVICE,
		verbose=True,
	)
	
	final_params = calibrated_model.get_calibration_params()
	print(f"\nCalibrated best val loss: {cal_best_loss:.5f}")
	print(f"Final calibration parameters:")
	print(f"  α (market temperature): {final_params['alpha']:.4f}")
	print(f"  β (residual scaling):   {final_params['beta']:.4f}")
	print(f"  b (class biases):       Home={final_params['class_bias'][0]:.4f}, "
		  f"Draw={final_params['class_bias'][1]:.4f}, Away={final_params['class_bias'][2]:.4f}")
	
	# ========================================================================
	# TRAIN CONTEXTUAL GATED MODEL
	# ========================================================================
	print(f"\n{'='*80}")
	print("TRAINING CONTEXTUAL GATED MODEL (log p_mkt + g·r(x), g = σ(gate_head([h(x), mkt_features])))")
	print(f"{'='*80}\n")
	
	# Create gated model (it creates its own base MLP internally)
	gated_model = ContextualGatedResidualModel(
		input_dim=len(feature_cols),
		hidden_layers=config["hidden_layers"],
		n_classes=3,
		cat_config=cat_config,
		gate_hidden_dim=64,
		dropout=config["dropout"],
		norm=config.get("norm", "none"),
		activation=config.get("activation", "relu"),
	)
	
	gated_model, gated_history, gated_best_loss = train_contextual_gated_model(
		gated_model,
		train_loader,
		val_loader,
		lr=config["lr"],
		weight_decay=config.get("weight_decay", 1e-4),
		beta1=config.get("beta1", 0.9),
		epochs=config.get("final_epochs", 50),
		patience=15,
		device=DEVICE,
		gate_reg_weight=0.01,  # Small regularization to push gates toward 0.5
		verbose=True,
	)
	
	# Get final gate statistics on validation set
	gated_model.eval()
	val_gate_stats = gated_model.get_gate_stats(
		torch.tensor(val_data["X"], dtype=torch.float32, device=DEVICE),
		torch.tensor(val_data["cat_features"], dtype=torch.long, device=DEVICE),
		torch.tensor(val_data["implied"], dtype=torch.float32, device=DEVICE),
	)
	
	print(f"\nGated best val loss: {gated_best_loss:.5f}")
	print(f"Final gate statistics (validation set):")
	print(f"  Gate means:  Home={val_gate_stats['gate_mean'][0]:.4f}, "
		  f"Draw={val_gate_stats['gate_mean'][1]:.4f}, Away={val_gate_stats['gate_mean'][2]:.4f}")
	print(f"  Gate stds:   Home={val_gate_stats['gate_std'][0]:.4f}, "
		  f"Draw={val_gate_stats['gate_std'][1]:.4f}, Away={val_gate_stats['gate_std'][2]:.4f}")
	print(f"  Gate ranges: Home=[{val_gate_stats['gate_min'][0]:.3f}, {val_gate_stats['gate_max'][0]:.3f}], "
		  f"Draw=[{val_gate_stats['gate_min'][1]:.3f}, {val_gate_stats['gate_max'][1]:.3f}], "
		  f"Away=[{val_gate_stats['gate_min'][2]:.3f}, {val_gate_stats['gate_max'][2]:.3f}]")
	
	# ========================================================================
	# EVALUATE ON TEST SET
	# ========================================================================
	print(f"\n{'='*80}")
	print("EVALUATING ON TEST SET")
	print(f"{'='*80}\n")
	
	# Get baseline predictions
	baseline_model.eval()
	with torch.no_grad():
		X_test = torch.tensor(test_data["X"], dtype=torch.float32, device=DEVICE)
		cat_test = torch.tensor(test_data["cat_features"], dtype=torch.long, device=DEVICE)
		implied_test = torch.tensor(test_data["implied"], dtype=torch.float32, device=DEVICE)
		
		# Baseline: r(x) + log(p_mkt) -> softmax
		baseline_logits = baseline_model(X_test, cat_test)
		implied_log = _log_softmax_from_implied(implied_test)
		baseline_probs = F.softmax(baseline_logits + implied_log, dim=1).cpu().numpy()
	
	# Get calibrated predictions
	calibrated_probs = get_calibrated_model_predictions(
		calibrated_model,
		test_data["X"],
		test_data["cat_features"],
		test_data["implied"],
		DEVICE,
	)
	
	# Get gated model predictions
	gated_model.eval()
	with torch.no_grad():
		gated_logits = gated_model(X_test, cat_test, implied_test)
		gated_probs = F.softmax(gated_logits, dim=1).cpu().numpy()
	
	# Get gate statistics on test set
	test_gate_stats = gated_model.get_gate_stats(X_test, cat_test, implied_test)
	
	# Compute log losses
	from sklearn.metrics import log_loss as sklearn_log_loss
	
	baseline_ll = sklearn_log_loss(test_data["y"], baseline_probs, labels=[0, 1, 2])
	calibrated_ll = sklearn_log_loss(test_data["y"], calibrated_probs, labels=[0, 1, 2])
	gated_ll = sklearn_log_loss(test_data["y"], gated_probs, labels=[0, 1, 2])
	market_ll = sklearn_log_loss(test_data["y"], test_data["implied"], labels=[0, 1, 2])
	
	print(f"Test Set Log Loss:")
	print(f"  Market (implied):     {market_ll:.5f}")
	print(f"  Baseline model:       {baseline_ll:.5f} (Δ vs market: {baseline_ll - market_ll:+.5f})")
	print(f"  Calibrated model:     {calibrated_ll:.5f} (Δ vs market: {calibrated_ll - market_ll:+.5f})")
	print(f"  Gated model:          {gated_ll:.5f} (Δ vs market: {gated_ll - market_ll:+.5f})")
	print(f"\n  Calibrated vs Base:   {calibrated_ll - baseline_ll:+.5f}")
	print(f"  Gated vs Base:        {gated_ll - baseline_ll:+.5f}")
	print(f"  Gated vs Calibrated:  {gated_ll - calibrated_ll:+.5f}")
	
	print(f"\nTest Set Gate Statistics:")
	print(f"  Gate means:  Home={test_gate_stats['gate_mean'][0]:.4f}, "
		  f"Draw={test_gate_stats['gate_mean'][1]:.4f}, Away={test_gate_stats['gate_mean'][2]:.4f}")
	print(f"  Gate stds:   Home={test_gate_stats['gate_std'][0]:.4f}, "
		  f"Draw={test_gate_stats['gate_std'][1]:.4f}, Away={test_gate_stats['gate_std'][2]:.4f}")
	
	# ========================================================================
	# BETTING EVALUATION
	# ========================================================================
	print(f"\n{'='*80}")
	print("BETTING PERFORMANCE EVALUATION")
	print(f"{'='*80}")
	
	# Market betting metrics
	market_betting = evaluate_betting_with_sharpe(
		test_data["implied"],
		test_data["y"],
		test_data["odds_home"],
		test_data["odds_draw"],
		test_data["odds_away"],
		test_data["dates"],
	)
	
	# Baseline betting metrics
	baseline_betting = evaluate_betting_with_sharpe(
		baseline_probs,
		test_data["y"],
		test_data["odds_home"],
		test_data["odds_draw"],
		test_data["odds_away"],
		test_data["dates"],
	)
	
	# Calibrated betting metrics
	calibrated_betting = evaluate_betting_with_sharpe(
		calibrated_probs,
		test_data["y"],
		test_data["odds_home"],
		test_data["odds_draw"],
		test_data["odds_away"],
		test_data["dates"],
	)
	
	# Gated betting metrics
	gated_betting = evaluate_betting_with_sharpe(
		gated_probs,
		test_data["y"],
		test_data["odds_home"],
		test_data["odds_draw"],
		test_data["odds_away"],
		test_data["dates"],
	)
	
	print_betting_comparison_three_way(
		baseline_betting, calibrated_betting, gated_betting, market_betting
	)
	
	# ========================================================================
	# RESIDUAL ANALYSIS COMPARISON
	# ========================================================================
	print(f"\n{'='*80}")
	print("RESIDUAL ANALYSIS BY DECILE")
	print(f"{'='*80}")
	
	comparison = compare_residual_analysis(
		test_data["y"],
		baseline_probs,
		calibrated_probs,
		test_data["implied"],
		n_bins=10,
	)
	
	print_comparison_table(comparison)
	
	# Plot comparison
	plot_path = plots_dir / "calibrated_vs_baseline_residuals.png"
	plot_comparison(comparison, final_params, plot_path)
	
	# ========================================================================
	# CALIBRATION ANALYSIS
	# ========================================================================
	print(f"\n{'='*80}")
	print("CALIBRATION ANALYSIS")
	print(f"{'='*80}")
	
	print("\n--- BASELINE MODEL ---")
	baseline_calib = analyze_calibration_by_predicted_prob(
		test_data["y"], baseline_probs, test_data["implied"], "multiclass", 10
	)
	print_calibration_table(baseline_calib, "multiclass")
	
	print("\n--- CALIBRATED MODEL ---")
	calibrated_calib = analyze_calibration_by_predicted_prob(
		test_data["y"], calibrated_probs, test_data["implied"], "multiclass", 10
	)
	print_calibration_table(calibrated_calib, "multiclass")
	
	print("\n--- GATED MODEL ---")
	gated_calib = analyze_calibration_by_predicted_prob(
		test_data["y"], gated_probs, test_data["implied"], "multiclass", 10
	)
	print_calibration_table(gated_calib, "multiclass")
	
	# ========================================================================
	# PER-CLASS CALIBRATION COMPARISON (THREE-WAY)
	# ========================================================================
	print(f"\n{'='*80}")
	print("PER-CLASS CALIBRATION COMPARISON (THREE-WAY)")
	print(f"{'='*80}")
	
	print_per_class_calibration_comparison(baseline_calib, calibrated_calib)
	
	# Plot three-way per-class calibration comparison
	three_way_plot_path = plots_dir / "three_way_calibration_comparison.png"
	plot_three_way_calibration_comparison(
		baseline_calib, calibrated_calib, gated_calib, 
		final_params, test_gate_stats, three_way_plot_path
	)
	
	# Also keep the old two-way plot for reference
	per_class_plot_path = plots_dir / "per_class_calibration_comparison.png"
	plot_per_class_calibration_comparison(
		baseline_calib, calibrated_calib, final_params, per_class_plot_path
	)
	
	# ========================================================================
	# PARAMETER EVOLUTION PLOT
	# ========================================================================
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	fig.suptitle("Calibration Parameter Evolution During Training", fontsize=12, fontweight="bold")
	
	epochs = range(1, len(cal_history["alpha"]) + 1)
	
	axes[0].plot(epochs, cal_history["alpha"], 'b-', linewidth=2)
	axes[0].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='α=1 (original)')
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("α (Market Temperature)")
	axes[0].set_title("α Evolution")
	axes[0].legend()
	axes[0].grid(True, alpha=0.3)
	
	axes[1].plot(epochs, cal_history["beta"], 'g-', linewidth=2)
	axes[1].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='β=1 (original)')
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("β (Residual Scaling)")
	axes[1].set_title("β Evolution")
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)
	
	biases = np.array(cal_history["bias"])
	axes[2].plot(epochs, biases[:, 0], 'b-', linewidth=2, label='Home')
	axes[2].plot(epochs, biases[:, 1], 'g-', linewidth=2, label='Draw')
	axes[2].plot(epochs, biases[:, 2], 'r-', linewidth=2, label='Away')
	axes[2].axhline(0.0, color='black', linestyle='--', alpha=0.7)
	axes[2].set_xlabel("Epoch")
	axes[2].set_ylabel("b (Class Bias)")
	axes[2].set_title("Class Bias Evolution")
	axes[2].legend()
	axes[2].grid(True, alpha=0.3)
	
	plt.tight_layout()
	param_plot_path = plots_dir / "calibration_params_evolution.png"
	plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"\nParameter evolution plot saved: {param_plot_path}")
	
	# ========================================================================
	# GATE EVOLUTION PLOT
	# ========================================================================
	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle("Contextual Gate Evolution During Training", fontsize=12, fontweight="bold")
	
	gate_epochs = range(1, len(gated_history["gate_mean"]) + 1)
	gate_means = np.array(gated_history["gate_mean"])
	gate_stds = np.array(gated_history["gate_std"])
	
	# Gate means evolution
	axes[0].plot(gate_epochs, gate_means[:, 0], 'b-', linewidth=2, label='Home')
	axes[0].plot(gate_epochs, gate_means[:, 1], 'g-', linewidth=2, label='Draw')
	axes[0].plot(gate_epochs, gate_means[:, 2], 'r-', linewidth=2, label='Away')
	axes[0].axhline(0.5, color='black', linestyle='--', alpha=0.7, label='g=0.5 (neutral)')
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Gate Mean")
	axes[0].set_title("Gate Mean Evolution by Class")
	axes[0].legend()
	axes[0].grid(True, alpha=0.3)
	axes[0].set_ylim(0, 1)
	
	# Gate stds evolution
	axes[1].plot(gate_epochs, gate_stds[:, 0], 'b-', linewidth=2, label='Home')
	axes[1].plot(gate_epochs, gate_stds[:, 1], 'g-', linewidth=2, label='Draw')
	axes[1].plot(gate_epochs, gate_stds[:, 2], 'r-', linewidth=2, label='Away')
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Gate Std")
	axes[1].set_title("Gate Variance Evolution by Class")
	axes[1].legend()
	axes[1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	gate_plot_path = plots_dir / "gate_evolution.png"
	plt.savefig(gate_plot_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Gate evolution plot saved: {gate_plot_path}")
	
	# ========================================================================
	# GATE ANALYSIS BY MARKET ENTROPY
	# ========================================================================
	print(f"\n{'='*80}")
	print("GATE ANALYSIS BY MARKET ENTROPY")
	print(f"{'='*80}\n")
	
	# Compute gate values and market features for test set
	with torch.no_grad():
		gate_values = test_gate_stats["gate_values"]
		
		# Compute entropy of implied probs
		implied_np = test_data["implied"]
		entropy = -np.sum(implied_np * np.log(implied_np + 1e-8), axis=1)
		
		# Bin by entropy
		entropy_bins = np.percentile(entropy, [0, 25, 50, 75, 100])
		entropy_labels = ["Very Low", "Low", "Medium", "High"]
		
		print(f"Gate Values by Market Entropy (uncertainty):")
		print(f"  {'Entropy Bin':<15} | {'N':>6} | {'Gate H':>8} | {'Gate D':>8} | {'Gate A':>8} | {'Avg Gate':>8}")
		print(f"  {'-'*65}")
		
		for i in range(len(entropy_labels)):
			mask = (entropy >= entropy_bins[i]) & (entropy < entropy_bins[i+1] if i < 3 else entropy <= entropy_bins[i+1])
			if mask.sum() > 0:
				gate_subset = gate_values[mask]
				print(f"  {entropy_labels[i]:<15} | {mask.sum():>6d} | {gate_subset[:, 0].mean():>8.4f} | "
					  f"{gate_subset[:, 1].mean():>8.4f} | {gate_subset[:, 2].mean():>8.4f} | "
					  f"{gate_subset.mean():>8.4f}")
	
	# Plot gate distribution
	fig, axes = plt.subplots(1, 3, figsize=(15, 4))
	fig.suptitle("Gate Value Distribution by Class (Test Set)", fontsize=12, fontweight="bold")
	
	class_names = ["Home", "Draw", "Away"]
	colors = ["blue", "green", "red"]
	
	for i, (name, color) in enumerate(zip(class_names, colors)):
		axes[i].hist(gate_values[:, i], bins=50, color=color, alpha=0.7, edgecolor='black')
		axes[i].axvline(gate_values[:, i].mean(), color='black', linestyle='--', 
					   linewidth=2, label=f'Mean={gate_values[:, i].mean():.3f}')
		axes[i].axvline(0.5, color='gray', linestyle=':', linewidth=2, label='g=0.5')
		axes[i].set_xlabel(f"Gate Value ({name})")
		axes[i].set_ylabel("Frequency")
		axes[i].set_title(f"{name} Gate Distribution")
		axes[i].legend()
		axes[i].set_xlim(0, 1)
	
	plt.tight_layout()
	gate_dist_path = plots_dir / "gate_distribution.png"
	plt.savefig(gate_dist_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"\nGate distribution plot saved: {gate_dist_path}")
	
	# ========================================================================
	# SUMMARY
	# ========================================================================
	print(f"\n{'='*80}")
	print("SUMMARY")
	print(f"{'='*80}\n")
	
	print("Learnable Calibration Parameters:")
	print(f"  α = {final_params['alpha']:.4f}")
	if final_params['alpha'] < 1:
		print("    → α < 1 softens market distribution (less confident in market)")
	else:
		print("    → α > 1 sharpens market distribution (more confident in market)")
	
	print(f"  β = {final_params['beta']:.4f}")
	if final_params['beta'] < 1:
		print("    → β < 1 shrinks residual corrections (trusts model less)")
	else:
		print("    → β > 1 expands residual corrections (trusts model more)")
	
	print(f"  b = [{final_params['class_bias'][0]:.4f}, {final_params['class_bias'][1]:.4f}, {final_params['class_bias'][2]:.4f}]")
	print("    → Per-class systematic bias adjustment\n")
	
	# Log Loss Summary
	ll_improvement_cal = baseline_ll - calibrated_ll
	ll_improvement_gated = baseline_ll - gated_ll
	
	print("LOG LOSS PERFORMANCE:")
	print(f"  Baseline:      {baseline_ll:.5f}")
	print(f"  Calibrated:    {calibrated_ll:.5f} (Δ = {-ll_improvement_cal:+.5f})")
	print(f"  Gated:         {gated_ll:.5f} (Δ = {-ll_improvement_gated:+.5f})")
	
	best_ll_model = min([("Baseline", baseline_ll), ("Calibrated", calibrated_ll), ("Gated", gated_ll)], key=lambda x: x[1])
	print(f"  → Best: {best_ll_model[0]} ({best_ll_model[1]:.5f})")
	
	# Betting Summary
	print("\nBETTING PERFORMANCE:")
	print(f"  {'Model':<12} | {'Sharpe':>8} | {'Profit':>10} | {'Bets':>5}")
	print(f"  {'-'*45}")
	print(f"  {'Baseline':<12} | {baseline_betting['sharpe_ratio']:>+8.4f} | {baseline_betting['sharpe_total_profit']:>+10.2f} | {baseline_betting['n_bets']:>5d}")
	print(f"  {'Calibrated':<12} | {calibrated_betting['sharpe_ratio']:>+8.4f} | {calibrated_betting['sharpe_total_profit']:>+10.2f} | {calibrated_betting['n_bets']:>5d}")
	print(f"  {'Gated':<12} | {gated_betting['sharpe_ratio']:>+8.4f} | {gated_betting['sharpe_total_profit']:>+10.2f} | {gated_betting['n_bets']:>5d}")
	
	best_sharpe = max([("Baseline", baseline_betting), ("Calibrated", calibrated_betting), ("Gated", gated_betting)], key=lambda x: x[1]["sharpe_ratio"])
	best_profit = max([("Baseline", baseline_betting), ("Calibrated", calibrated_betting), ("Gated", gated_betting)], key=lambda x: x[1]["sharpe_total_profit"])
	
	print(f"\n  → Best Sharpe: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:+.4f})")
	print(f"  → Best Profit: {best_profit[0]} ({best_profit[1]['sharpe_total_profit']:+.2f})")
	
	# Gate Analysis Summary
	print("\nGATED MODEL ANALYSIS:")
	print(f"  Gate means (test):  H={test_gate_stats['gate_mean'][0]:.4f}, D={test_gate_stats['gate_mean'][1]:.4f}, A={test_gate_stats['gate_mean'][2]:.4f}")
	print(f"  Gate stds (test):   H={test_gate_stats['gate_std'][0]:.4f}, D={test_gate_stats['gate_std'][1]:.4f}, A={test_gate_stats['gate_std'][2]:.4f}")
	
	avg_gate = np.mean(test_gate_stats['gate_mean'])
	if avg_gate < 0.5:
		print(f"  → Average gate {avg_gate:.4f} < 0.5: Model learns to REDUCE corrections on average")
	else:
		print(f"  → Average gate {avg_gate:.4f} ≥ 0.5: Model learns to APPLY corrections on average")
	
	gate_variance = np.mean(test_gate_stats['gate_std'])
	if gate_variance > 0.1:
		print(f"  → High gate variance ({gate_variance:.4f}): Model learns CONTEXT-DEPENDENT gating")
	else:
		print(f"  → Low gate variance ({gate_variance:.4f}): Gate is nearly constant (like calibrated model)")
	
	print(f"\n{'='*80}\n")


if __name__ == "__main__":
	main()
