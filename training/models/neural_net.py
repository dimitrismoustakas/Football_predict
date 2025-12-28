"""
Neural network architecture and loss functions for football match prediction.
"""


from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeGLU(nn.Module):
	"""Gated GELU activation with linear projection."""

	def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True):
		super().__init__()
		self.proj = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		v, g = self.proj(x).chunk(2, dim=-1)
		return v * F.gelu(g)


class MLP(nn.Module):
	"""Flexible MLP with configurable layers, dropout, normalization, and activation."""

	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
	):
		super().__init__()
		layers = []
		prev = input_dim
		NormClass = {"none": None, "bn": nn.BatchNorm1d, "ln": nn.LayerNorm}.get(norm)

		# Activation factory
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
			if activation == "geglu":
				# GeGLU combines linear + activation in one module
				layers.append(GeGLU(prev, h))
			else:
				layers.append(nn.Linear(prev, h))
				if NormClass is not None:
					layers.append(NormClass(h))
				layers.append(get_activation())
			layers.append(nn.Dropout(dropout))
			prev = h
		layers.append(nn.Linear(prev, 1))
		self.net = nn.Sequential(*layers)
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation

	def forward(self, x):
		return self.net(x)


@dataclass
class TrainConfig:
	"""Configuration for model training."""
	input_dim: int
	hidden_layers: List[int]
	dropout: float
	norm: str
	lr: float
	weight_decay: float
	lambda_repulsion: float
	lambda_corr: float
	activation: str = "relu"
	scheduler_type: str = "plateau"
	epochs: int = 100
	patience: int = 15
	batch_size: int = 128


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================


def _logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert probability to logits with numerical stability."""
	p = torch.clamp(p, eps, 1 - eps)
	return torch.log(p) - torch.log(1 - p)


def batch_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Compute batch correlation between two tensors."""
	x = x - x.mean()
	y = y - y.mean()
	vx = x.var(unbiased=False) + eps
	vy = y.var(unbiased=False) + eps
	cov = (x * y).mean()
	return cov / torch.sqrt(vx * vy)


def logits_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Approximate Corr(pred_logits, implied_logits | y)."""
	pred = pred_logits.view(-1)
	impl = implied_logits.view(-1)
	y = target.view(-1)
	total_n = pred.shape[0]
	rho_weighted = pred.new_tensor(0.0)
	weighted_sum = 0.0

	for r in (0, 1):
		mask = y == r
		n_r = int(mask.sum().item())
		if n_r > 1:
			pred_r = pred[mask]
			impl_r = impl[mask]
			rho_r = batch_corr(pred_r, impl_r, eps=eps)
			w_r = n_r / float(total_n)
			rho_weighted = rho_weighted + w_r * rho_r
			weighted_sum += w_r

	if weighted_sum == 0:
		return batch_corr(pred, impl, eps=eps)

	return rho_weighted


def residual_market_loss(
	residuals_logits: torch.Tensor,
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
) -> torch.Tensor:
	"""Loss function for residual market model."""
	implied_logit = _logits(implied_prob)
	pred_logits = residuals_logits + implied_logit
	base = F.binary_cross_entropy_with_logits(pred_logits, target)

	if lambda_repulsion <= 0:
		return base

	pred_prob = torch.sigmoid(pred_logits)
	repulsion = (pred_prob - implied_prob) ** 2
	repulsion = repulsion.mean()
	return base - lambda_repulsion * repulsion


def residual_market_loss_corr(
	residuals_logits: torch.Tensor,
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	conditional_corr: bool = True,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss function for residual market model with correlation penalty."""
	implied_logit = _logits(implied_prob)
	pred_logits = residuals_logits + implied_logit
	base = F.binary_cross_entropy_with_logits(pred_logits, target)
	loss = base

	if lambda_repulsion > 0.0:
		pred_prob = torch.sigmoid(pred_logits)
		repulsion = (pred_prob - implied_prob) ** 2
		repulsion = repulsion.mean()
		loss = loss - lambda_repulsion * repulsion

	if lambda_corr > 0.0:
		if conditional_corr:
			rho = logits_conditional_corr(pred_logits, implied_logit, target, eps=eps)
		else:
			rho = batch_corr(pred_logits.view(-1), implied_logit.view(-1), eps=eps)

		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty

	return loss
