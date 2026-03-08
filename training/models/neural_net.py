"""
Neural network architecture and loss functions for match-result prediction.

Architecture: contextual gated residual model.
The network starts from bookmaker implied probabilities and learns when to
deviate from them using a context-dependent gate.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CategoricalConfig:
	"""Configuration for categorical features."""

	num_leagues: int = 5
	league_embed_dim: int = 3


class CategoricalEmbedder(nn.Module):
	"""Embed league id and append promoted-team flags."""

	def __init__(self, cat_config: CategoricalConfig):
		super().__init__()
		self.cat_config = cat_config
		self.league_embed = nn.Embedding(cat_config.num_leagues, cat_config.league_embed_dim)
		self.output_dim = cat_config.league_embed_dim + 2

	def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
		league_idx = cat_features[:, 0].long()
		promoted = cat_features[:, 1:3].float()
		league_emb = self.league_embed(league_idx)
		return torch.cat([league_emb, promoted], dim=-1)


class MLPWithHiddenAccess(nn.Module):
	"""MLP that exposes its last hidden representation."""

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

		self.cat_embedder = None
		total_input_dim = input_dim
		if cat_config is not None:
			self.cat_embedder = CategoricalEmbedder(cat_config)
			total_input_dim = input_dim + self.cat_embedder.output_dim

		layers = []
		prev = total_input_dim
		NormClass = {"none": None, "bn": nn.BatchNorm1d, "ln": nn.LayerNorm}.get(norm)

		def get_activation():
			if activation == "relu":
				return nn.ReLU()
			if activation == "silu":
				return nn.SiLU()
			if activation == "gelu":
				return nn.GELU()
			raise ValueError(f"Unknown activation: {activation}")

		for width in hidden_layers:
			layers.append(nn.Linear(prev, width))
			if NormClass is not None:
				layers.append(NormClass(width))
			layers.append(get_activation())
			layers.append(nn.Dropout(dropout))
			prev = width

		self.hidden_net = nn.Sequential(*layers)
		self.final_layer = nn.Linear(prev, output_dim)
		self.hidden_dim = prev
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.output_dim = output_dim
		self.cat_config = cat_config

	def forward(self, x: torch.Tensor, cat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
		h = self.get_hidden(x, cat_features)
		return self.final_layer(h)

	def get_hidden(self, x: torch.Tensor, cat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
		if self.cat_embedder is not None:
			if cat_features is None:
				raise ValueError("cat_features required when model has cat_config")
			cat_emb = self.cat_embedder(cat_features)
			x = torch.cat([x, cat_emb], dim=-1)
		return self.hidden_net(x)


class GatedResidualModel(nn.Module):
	"""Multiclass gated residual model for Home/Draw/Away prediction."""

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
		gate_target_budget: float = 0.2,
	):
		super().__init__()
		self.n_classes = n_classes
		self.gate_target_budget = gate_target_budget

		self.base_model = MLPWithHiddenAccess(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			norm=norm,
			activation=activation,
			output_dim=n_classes,
			cat_config=cat_config,
		)

		self.market_feature_dim = market_feature_dim + 3
		gate_input_dim = self.base_model.hidden_dim + self.market_feature_dim
		self.gate_head = nn.Sequential(
			nn.Linear(gate_input_dim, gate_hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 0.5),
			nn.Linear(gate_hidden_dim, n_classes),
		)

		init_bias = math.log(gate_target_budget / (1 - gate_target_budget))
		self.gate_bias = nn.Parameter(torch.full((n_classes,), init_bias))

		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.cat_config = cat_config
		self.gate_hidden_dim = gate_hidden_dim

	def _compute_market_features(
		self,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		eps = 1e-6
		entropy = -torch.sum(implied_probs * torch.log(implied_probs + eps), dim=-1, keepdim=True)
		entropy = entropy / math.log(self.n_classes)
		max_prob = implied_probs.max(dim=-1, keepdim=True)[0]
		if raw_margin.dim() == 1:
			raw_margin = raw_margin.unsqueeze(-1)
		return torch.cat([implied_probs, entropy, max_prob, raw_margin], dim=-1)

	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_probs: Optional[torch.Tensor] = None,
		raw_margin: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		h = self.base_model.get_hidden(x, cat_features)
		residual_logits = self.base_model.final_layer(h)

		if implied_probs is None:
			return residual_logits
		if raw_margin is None:
			raise ValueError("raw_margin is required when implied_probs is provided")

		market_features = self._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = self.gate_head(gate_input)
		gate = torch.sigmoid(gate_logits + self.gate_bias)
		implied_log = _log_softmax_from_implied(implied_probs)
		return implied_log + gate * residual_logits

	def get_gate_stats(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor],
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> Dict[str, np.ndarray]:
		self.eval()
		with torch.no_grad():
			h = self.base_model.get_hidden(x, cat_features)
			market_features = self._compute_market_features(implied_probs, raw_margin)
			gate_input = torch.cat([h, market_features], dim=-1)
			gate_logits = self.gate_head(gate_input)
			gate = torch.sigmoid(gate_logits + self.gate_bias)

		return {
			"gate_values": gate.cpu().numpy(),
			"gate_mean": gate.mean(dim=0).cpu().numpy(),
			"gate_std": gate.std(dim=0).cpu().numpy(),
			"gate_min": gate.min(dim=0)[0].cpu().numpy(),
			"gate_max": gate.max(dim=0)[0].cpu().numpy(),
		}


@dataclass
class TrainConfig:
	"""Configuration for model training."""

	input_dim: int
	hidden_layers: List[int]
	dropout: float
	norm: str
	lr: float
	weight_decay: float
	activation: str = "relu"
	optimizer_name: str = "adamw"
	beta1: float = 0.9
	beta2: float = 0.999
	optimizer_eps: float = 1e-8
	epochs: int = 100
	patience: int = 15
	batch_size: int = 128
	cat_config: Optional[CategoricalConfig] = None
	scheduler_name: str = "cosine"
	scheduler_warmup_epochs: int = 0
	scheduler_warmup_start_factor: float = 0.1
	scheduler_min_lr_ratio: float = 0.01
	scheduler_plateau_factor: float = 0.5
	scheduler_plateau_patience: int = 3
	scheduler_plateau_threshold: float = 1e-4
	onecycle_pct_start: float = 0.3
	onecycle_div_factor: float = 25.0
	onecycle_final_div_factor: float = 1000.0
	max_grad_norm: float = 0.0
	gate_hidden_dim: int = 32
	gate_target_budget: float = 0.2
	gate_mean_weight: float = 0.01
	gate_sat_weight: float = 0.001
	lambda_repulsion: float = 0.0
	lambda_corr: float = 0.0


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert implied probabilities to log probabilities."""

	implied_probs = torch.clamp(implied_probs, eps, 1.0 - eps)
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	return torch.log(implied_probs)


def _batch_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Compute batch correlation between two tensors."""

	x = x - x.mean()
	y = y - y.mean()
	vx = x.var(unbiased=False) + eps
	vy = y.var(unbiased=False) + eps
	cov = (x * y).mean()
	return cov / torch.sqrt(vx * vy)


def _multiclass_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Compute conditional correlation for the true class logits."""

	batch_size = pred_logits.shape[0]
	target = target.view(-1).long()
	pred_true_class = pred_logits.gather(1, target.unsqueeze(1)).squeeze(1)
	impl_true_class = implied_logits.gather(1, target.unsqueeze(1)).squeeze(1)

	rho_weighted = pred_logits.new_tensor(0.0)
	weighted_sum = 0.0
	for label in range(3):
		mask = target == label
		n_label = int(mask.sum().item())
		if n_label > 1:
			rho_label = _batch_corr(pred_true_class[mask], impl_true_class[mask], eps=eps)
			weight = n_label / float(batch_size)
			rho_weighted = rho_weighted + weight * rho_label
			weighted_sum += weight

	if weighted_sum == 0:
		return _batch_corr(pred_true_class, impl_true_class, eps=eps)
	return rho_weighted


def gated_loss(
	model: GatedResidualModel,
	x: torch.Tensor,
	cat_features: Optional[torch.Tensor],
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	raw_margin: torch.Tensor,
	gate_mean_weight: float = 0.01,
	gate_sat_weight: float = 0.001,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss for the multiclass gated residual model."""

	pred_logits = model(x, cat_features, implied_probs, raw_margin)
	pred_probs = F.softmax(pred_logits, dim=-1)
	implied_log = _log_softmax_from_implied(implied_probs)
	target = target.view(-1).long()
	loss = F.cross_entropy(pred_logits, target)

	if lambda_repulsion > 0:
		implied_normalized = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
		repulsion = ((pred_probs - implied_normalized) ** 2).mean()
		loss = loss - lambda_repulsion * repulsion

	if lambda_corr > 0:
		rho = _multiclass_conditional_corr(pred_logits, implied_log, target, eps=eps)
		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty

	if gate_mean_weight > 0 or gate_sat_weight > 0:
		h = model.base_model.get_hidden(x, cat_features)
		market_features = model._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = model.gate_head(gate_input)
		gate = torch.sigmoid(gate_logits + model.gate_bias)

		if gate_mean_weight > 0:
			gate_mean_loss = (gate.mean() - model.gate_target_budget).pow(2)
			loss = loss + gate_mean_weight * gate_mean_loss

		if gate_sat_weight > 0:
			sat_loss = (-torch.log(gate * (1 - gate) + eps)).mean()
			loss = loss + gate_sat_weight * sat_loss

	return loss
