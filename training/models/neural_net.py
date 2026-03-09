"""
Neural network architecture and loss functions for match-result prediction.

Architecture: contextual gated residual model.
The network starts from bookmaker implied probabilities and learns when to
deviate from them using a context-dependent gate.
"""

import math
from dataclasses import dataclass, field
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


def _make_activation(name: str) -> nn.Module:
	if name == "relu":
		return nn.ReLU()
	if name == "silu":
		return nn.SiLU()
	if name == "gelu":
		return nn.GELU()
	raise ValueError(f"Unknown activation: {name}")


def _resolve_norm(name: str):
	return {"none": None, "bn": nn.BatchNorm1d, "ln": nn.LayerNorm}.get(name)


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
		NormClass = _resolve_norm(norm)

		for width in hidden_layers:
			layers.append(nn.Linear(prev, width))
			if NormClass is not None:
				layers.append(NormClass(width))
			layers.append(_make_activation(activation))
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


class ResidualBlock(nn.Module):
	"""Residual MLP block with optional projection."""

	def __init__(self, in_dim: int, out_dim: int, dropout: float, norm: str, activation: str):
		super().__init__()
		NormClass = _resolve_norm(norm)
		self.norm1 = None if NormClass is None else NormClass(in_dim)
		self.linear1 = nn.Linear(in_dim, out_dim)
		self.act = _make_activation(activation)
		self.dropout1 = nn.Dropout(dropout)
		self.norm2 = None if NormClass is None else NormClass(out_dim)
		self.linear2 = nn.Linear(out_dim, out_dim)
		self.dropout2 = nn.Dropout(dropout)
		self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.skip(x)
		h = x if self.norm1 is None else self.norm1(x)
		h = self.linear1(h)
		h = self.act(h)
		h = self.dropout1(h)
		h = h if self.norm2 is None else self.norm2(h)
		h = self.linear2(h)
		h = self.dropout2(h)
		return self.act(h + residual)


class ResNetWithHiddenAccess(nn.Module):
	"""Residual MLP backbone for tabular features."""

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
		if not hidden_layers:
			raise ValueError("hidden_layers must be non-empty for resnet backbone")
		self.cat_embedder = None
		total_input_dim = input_dim
		if cat_config is not None:
			self.cat_embedder = CategoricalEmbedder(cat_config)
			total_input_dim = input_dim + self.cat_embedder.output_dim

		blocks = []
		prev = total_input_dim
		for width in hidden_layers:
			blocks.append(ResidualBlock(prev, width, dropout=dropout, norm=norm, activation=activation))
			prev = width

		self.hidden_net = nn.Sequential(*blocks)
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


class CrossLayer(nn.Module):
	"""One explicit cross layer over the original tabular inputs."""

	def __init__(self, input_dim: int):
		super().__init__()
		self.linear = nn.Linear(input_dim, input_dim)

	def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		return x + x0 * self.linear(x)


class CrossResNetWithHiddenAccess(nn.Module):
	"""Parallel cross network plus residual MLP backbone for tabular features."""

	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
		output_dim: int = 3,
		cat_config: Optional[CategoricalConfig] = None,
		cross_layers: int = 2,
	):
		super().__init__()
		if not hidden_layers:
			raise ValueError("hidden_layers must be non-empty for cross_resnet backbone")
		if cross_layers <= 0:
			raise ValueError("cross_layers must be positive for cross_resnet backbone")

		self.cat_embedder = None
		total_input_dim = input_dim
		if cat_config is not None:
			self.cat_embedder = CategoricalEmbedder(cat_config)
			total_input_dim = input_dim + self.cat_embedder.output_dim

		blocks = []
		prev = total_input_dim
		for width in hidden_layers:
			blocks.append(ResidualBlock(prev, width, dropout=dropout, norm=norm, activation=activation))
			prev = width

		self.deep_net = nn.Sequential(*blocks)
		self.cross_net = nn.ModuleList([CrossLayer(total_input_dim) for _ in range(cross_layers)])
		self.hidden_dim = prev + total_input_dim
		self.final_layer = nn.Linear(self.hidden_dim, output_dim)
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.cross_layers = cross_layers
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
		deep_hidden = self.deep_net(x)
		cross_hidden = x
		for layer in self.cross_net:
			cross_hidden = layer(x, cross_hidden)
		return torch.cat([deep_hidden, cross_hidden], dim=-1)


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
		shared_gate: bool = False,
		linear_gate: bool = False,
		market_logit_scale: float = 1.0,
		learn_market_bias: bool = False,
		learn_league_market_bias: bool = False,
		learn_league_residual_bias: bool = False,
		num_leagues: int = 0,
		backbone_type: str = "mlp",
		cross_layers: int = 2,
	):
		super().__init__()
		self.n_classes = n_classes
		self.gate_target_budget = gate_target_budget
		self.shared_gate = shared_gate
		self.linear_gate = linear_gate
		self.market_logit_scale = market_logit_scale
		self.learn_market_bias = learn_market_bias
		self.learn_league_market_bias = learn_league_market_bias
		self.learn_league_residual_bias = learn_league_residual_bias
		self.num_leagues = num_leagues
		self.backbone_type = backbone_type
		if self.learn_league_market_bias and self.num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_market_bias is enabled")
		if self.learn_league_residual_bias and self.num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_residual_bias is enabled")

		backbone_kwargs = {
			"input_dim": input_dim,
			"hidden_layers": hidden_layers,
			"dropout": dropout,
			"norm": norm,
			"activation": activation,
			"output_dim": n_classes,
			"cat_config": cat_config,
		}
		if backbone_type == "mlp":
			BackboneClass = MLPWithHiddenAccess
		elif backbone_type == "resnet":
			BackboneClass = ResNetWithHiddenAccess
		elif backbone_type == "cross_resnet":
			BackboneClass = CrossResNetWithHiddenAccess
			backbone_kwargs["cross_layers"] = cross_layers
		else:
			raise ValueError(f"Unknown backbone_type: {backbone_type}")
		self.base_model = BackboneClass(**backbone_kwargs)

		self.market_feature_dim = market_feature_dim + 3
		gate_input_dim = self.base_model.hidden_dim + self.market_feature_dim
		if linear_gate:
			self.gate_head = nn.Linear(gate_input_dim, 1 if shared_gate else n_classes)
		else:
			self.gate_head = nn.Sequential(
				nn.Linear(gate_input_dim, gate_hidden_dim),
				nn.ReLU(),
				nn.Dropout(dropout * 0.5),
				nn.Linear(gate_hidden_dim, 1 if shared_gate else n_classes),
			)

		init_bias = math.log(gate_target_budget / (1 - gate_target_budget))
		self.gate_bias = nn.Parameter(torch.full((1 if shared_gate else n_classes,), init_bias))
		if learn_market_bias:
			self.market_bias = nn.Parameter(torch.zeros(n_classes))
		else:
			self.register_buffer("market_bias", torch.zeros(n_classes))
		if learn_league_market_bias:
			self.league_market_bias = nn.Embedding(num_leagues, n_classes)
			nn.init.zeros_(self.league_market_bias.weight)
		else:
			self.league_market_bias = None
		if learn_league_residual_bias:
			self.league_residual_bias = nn.Embedding(num_leagues, n_classes)
			nn.init.zeros_(self.league_residual_bias.weight)
		else:
			self.league_residual_bias = None

		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.cat_config = cat_config
		self.gate_hidden_dim = gate_hidden_dim
		self.shared_gate = shared_gate
		self.linear_gate = linear_gate
		self.market_logit_scale = market_logit_scale
		self.learn_market_bias = learn_market_bias
		self.learn_league_market_bias = learn_league_market_bias
		self.learn_league_residual_bias = learn_league_residual_bias
		self.num_leagues = num_leagues
		self.backbone_type = backbone_type
		self.cross_layers = cross_layers

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

	def _compute_implied_logits(
		self,
		implied_probs: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		implied_logits = _log_softmax_from_implied(implied_probs) * self.market_logit_scale + self.market_bias
		if self.league_market_bias is not None:
			if cat_features is None:
				raise ValueError("cat_features required when learn_league_market_bias is enabled")
			implied_logits = implied_logits + self.league_market_bias(cat_features[:, 0].long())
		return implied_logits

	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_probs: Optional[torch.Tensor] = None,
		raw_margin: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		h = self.base_model.get_hidden(x, cat_features)
		residual_logits = self._compute_residual_logits(h, cat_features)

		if implied_probs is None:
			return residual_logits
		if raw_margin is None:
			raise ValueError("raw_margin is required when implied_probs is provided")

		market_features = self._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = self.gate_head(gate_input)
		gate = torch.sigmoid(gate_logits + self.gate_bias)
		if self.shared_gate:
			gate = gate.expand(-1, self.n_classes)
		implied_logits = self._compute_implied_logits(implied_probs, cat_features)
		return implied_logits + gate * residual_logits

	def _compute_residual_logits(
		self,
		hidden: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		residual_logits = self.base_model.final_layer(hidden)
		if self.league_residual_bias is not None:
			if cat_features is None:
				raise ValueError("cat_features required when learn_league_residual_bias is enabled")
			residual_logits = residual_logits + self.league_residual_bias(cat_features[:, 0].long())
		return residual_logits

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
			if self.shared_gate:
				gate = gate.expand(-1, self.n_classes)

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
	lr: float
	weight_decay: float
	model_kwargs: dict = field(default_factory=dict)
	beta1: float = 0.9
	beta2: float = 0.999
	optimizer_eps: float = 1e-8
	epochs: int = 100
	patience: int = 15
	batch_size: int = 128
	cat_config: Optional[CategoricalConfig] = None
	scheduler_min_lr_ratio: float = 0.01
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
		if model.shared_gate:
			gate = gate.expand(-1, model.n_classes)

		if gate_mean_weight > 0:
			gate_mean_loss = (gate.mean() - model.gate_target_budget).pow(2)
			loss = loss + gate_mean_weight * gate_mean_loss

		if gate_sat_weight > 0:
			sat_loss = (-torch.log(gate * (1 - gate) + eps)).mean()
			loss = loss + gate_sat_weight * sat_loss

	return loss
