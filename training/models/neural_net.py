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


class CrossLayer(nn.Module):
	"""One explicit cross layer over the original tabular inputs."""

	def __init__(self, input_dim: int):
		super().__init__()
		self.linear = nn.Linear(input_dim, input_dim)

	def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		return x + x0 * self.linear(x)


class FeatureBackbone(nn.Module):
	"""Canonical tabular backbone with residual and explicit cross features."""

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
			raise ValueError("hidden_layers must be non-empty for feature backbone")
		if cross_layers <= 0:
			raise ValueError("cross_layers must be positive for feature backbone")

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
		learn_market_class_scale: bool = False,
		learn_league_market_bias: bool = False,
		learn_league_market_scale: bool = False,
		league_market_scale_enabled_leagues: Optional[List[int]] = None,
		learn_league_market_class_scale: bool = False,
		league_market_class_scale_enabled_leagues: Optional[List[int]] = None,
		learn_league_market_logit_mixer: bool = False,
		learn_league_gate_bias: bool = False,
		learn_league_residual_bias: bool = False,
		num_leagues: int = 0,
		cross_layers: int = 2,
	):
		super().__init__()
		if learn_league_market_bias and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_market_bias is enabled")
		if learn_league_market_scale and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_market_scale is enabled")
		if learn_league_market_class_scale and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_market_class_scale is enabled")
		if learn_league_market_logit_mixer and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_market_logit_mixer is enabled")
		if learn_league_gate_bias and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_gate_bias is enabled")
		if learn_league_residual_bias and num_leagues <= 0:
			raise ValueError("num_leagues must be positive when learn_league_residual_bias is enabled")

		self.backbone = FeatureBackbone(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			norm=norm,
			activation=activation,
			output_dim=n_classes,
			cat_config=cat_config,
			cross_layers=cross_layers,
		)

		if market_feature_dim not in {3, 4, 5}:
			raise ValueError(f"Unsupported market_feature_dim: {market_feature_dim}")
		self.market_feature_dim = market_feature_dim + 3
		gate_input_dim = self.backbone.hidden_dim + self.market_feature_dim
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
		if learn_market_class_scale:
			self.market_class_scale = nn.Parameter(torch.zeros(n_classes))
		else:
			self.market_class_scale = None
		if learn_league_market_bias:
			self.league_market_bias = nn.Embedding(num_leagues, n_classes)
			nn.init.zeros_(self.league_market_bias.weight)
		else:
			self.league_market_bias = None
		if learn_league_market_scale:
			self.league_market_scale = nn.Embedding(num_leagues, 1)
			nn.init.zeros_(self.league_market_scale.weight)
		else:
			self.league_market_scale = None
		if league_market_scale_enabled_leagues is not None:
			enabled_mask = torch.zeros(num_leagues, 1, dtype=torch.float32)
			for league_idx in league_market_scale_enabled_leagues:
				if league_idx < 0 or league_idx >= num_leagues:
					raise ValueError(
						f"league_market_scale_enabled_leagues index {league_idx} outside [0, {num_leagues - 1}]"
					)
				enabled_mask[int(league_idx), 0] = 1.0
			self.register_buffer("league_market_scale_enabled_mask", enabled_mask)
		else:
			self.league_market_scale_enabled_mask = None
		if learn_league_market_class_scale:
			self.league_market_class_scale = nn.Embedding(num_leagues, n_classes)
			nn.init.zeros_(self.league_market_class_scale.weight)
		else:
			self.league_market_class_scale = None
		if league_market_class_scale_enabled_leagues is not None:
			class_enabled_mask = torch.zeros(num_leagues, 1, dtype=torch.float32)
			for league_idx in league_market_class_scale_enabled_leagues:
				if league_idx < 0 or league_idx >= num_leagues:
					raise ValueError(
						f"league_market_class_scale_enabled_leagues index {league_idx} outside [0, {num_leagues - 1}]"
					)
				class_enabled_mask[int(league_idx), 0] = 1.0
			self.register_buffer("league_market_class_scale_enabled_mask", class_enabled_mask)
		else:
			self.league_market_class_scale_enabled_mask = None
		if learn_league_market_logit_mixer:
			self.league_market_logit_mixer = nn.Embedding(num_leagues, n_classes * n_classes)
			nn.init.zeros_(self.league_market_logit_mixer.weight)
		else:
			self.league_market_logit_mixer = None
		if learn_league_gate_bias:
			self.league_gate_bias = nn.Embedding(num_leagues, 1 if shared_gate else n_classes)
			nn.init.zeros_(self.league_gate_bias.weight)
		else:
			self.league_gate_bias = None
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
		self.n_classes = n_classes
		self.gate_hidden_dim = gate_hidden_dim
		self.gate_target_budget = gate_target_budget
		self.shared_gate = shared_gate
		self.linear_gate = linear_gate
		self.market_logit_scale = market_logit_scale
		self.learn_market_bias = learn_market_bias
		self.learn_market_class_scale = learn_market_class_scale
		self.learn_league_market_bias = learn_league_market_bias
		self.learn_league_market_scale = learn_league_market_scale
		self.league_market_scale_enabled_leagues = league_market_scale_enabled_leagues
		self.learn_league_market_class_scale = learn_league_market_class_scale
		self.league_market_class_scale_enabled_leagues = league_market_class_scale_enabled_leagues
		self.learn_league_market_logit_mixer = learn_league_market_logit_mixer
		self.learn_league_gate_bias = learn_league_gate_bias
		self.learn_league_residual_bias = learn_league_residual_bias
		self.market_feature_stats = market_feature_dim
		self.num_leagues = num_leagues
		self.cross_layers = cross_layers

	def _compute_gate_logits(
		self,
		hidden: torch.Tensor,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		market_features = self._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([hidden, market_features], dim=-1)
		gate_logits = self.gate_head(gate_input)
		if self.league_gate_bias is not None:
			if cat_features is None:
				raise ValueError("cat_features required when learn_league_gate_bias is enabled")
			gate_logits = gate_logits + self.league_gate_bias(cat_features[:, 0].long())
		return gate_logits

	def _compute_market_features(
		self,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		eps = 1e-6
		entropy = -torch.sum(implied_probs * torch.log(implied_probs + eps), dim=-1, keepdim=True)
		entropy = entropy / math.log(self.n_classes)
		max_prob = implied_probs.max(dim=-1, keepdim=True)[0]
		min_prob = implied_probs.min(dim=-1, keepdim=True)[0]
		sorted_probs = torch.sort(implied_probs, dim=-1, descending=True)[0]
		top2_gap = sorted_probs[:, :1] - sorted_probs[:, 1:2]
		if raw_margin.dim() == 1:
			raw_margin = raw_margin.unsqueeze(-1)
		feature_parts = [implied_probs, entropy, max_prob, raw_margin]
		if self.market_feature_stats >= 4:
			feature_parts.append(min_prob)
		if self.market_feature_stats >= 5:
			feature_parts.append(top2_gap)
		return torch.cat(feature_parts, dim=-1)

	def _compute_implied_logits(
		self,
		implied_probs: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		log_implied = _log_softmax_from_implied(implied_probs)
		league_idx = None
		if cat_features is not None:
			league_idx = cat_features[:, 0].long()
		if self.league_market_scale is not None:
			if league_idx is None:
				raise ValueError("cat_features required when learn_league_market_scale is enabled")
			league_scale = torch.exp(self.league_market_scale(league_idx))
			if self.league_market_scale_enabled_mask is not None:
				enabled = self.league_market_scale_enabled_mask[league_idx]
				league_scale = enabled * league_scale + (1.0 - enabled)
			log_implied = log_implied * (self.market_logit_scale * league_scale)
		else:
			log_implied = log_implied * self.market_logit_scale
		if self.market_class_scale is not None:
			log_implied = log_implied * torch.exp(self.market_class_scale)
		if self.league_market_class_scale is not None:
			if league_idx is None:
				raise ValueError("cat_features required when learn_league_market_class_scale is enabled")
			class_scale = torch.exp(self.league_market_class_scale(league_idx))
			if self.league_market_class_scale_enabled_mask is not None:
				enabled = self.league_market_class_scale_enabled_mask[league_idx]
				class_scale = enabled * class_scale + (1.0 - enabled)
			log_implied = log_implied * class_scale
		if self.league_market_logit_mixer is not None:
			if league_idx is None:
				raise ValueError("cat_features required when learn_league_market_logit_mixer is enabled")
			mix = self.league_market_logit_mixer(league_idx).view(-1, self.n_classes, self.n_classes)
			mix = mix - torch.diag_embed(torch.diagonal(mix, dim1=-2, dim2=-1))
			log_implied = log_implied + torch.bmm(log_implied.unsqueeze(1), mix).squeeze(1)
		implied_logits = log_implied + self.market_bias
		if self.league_market_bias is not None:
			if league_idx is None:
				raise ValueError("cat_features required when learn_league_market_bias is enabled")
			implied_logits = implied_logits + self.league_market_bias(league_idx)
		return implied_logits

	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_probs: Optional[torch.Tensor] = None,
		raw_margin: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		h = self.backbone.get_hidden(x, cat_features)
		residual_logits = self._compute_residual_logits(h, cat_features)

		if implied_probs is None:
			return residual_logits
		if raw_margin is None:
			raise ValueError("raw_margin is required when implied_probs is provided")

		gate_logits = self._compute_gate_logits(h, implied_probs, raw_margin, cat_features)
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
		residual_logits = self.backbone.final_layer(hidden)
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
			h = self.backbone.get_hidden(x, cat_features)
			gate_logits = self._compute_gate_logits(h, implied_probs, raw_margin, cat_features)
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
	lambda_logit_delta: float = 0.0
	market_target_mix: float = 0.0
	market_target_surprise_scale: float = 0.0
	market_target_draw_weight: float = 1.0
	market_target_away_weight: float = 1.0
	market_target_entropy_scale: float = 0.0
	market_target_entropy_mode: str = "linear"
	entropy_curriculum_mode: str = "none"
	entropy_curriculum_strength: float = 0.0
	gce_mix_weight: float = 0.0
	gce_q: float = 0.7


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert implied probabilities to log probabilities."""

	implied_probs = torch.clamp(implied_probs, eps, 1.0 - eps)
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	return torch.log(implied_probs)


def _normalized_market_entropy(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Normalized market entropy in [0, 1]."""

	implied_probs = torch.clamp(implied_probs, eps, 1.0)
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	entropy = -torch.sum(implied_probs * torch.log(implied_probs), dim=-1)
	return entropy / math.log(implied_probs.shape[-1])


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


def _apply_true_class_surprise_scaling(
	base_mix: torch.Tensor,
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	scale: float,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Increase mix weights for outcomes the market assigned lower true-class probability."""

	if abs(scale) <= eps:
		return base_mix
	true_class_prob = implied_probs.gather(1, target.view(-1, 1).long()).clamp(0.0, 1.0)
	surprise = 1.0 - true_class_prob
	return base_mix * (1.0 + scale * surprise)


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
	lambda_logit_delta: float = 0.0,
	market_target_mix: float = 0.0,
	market_target_surprise_scale: float = 0.0,
	market_target_draw_weight: float = 1.0,
	market_target_away_weight: float = 1.0,
	market_target_entropy_scale: float = 0.0,
	market_target_entropy_mode: str = "linear",
	sample_weights: Optional[torch.Tensor] = None,
	gce_mix_weight: float = 0.0,
	gce_q: float = 0.7,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss for the multiclass gated residual model."""

	pred_logits = model(x, cat_features, implied_probs, raw_margin)
	pred_probs = F.softmax(pred_logits, dim=-1)
	log_probs = F.log_softmax(pred_logits, dim=-1)
	implied_log = _log_softmax_from_implied(implied_probs)
	target = target.view(-1).long()
	if market_target_mix > 0:
		soft_target = F.one_hot(target, num_classes=model.n_classes).float()
		mix = pred_logits.new_full((soft_target.shape[0], 1), market_target_mix)
		if abs(market_target_draw_weight - 1.0) > eps or abs(market_target_away_weight - 1.0) > eps:
			class_weight = torch.ones_like(mix)
			draw_mask = (target == 1).float().unsqueeze(-1)
			away_mask = (target == 2).float().unsqueeze(-1)
			class_weight = class_weight + draw_mask * (market_target_draw_weight - 1.0)
			class_weight = class_weight + away_mask * (market_target_away_weight - 1.0)
			mix = mix * class_weight
		if abs(market_target_entropy_scale) > eps:
			normalized_entropy = _normalized_market_entropy(implied_probs, eps=eps).unsqueeze(-1)
			if market_target_entropy_mode == "linear":
				centered_entropy = 2.0 * (normalized_entropy - 0.5)
			elif market_target_entropy_mode == "edge":
				edge_signal = 2.0 * torch.abs(normalized_entropy - 0.5)
				centered_entropy = 2.0 * (edge_signal - 0.5)
			else:
				raise ValueError(f"Unsupported market_target_entropy_mode: {market_target_entropy_mode}")
			mix = (mix * (1.0 + market_target_entropy_scale * centered_entropy)).clamp(0.0, 1.0)
		mix = _apply_true_class_surprise_scaling(
			mix,
			implied_probs,
			target,
			market_target_surprise_scale,
			eps=eps,
		).clamp(0.0, 1.0)
		soft_target = (1.0 - mix) * soft_target + mix * implied_probs
		base_loss = -(soft_target * log_probs).sum(dim=-1)
	else:
		base_loss = F.nll_loss(log_probs, target, reduction="none")

	if gce_mix_weight > 0:
		true_class_probs = pred_probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(eps)
		if abs(gce_q) <= eps:
			gce_loss = -torch.log(true_class_probs)
		else:
			gce_loss = (1.0 - true_class_probs.pow(gce_q)) / gce_q
		base_loss = (1.0 - gce_mix_weight) * base_loss + gce_mix_weight * gce_loss

	if sample_weights is not None:
		sample_weights = sample_weights.view(-1).clamp_min(eps)
		sample_weights = sample_weights / sample_weights.mean().clamp_min(eps)
		loss = (base_loss * sample_weights).mean()
	else:
		loss = base_loss.mean()

	if lambda_repulsion > 0:
		implied_normalized = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
		repulsion = ((pred_probs - implied_normalized) ** 2).mean()
		loss = loss - lambda_repulsion * repulsion

	if lambda_corr > 0:
		rho = _multiclass_conditional_corr(pred_logits, implied_log, target, eps=eps)
		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty

	if lambda_logit_delta > 0:
		implied_anchor = model._compute_implied_logits(implied_probs, cat_features)
		logit_delta = pred_logits - implied_anchor
		loss = loss + lambda_logit_delta * logit_delta.pow(2).mean()

	if gate_mean_weight > 0 or gate_sat_weight > 0:
		h = model.backbone.get_hidden(x, cat_features)
		gate_logits = model._compute_gate_logits(h, implied_probs, raw_margin, cat_features)
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
