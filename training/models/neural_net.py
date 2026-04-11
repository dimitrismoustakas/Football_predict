"""
Neural network architecture and loss functions for match-result prediction.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
	"""Canonical residual MLP block used by the live result model."""

	def __init__(self, in_dim: int, out_dim: int, dropout: float):
		super().__init__()
		self.linear1 = nn.Linear(in_dim, out_dim)
		self.act = nn.GELU()
		self.dropout1 = nn.Dropout(dropout)
		self.linear2 = nn.Linear(out_dim, out_dim)
		self.dropout2 = nn.Dropout(dropout)
		self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		residual = self.skip(x)
		h = self.act(self.linear1(x))
		h = self.dropout1(h)
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
		output_dim: int = 3,
		cross_layers: int = 2,
	):
		super().__init__()
		if not hidden_layers:
			raise ValueError("hidden_layers must be non-empty for feature backbone")
		if cross_layers <= 0:
			raise ValueError("cross_layers must be positive for feature backbone")

		blocks = []
		prev = input_dim
		for width in hidden_layers:
			blocks.append(
				ResidualBlock(
					prev,
					width,
					dropout=dropout,
				)
			)
			prev = width

		self.deep_net = nn.Sequential(*blocks)
		self.cross_net = nn.ModuleList([CrossLayer(input_dim) for _ in range(cross_layers)])
		self.hidden_dim = prev + input_dim
		self.final_layer = nn.Linear(self.hidden_dim, output_dim)
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.cross_layers = cross_layers
		self.dropout = dropout
		self.norm = "none"
		self.activation = "gelu"
		self.output_dim = output_dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		h = self.get_hidden(x)
		return self.final_layer(h)

	def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
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
		dropout: float = 0.3,
		gate_target_budget: float = 0.2,
		market_logit_scale: float = 1.0,
		league_market_bias_enabled_leagues: Optional[List[int]] = None,
		league_market_scale_enabled_leagues: Optional[List[int]] = None,
		league_market_class_scale_enabled_leagues: Optional[List[int]] = None,
		league_market_logit_mixer_enabled_leagues: Optional[List[int]] = None,
		num_leagues: int = 0,
		cross_layers: int = 2,
		high_draw_positive_residual_scale: float = 1.0,
		high_draw_positive_threshold: float = 0.26,
		low_draw_negative_residual_scale: float = 1.0,
		low_draw_negative_threshold: float = 0.22,
	):
		super().__init__()
		if num_leagues <= 0:
			raise ValueError("num_leagues must be positive for the canonical result model")

		self.backbone = FeatureBackbone(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			output_dim=n_classes,
			cross_layers=cross_layers,
		)

		gate_input_dim = self.backbone.hidden_dim + 7
		self.gate_head = nn.Linear(gate_input_dim, 1)

		init_bias = math.log(gate_target_budget / (1 - gate_target_budget))
		self.gate_bias = nn.Parameter(torch.full((1,), init_bias))
		self.register_buffer("market_bias", torch.zeros(n_classes))
		self.market_class_scale = None
		self.league_market_bias = nn.Embedding(num_leagues, n_classes)
		nn.init.zeros_(self.league_market_bias.weight)
		self._register_enabled_mask(
			"league_market_bias_enabled_mask",
			num_leagues,
			league_market_bias_enabled_leagues,
			"league_market_bias_enabled_leagues",
		)
		self.league_market_scale = nn.Embedding(num_leagues, 1)
		nn.init.zeros_(self.league_market_scale.weight)
		self._register_enabled_mask(
			"league_market_scale_enabled_mask",
			num_leagues,
			league_market_scale_enabled_leagues,
			"league_market_scale_enabled_leagues",
		)
		self.league_market_class_scale = nn.Embedding(num_leagues, n_classes)
		nn.init.zeros_(self.league_market_class_scale.weight)
		self._register_enabled_mask(
			"league_market_class_scale_enabled_mask",
			num_leagues,
			league_market_class_scale_enabled_leagues,
			"league_market_class_scale_enabled_leagues",
		)
		self.league_market_logit_mixer = nn.Embedding(num_leagues, n_classes * n_classes)
		nn.init.zeros_(self.league_market_logit_mixer.weight)
		self._register_enabled_mask(
			"league_market_logit_mixer_enabled_mask",
			num_leagues,
			league_market_logit_mixer_enabled_leagues,
			"league_market_logit_mixer_enabled_leagues",
		)

		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = "none"
		self.activation = "gelu"
		self.n_classes = n_classes
		self.gate_target_budget = gate_target_budget
		self.shared_gate = True
		self.linear_gate = True
		self.market_logit_scale = market_logit_scale
		self.learn_league_market_bias = True
		self.league_market_bias_enabled_leagues = league_market_bias_enabled_leagues
		self.learn_league_market_scale = True
		self.league_market_scale_enabled_leagues = league_market_scale_enabled_leagues
		self.learn_league_market_class_scale = True
		self.league_market_class_scale_enabled_leagues = league_market_class_scale_enabled_leagues
		self.learn_league_market_logit_mixer = True
		self.league_market_logit_mixer_enabled_leagues = league_market_logit_mixer_enabled_leagues
		self.num_leagues = num_leagues
		self.cross_layers = cross_layers
		self.high_draw_positive_residual_scale = float(high_draw_positive_residual_scale)
		self.high_draw_positive_threshold = float(high_draw_positive_threshold)
		self.low_draw_negative_residual_scale = float(low_draw_negative_residual_scale)
		self.low_draw_negative_threshold = float(low_draw_negative_threshold)

	@staticmethod
	def _build_enabled_mask(
		num_leagues: int,
		enabled_leagues: Optional[List[int]],
		name: str,
	) -> Optional[torch.Tensor]:
		if enabled_leagues is None:
			return None
		enabled_mask = torch.zeros(num_leagues, 1, dtype=torch.float32)
		for league_idx in enabled_leagues:
			if league_idx < 0 or league_idx >= num_leagues:
				raise ValueError(f"{name} index {league_idx} outside [0, {num_leagues - 1}]")
			enabled_mask[int(league_idx), 0] = 1.0
		return enabled_mask

	def _register_enabled_mask(
		self,
		buffer_name: str,
		num_leagues: int,
		enabled_leagues: Optional[List[int]],
		config_name: str,
	):
		mask = self._build_enabled_mask(num_leagues, enabled_leagues, config_name)
		self.register_buffer(buffer_name, mask)

	def _compute_gate_logits(
		self,
		hidden: torch.Tensor,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		market_features = self._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([hidden, market_features], dim=-1)
		return self.gate_head(gate_input)

	def _compute_gate(
		self,
		hidden: torch.Tensor,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		gate_logits = self._compute_gate_logits(hidden, implied_probs, raw_margin)
		return torch.sigmoid(gate_logits + self.gate_bias).expand(-1, self.n_classes)

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
		if raw_margin.dim() == 1:
			raw_margin = raw_margin.unsqueeze(-1)
		return torch.cat([implied_probs, entropy, max_prob, raw_margin, min_prob], dim=-1)

	def _compute_implied_logits(
		self,
		implied_probs: torch.Tensor,
		cat_features: torch.Tensor,
	) -> torch.Tensor:
		if cat_features is None:
			raise ValueError("cat_features are required for the canonical result model")
		log_implied = _log_softmax_from_implied(implied_probs)
		league_idx = cat_features[:, 0].long()
		league_scale = torch.exp(self.league_market_scale(league_idx))
		if self.league_market_scale_enabled_mask is not None:
			enabled = self.league_market_scale_enabled_mask[league_idx]
			league_scale = enabled * league_scale + (1.0 - enabled)
		scale = self.market_logit_scale * league_scale
		log_implied = log_implied * scale
		class_scale = torch.exp(self.league_market_class_scale(league_idx))
		if self.league_market_class_scale_enabled_mask is not None:
			enabled = self.league_market_class_scale_enabled_mask[league_idx]
			class_scale = enabled * class_scale + (1.0 - enabled)
		log_implied = log_implied * class_scale
		mix = self.league_market_logit_mixer(league_idx).view(-1, self.n_classes, self.n_classes)
		mix = mix - torch.diag_embed(torch.diagonal(mix, dim1=-2, dim2=-1))
		if self.league_market_logit_mixer_enabled_mask is not None:
			enabled = self.league_market_logit_mixer_enabled_mask[league_idx].unsqueeze(-1)
			mix = mix * enabled
		log_implied = log_implied + torch.bmm(log_implied.unsqueeze(1), mix).squeeze(1)
		implied_logits = log_implied + self.market_bias
		league_bias = self.league_market_bias(league_idx)
		if self.league_market_bias_enabled_mask is not None:
			league_bias = league_bias * self.league_market_bias_enabled_mask[league_idx]
		implied_logits = implied_logits + league_bias
		return implied_logits

	def forward(
		self,
		x: torch.Tensor,
		cat_features: torch.Tensor,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		h = self.backbone.get_hidden(x)
		implied_logits = self._compute_implied_logits(implied_probs, cat_features)
		anchor_draw_prob = torch.softmax(implied_logits, dim=-1)[:, 1]
		residual_logits = self._compute_residual_logits(h, anchor_draw_prob=anchor_draw_prob)
		gate = self._compute_gate(h, implied_probs, raw_margin)
		return implied_logits + gate * residual_logits

	def _compute_residual_logits(
		self,
		hidden: torch.Tensor,
		anchor_draw_prob: torch.Tensor | None = None,
	) -> torch.Tensor:
		residual_logits = self.backbone.final_layer(hidden)
		if (
			self.high_draw_positive_residual_scale == 1.0
			and self.low_draw_negative_residual_scale == 1.0
		):
			return residual_logits
		if anchor_draw_prob is None:
			raise ValueError("anchor_draw_prob is required when draw residual caps are active")
		draw_logit = residual_logits[:, 1]
		draw_scale = torch.ones_like(draw_logit)
		if self.high_draw_positive_residual_scale != 1.0:
			positive_high_draw_mask = (anchor_draw_prob.view(-1) >= self.high_draw_positive_threshold) & (draw_logit > 0)
			draw_scale = torch.where(
				positive_high_draw_mask,
				draw_scale.new_full(draw_scale.shape, self.high_draw_positive_residual_scale),
				draw_scale,
			)
		if self.low_draw_negative_residual_scale != 1.0:
			negative_low_draw_mask = (anchor_draw_prob.view(-1) <= self.low_draw_negative_threshold) & (draw_logit < 0)
			draw_scale = torch.where(
				negative_low_draw_mask,
				draw_scale.new_full(draw_scale.shape, self.low_draw_negative_residual_scale),
				draw_scale,
			)
		scaled_draw = draw_logit.unsqueeze(-1) * draw_scale.unsqueeze(-1)
		return torch.cat([residual_logits[:, :1], scaled_draw, residual_logits[:, 2:]], dim=-1)

	def get_gate_stats(
		self,
		x: torch.Tensor,
		cat_features: torch.Tensor,
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> Dict[str, np.ndarray]:
		self.eval()
		with torch.no_grad():
			h = self.backbone.get_hidden(x)
			gate = self._compute_gate(h, implied_probs, raw_margin)

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
	scheduler_min_lr_ratio: float = 0.01
	gate_mean_weight: float = 0.01
	gate_sat_weight: float = 0.001
	lambda_repulsion: float = 0.0
	lambda_corr: float = 0.0
	lambda_logit_delta: float = 0.0
	logit_delta_home_weight: float = 1.0
	logit_delta_draw_weight: float = 1.0
	logit_delta_away_weight: float = 1.0
	market_target_mix: float = 0.0
	market_target_surprise_scale: float = 0.0
	market_target_surprise_power: float = 1.0
	market_target_surprise_floor: float = 0.0
	market_target_draw_surprise_scale: Optional[float] = None
	market_target_away_surprise_scale: Optional[float] = None
	market_target_draw_surprise_floor: Optional[float] = None
	market_target_away_surprise_floor: Optional[float] = None
	market_target_surprise_mode: str = "power"
	market_target_surprise_center: float = 0.5
	market_target_surprise_width: float = 0.3
	market_target_surprise_slope: float = 12.0
	market_target_draw_weight: float = 1.0
	market_target_away_weight: float = 1.0
	market_target_entropy_scale: float = 0.0
	market_target_entropy_mode: str = "linear"
	entropy_curriculum_mode: str = "none"
	entropy_curriculum_strength: float = 0.0
	confidence_penalty_weight: float = 0.0
	brier_aux_weight: float = 0.0
	symmetric_ce_weight: float = 0.0
	symmetric_ce_label_floor: float = 1e-4
	gce_mix_weight: float = 0.0
	gce_q: float = 0.7
	bi_tempered_mix_weight: float = 0.0
	bi_tempered_t1: float = 1.0
	bi_tempered_t2: float = 1.0
	bi_tempered_num_iters: int = 5
	anchor_regret_weight: float = 0.0
	anchor_regret_margin: float = 0.0
	anchor_regret_power: float = 1.0


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


def _reverse_cross_entropy(
	pred_probs: torch.Tensor,
	target_distribution: torch.Tensor,
	label_floor: float = 1e-4,
) -> torch.Tensor:
	"""Reverse cross entropy with clipped targets for one-hot or soft labels."""

	if label_floor <= 0 or label_floor >= 1:
		raise ValueError("symmetric_ce_label_floor must be in (0, 1)")
	clipped_target = target_distribution.clamp_min(label_floor)
	return -(pred_probs * torch.log(clipped_target)).sum(dim=-1)


def _log_t(x: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
	"""Tempered logarithm from the bi-tempered logistic loss."""

	x = x.clamp_min(eps)
	if abs(t - 1.0) <= eps:
		return torch.log(x)
	return (x.pow(1.0 - t) - 1.0) / (1.0 - t)


def _exp_t(x: torch.Tensor, t: float, eps: float = 1e-6) -> torch.Tensor:
	"""Tempered exponential from the bi-tempered logistic loss."""

	if abs(t - 1.0) <= eps:
		return torch.exp(x)
	base = (1.0 + (1.0 - t) * x).clamp_min(0.0)
	return base.pow(1.0 / (1.0 - t))


def _tempered_normalization_fixed_point(
	activations: torch.Tensor,
	t: float,
	num_iters: int = 5,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Fixed-point normalization for heavy-tailed tempered softmax (t >= 1)."""

	if t < 1.0 - eps:
		raise ValueError("bi_tempered_t2 must be >= 1.0 for the fixed-point normalizer")
	mu = activations.max(dim=-1, keepdim=True).values
	normalized_step0 = activations - mu
	normalized = normalized_step0
	for _ in range(max(1, int(num_iters))):
		partition = _exp_t(normalized, t, eps=eps).sum(dim=-1, keepdim=True).clamp_min(eps)
		normalized = normalized_step0 * partition.pow(1.0 - t)
	partition = _exp_t(normalized, t, eps=eps).sum(dim=-1, keepdim=True).clamp_min(eps)
	return -_log_t(partition.reciprocal(), t, eps=eps) + mu


def _tempered_softmax(
	activations: torch.Tensor,
	t: float,
	num_iters: int = 5,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Tempered softmax; matches softmax exactly when t == 1."""

	if abs(t - 1.0) <= eps:
		return F.softmax(activations, dim=-1)
	normalization = _tempered_normalization_fixed_point(activations, t, num_iters=num_iters, eps=eps)
	probs = _exp_t(activations - normalization, t, eps=eps)
	return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def _validate_bi_tempered_temperatures(t1: float, t2: float, eps: float = 1e-6):
	if t1 >= 2.0 - eps:
		raise ValueError("bi_tempered_t1 must be < 2.0")
	if t1 <= 0.0:
		raise ValueError("bi_tempered_t1 must be positive")
	if t2 <= 0.0:
		raise ValueError("bi_tempered_t2 must be positive")


def _bi_tempered_logistic_loss_autograd(
	activations: torch.Tensor,
	target_distribution: torch.Tensor,
	t1: float,
	t2: float,
	num_iters: int = 5,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Reference bi-tempered logistic loss with the full fixed-point loop in autograd."""

	_validate_bi_tempered_temperatures(t1, t2, eps=eps)
	probabilities = _tempered_softmax(activations, t2, num_iters=num_iters, eps=eps).clamp_min(eps)
	target_distribution = target_distribution.clamp_min(0.0)
	tempered_kl = (
		(_log_t(target_distribution + eps, t1, eps=eps) - _log_t(probabilities, t1, eps=eps))
		* target_distribution
	)
	bias_correction = (
		target_distribution.pow(2.0 - t1) - probabilities.pow(2.0 - t1)
	) / (2.0 - t1)
	return (tempered_kl - bias_correction).sum(dim=-1)


class _BiTemperedLogisticLoss(torch.autograd.Function):
	@staticmethod
	def forward(
		ctx,
		activations: torch.Tensor,
		target_distribution: torch.Tensor,
		t1: float,
		t2: float,
		num_iters: int,
		eps: float,
	) -> torch.Tensor:
		t1 = float(t1)
		t2 = float(t2)
		num_iters = int(num_iters)
		eps = float(eps)
		_validate_bi_tempered_temperatures(t1, t2, eps=eps)
		probabilities = _tempered_softmax(activations, t2, num_iters=num_iters, eps=eps).clamp_min(eps)
		target_distribution = target_distribution.clamp_min(0.0)
		tempered_kl = (
			(_log_t(target_distribution + eps, t1, eps=eps) - _log_t(probabilities, t1, eps=eps))
			* target_distribution
		)
		bias_correction = (
			target_distribution.pow(2.0 - t1) - probabilities.pow(2.0 - t1)
		) / (2.0 - t1)
		ctx.save_for_backward(probabilities, target_distribution)
		ctx.t1 = t1
		ctx.t2 = t2
		ctx.eps = eps
		return (tempered_kl - bias_correction).sum(dim=-1)

	@staticmethod
	def backward(ctx, grad_output: torch.Tensor):
		probabilities, target_distribution = ctx.saved_tensors
		delta_probs = probabilities - target_distribution
		forget_factor = probabilities.pow(ctx.t2 - ctx.t1)
		delta_probs_times_forget_factor = delta_probs * forget_factor
		delta_forget_sum = delta_probs_times_forget_factor.sum(dim=-1, keepdim=True)
		escorts = probabilities.pow(ctx.t2)
		escorts = escorts / escorts.sum(dim=-1, keepdim=True).clamp_min(ctx.eps)
		derivative = delta_probs_times_forget_factor - escorts * delta_forget_sum
		return grad_output.unsqueeze(-1) * derivative, None, None, None, None, None


def _bi_tempered_logistic_loss(
	activations: torch.Tensor,
	target_distribution: torch.Tensor,
	t1: float,
	t2: float,
	num_iters: int = 5,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Bi-tempered logistic loss for one-hot or soft labels with analytical backward."""

	return _BiTemperedLogisticLoss.apply(
		activations,
		target_distribution,
		float(t1),
		float(t2),
		int(num_iters),
		float(eps),
	)


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


def _normalized_true_class_surprise(
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	floor: float | torch.Tensor = 0.0,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Return normalized true-class surprise after applying an optional floor."""

	true_class_prob = implied_probs.gather(1, target.view(-1, 1).long()).clamp(0.0, 1.0)
	floor = torch.as_tensor(floor, dtype=true_class_prob.dtype, device=true_class_prob.device)
	if torch.any((floor < 0) | (floor >= 1)):
		raise ValueError("market_target_surprise_floor must be in [0, 1)")
	surprise = 1.0 - true_class_prob
	if torch.any(floor > eps):
		surprise = (surprise - floor).clamp_min(0.0) / (1.0 - floor).clamp_min(eps)
	return surprise.clamp(0.0, 1.0)


def _resolve_true_class_surprise_parameter(
	target: torch.Tensor,
	base_value: float,
	draw_value: Optional[float],
	away_value: Optional[float],
	like: torch.Tensor,
) -> torch.Tensor:
	"""Expand scalar surprise settings into sample-wise values by target class."""

	value = like.new_full((target.shape[0], 1), float(base_value))
	if draw_value is not None:
		draw_tensor = like.new_full((target.shape[0], 1), float(draw_value))
		value = torch.where(target.view(-1, 1) == 1, draw_tensor, value)
	if away_value is not None:
		away_tensor = like.new_full((target.shape[0], 1), float(away_value))
		value = torch.where(target.view(-1, 1) == 2, away_tensor, value)
	return value


def _logistic_surprise_response(
	surprise: torch.Tensor,
	center: float,
	slope: float,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Monotone saturating response on normalized surprise."""

	if center < 0 or center > 1:
		raise ValueError("market_target_surprise_center must be in [0, 1]")
	if slope <= 0:
		raise ValueError("market_target_surprise_slope must be positive")
	lower = torch.sigmoid(surprise.new_tensor(-slope * center))
	upper = torch.sigmoid(surprise.new_tensor(slope * (1.0 - center)))
	raw = torch.sigmoid(slope * (surprise - center))
	return ((raw - lower) / (upper - lower).clamp_min(eps)).clamp(0.0, 1.0)


def _band_surprise_response(
	surprise: torch.Tensor,
	center: float,
	width: float,
	slope: float,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Windowed response that concentrates extra smoothing inside a surprise band."""

	if center < 0 or center > 1:
		raise ValueError("market_target_surprise_center must be in [0, 1]")
	if width <= 0:
		raise ValueError("market_target_surprise_width must be positive")
	if slope <= 0:
		raise ValueError("market_target_surprise_slope must be positive")
	half_width = width / 2.0
	start = center - half_width
	end = center + half_width
	rise = torch.sigmoid(slope * (surprise - start))
	fall = torch.sigmoid(slope * (end - surprise))
	peak = torch.sigmoid(surprise.new_tensor(slope * half_width)).pow(2)
	return ((rise * fall) / peak.clamp_min(eps)).clamp(0.0, 1.0)


def _apply_true_class_surprise_scaling(
	base_mix: torch.Tensor,
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	scale: float,
	power: float = 1.0,
	floor: float = 0.0,
	draw_scale: Optional[float] = None,
	away_scale: Optional[float] = None,
	draw_floor: Optional[float] = None,
	away_floor: Optional[float] = None,
	mode: str = "power",
	center: float = 0.5,
	width: float = 0.3,
	slope: float = 12.0,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Increase mix weights for outcomes the market assigned lower true-class probability."""

	scale = _resolve_true_class_surprise_parameter(
		target,
		base_value=scale,
		draw_value=draw_scale,
		away_value=away_scale,
		like=base_mix,
	)
	if torch.all(scale.abs() <= eps):
		return base_mix
	floor = _resolve_true_class_surprise_parameter(
		target,
		base_value=floor,
		draw_value=draw_floor,
		away_value=away_floor,
		like=base_mix,
	)
	surprise = _normalized_true_class_surprise(implied_probs, target, floor=floor, eps=eps)
	if mode == "power":
		if power <= 0:
			raise ValueError("market_target_surprise_power must be positive")
		response = surprise if abs(power - 1.0) <= eps else surprise.pow(power)
	elif mode == "logistic":
		response = _logistic_surprise_response(surprise, center=center, slope=slope, eps=eps)
	elif mode == "band":
		response = _band_surprise_response(surprise, center=center, width=width, slope=slope, eps=eps)
	else:
		raise ValueError(f"Unsupported market_target_surprise_mode: {mode}")
	return base_mix * (1.0 + scale * response)


def _anchor_regret_penalty(
	final_log_probs: torch.Tensor,
	anchor_logits: torch.Tensor,
	target: torch.Tensor,
	margin: float = 0.0,
	power: float = 1.0,
) -> torch.Tensor:
	"""Positive-part excess log-loss over the calibrated anchor."""

	if margin < 0:
		raise ValueError("anchor_regret_margin must be non-negative")
	if power <= 0:
		raise ValueError("anchor_regret_power must be positive")
	target = target.view(-1).long()
	final_nll = F.nll_loss(final_log_probs, target, reduction="none")
	anchor_nll = F.cross_entropy(anchor_logits, target, reduction="none")
	regret = (final_nll - anchor_nll - margin).clamp_min(0.0)
	if abs(power - 1.0) > 1e-6:
		regret = regret.pow(power)
	return regret


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
	logit_delta_home_weight: float = 1.0,
	logit_delta_draw_weight: float = 1.0,
	logit_delta_away_weight: float = 1.0,
	market_target_mix: float = 0.0,
	market_target_surprise_scale: float = 0.0,
	market_target_surprise_power: float = 1.0,
	market_target_surprise_floor: float = 0.0,
	market_target_draw_surprise_scale: Optional[float] = None,
	market_target_away_surprise_scale: Optional[float] = None,
	market_target_draw_surprise_floor: Optional[float] = None,
	market_target_away_surprise_floor: Optional[float] = None,
	market_target_surprise_mode: str = "power",
	market_target_surprise_center: float = 0.5,
	market_target_surprise_width: float = 0.3,
	market_target_surprise_slope: float = 12.0,
	market_target_draw_weight: float = 1.0,
	market_target_away_weight: float = 1.0,
	market_target_entropy_scale: float = 0.0,
	market_target_entropy_mode: str = "linear",
	sample_weights: Optional[torch.Tensor] = None,
	confidence_penalty_weight: float = 0.0,
	brier_aux_weight: float = 0.0,
	symmetric_ce_weight: float = 0.0,
	symmetric_ce_label_floor: float = 1e-4,
	gce_mix_weight: float = 0.0,
	gce_q: float = 0.7,
	bi_tempered_mix_weight: float = 0.0,
	bi_tempered_t1: float = 1.0,
	bi_tempered_t2: float = 1.0,
	bi_tempered_num_iters: int = 5,
	anchor_regret_weight: float = 0.0,
	anchor_regret_margin: float = 0.0,
	anchor_regret_power: float = 1.0,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss for the multiclass gated residual model."""

	pred_logits = model(x, cat_features, implied_probs, raw_margin)
	pred_probs = F.softmax(pred_logits, dim=-1)
	log_probs = F.log_softmax(pred_logits, dim=-1)
	implied_log = _log_softmax_from_implied(implied_probs)
	target = target.view(-1).long()
	target_distribution = F.one_hot(target, num_classes=model.n_classes).float()
	if market_target_mix > 0:
		soft_target = target_distribution
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
			power=market_target_surprise_power,
			floor=market_target_surprise_floor,
			draw_scale=market_target_draw_surprise_scale,
			away_scale=market_target_away_surprise_scale,
			draw_floor=market_target_draw_surprise_floor,
			away_floor=market_target_away_surprise_floor,
			mode=market_target_surprise_mode,
			center=market_target_surprise_center,
			width=market_target_surprise_width,
			slope=market_target_surprise_slope,
			eps=eps,
		).clamp(0.0, 1.0)
		soft_target = (1.0 - mix) * soft_target + mix * implied_probs
		target_distribution = soft_target
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

	if bi_tempered_mix_weight > 0:
		bi_tempered_loss = _bi_tempered_logistic_loss(
			pred_logits,
			target_distribution,
			t1=bi_tempered_t1,
			t2=bi_tempered_t2,
			num_iters=bi_tempered_num_iters,
			eps=eps,
		)
		base_loss = (1.0 - bi_tempered_mix_weight) * base_loss + bi_tempered_mix_weight * bi_tempered_loss

	if confidence_penalty_weight > 0:
		confidence_penalty = (pred_probs * log_probs).sum(dim=-1)
		base_loss = base_loss + confidence_penalty_weight * confidence_penalty

	if brier_aux_weight > 0:
		brier_aux = ((pred_probs - target_distribution) ** 2).sum(dim=-1)
		base_loss = base_loss + brier_aux_weight * brier_aux

	if symmetric_ce_weight > 0:
		reverse_ce = _reverse_cross_entropy(
			pred_probs,
			target_distribution,
			label_floor=symmetric_ce_label_floor,
		)
		base_loss = base_loss + symmetric_ce_weight * reverse_ce

	if sample_weights is not None:
		sample_weights = sample_weights.view(-1).clamp_min(eps)
		sample_weights = sample_weights / sample_weights.mean().clamp_min(eps)
		effective_base_loss = base_loss * sample_weights
	else:
		effective_base_loss = base_loss
	loss = effective_base_loss.mean()

	if anchor_regret_weight > 0:
		anchor_logits = model._compute_implied_logits(implied_probs, cat_features)
		regret_penalty = _anchor_regret_penalty(
			log_probs,
			anchor_logits,
			target,
			margin=anchor_regret_margin,
			power=anchor_regret_power,
		)
		if sample_weights is not None:
			regret_penalty = regret_penalty * sample_weights
		loss = loss + anchor_regret_weight * regret_penalty.mean()

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
		logit_delta_class_weights = pred_logits.new_tensor(
			[
				float(logit_delta_home_weight),
				float(logit_delta_draw_weight),
				float(logit_delta_away_weight),
			]
		).view(1, -1)
		logit_delta_penalty = ((pred_logits - implied_anchor).pow(2) * logit_delta_class_weights).mean(dim=-1)
		loss = loss + lambda_logit_delta * logit_delta_penalty.mean()

	if gate_mean_weight > 0 or gate_sat_weight > 0:
		hidden = model.backbone.get_hidden(x)
		gate = model._compute_gate(hidden, implied_probs, raw_margin)

		if gate_mean_weight > 0:
			gate_mean_loss = (gate.mean() - model.gate_target_budget).pow(2)
			loss = loss + gate_mean_weight * gate_mean_loss

		if gate_sat_weight > 0:
			sat_loss = (-torch.log(gate * (1 - gate) + eps)).mean()
			loss = loss + gate_sat_weight * sat_loss

	return loss
