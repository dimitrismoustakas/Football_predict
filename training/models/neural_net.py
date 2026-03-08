"""
Neural network architecture and loss functions for football match prediction.

Supports two task types:
- Binary classification (over/under 2.5 goals): 1 output, 1 implied odd
- Multiclass classification (home/draw/away): 3 outputs, 3 implied odds

Architecture: Contextual Gated Residual Model
- Learns WHEN to deviate from market odds using a context-dependent gate
- Gate g(x) = sigmoid(gate_head([h(x), market_features]))
- pred_logits = log(p_mkt) + g * r(x)

Categorical Features (via CategoricalEmbedder):
- League: embedded (configurable embedding dim, default 3)
- Promoted status: binary features (home_promoted, away_promoted)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TaskType = Literal["binary", "multiclass"]


@dataclass
class CategoricalConfig:
	"""Configuration for categorical features (league embedding + promoted flags)."""
	num_leagues: int = 5
	league_embed_dim: int = 3


class GeGLU(nn.Module):
	"""Gated GELU activation with linear projection."""

	def __init__(self, in_dim: int, hidden_dim: int, bias: bool = True):
		super().__init__()
		self.proj = nn.Linear(in_dim, 2 * hidden_dim, bias=bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		v, g = self.proj(x).chunk(2, dim=-1)
		return v * F.gelu(g)


class CategoricalEmbedder(nn.Module):
	"""
	Embeds categorical features: league (embedding) + promoted flags (binary).
	
	Input tensor layout (cat_features):
		- [:, 0]: league_idx (int, 0 to num_leagues-1)
		- [:, 1]: home_promoted (0 or 1)
		- [:, 2]: away_promoted (0 or 1)
	
	Output: concatenated [league_embed, home_promoted, away_promoted]
	"""
	
	def __init__(self, cat_config: CategoricalConfig):
		super().__init__()
		self.cat_config = cat_config
		self.league_embed = nn.Embedding(cat_config.num_leagues, cat_config.league_embed_dim)
		# Output dim = league_embed_dim + 2 (binary promoted flags)
		self.output_dim = cat_config.league_embed_dim + 2
	
	def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			cat_features: (batch, 3) tensor with [league_idx, home_promoted, away_promoted]
		Returns:
			(batch, output_dim) embedded categorical features
		"""
		league_idx = cat_features[:, 0].long()
		promoted = cat_features[:, 1:3].float()  # home_promoted, away_promoted
		
		league_emb = self.league_embed(league_idx)  # (batch, league_embed_dim)
		return torch.cat([league_emb, promoted], dim=-1)


class MLPWithHiddenAccess(nn.Module):
	"""
	MLP that provides access to the last hidden layer representation.
	
	This is the base network that extracts h(x) before the final linear layer.
	Used by the gated models to compute both residual logits and gates.
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
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.output_dim = output_dim
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


# ============================================================================
# CONTEXTUAL GATED RESIDUAL MODEL - MULTICLASS (Home/Draw/Away)
# ============================================================================

class GatedResidualModel(nn.Module):
	"""
	Neural network with contextual gating that learns WHEN to deviate from market.
	
	The gate g is computed from both hidden features and market features:
		g = sigmoid(gate_head([h(x), market_features]))
		pred_logits = log p_mkt + g * r(x)
	
	Where:
		- h(x): hidden representation from the base network
		- market_features: implied probabilities + derived features (entropy, confidence, margin)
		- g: per-sample, per-class gate controlling how much to trust model
		- r(x): residual logits from base model
	
	This allows the model to learn contexts where it should trust itself more
	(e.g., when market is uncertain, specific leagues, etc.)
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
		gate_target_budget: float = 0.2,
	):
		super().__init__()
		self.n_classes = n_classes
		self.gate_target_budget = gate_target_budget
		
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
		
		# Gate head: takes [h(x), market_features] -> g (n_classes-vector)
		# Market features: implied probs + derived features (entropy, max_prob, raw_margin)
		self.market_feature_dim = market_feature_dim + 3  # +3 for derived features
		gate_input_dim = self.base_model.hidden_dim + self.market_feature_dim
		
		self.gate_head = nn.Sequential(
			nn.Linear(gate_input_dim, gate_hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 0.5),
			nn.Linear(gate_hidden_dim, n_classes),
		)
		
		# Learnable bias in logit space - initialized to target budget
		init_bias = math.log(gate_target_budget / (1 - gate_target_budget))
		self.gate_bias = nn.Parameter(torch.full((n_classes,), init_bias))
		
		# Store config for serialization
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
		"""
		Compute market features from implied probabilities.
		
		Features:
			- implied_probs: normalized probabilities (n_classes)
			- entropy: market uncertainty (1)
			- max_prob: market confidence (1)
			- raw_margin: bookmaker overround (1)
		"""
		eps = 1e-6
		
		# Entropy: -sum(p * log(p)), normalized to [0, 1]
		entropy = -torch.sum(implied_probs * torch.log(implied_probs + eps), dim=-1, keepdim=True)
		entropy = entropy / math.log(self.n_classes)
		
		# Max probability (market confidence)
		max_prob = implied_probs.max(dim=-1, keepdim=True)[0]
		
		# Raw margin (overround)
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
		"""
		Forward pass with contextual gating.
		
		Args:
			x: Continuous features (batch, input_dim)
			cat_features: Optional categorical features (batch, n_cat)
			implied_probs: Market implied probabilities (batch, n_classes)
			raw_margin: Raw margin before normalization (batch,)
		
		Returns:
			Final logits: log(p_mkt) + g * r(x)
		"""
		# Get hidden representation and residual logits
		h = self.base_model.get_hidden(x, cat_features)
		residual_logits = self.base_model.final_layer(h)
		
		if implied_probs is None:
			return residual_logits
		
		if raw_margin is None:
			raise ValueError("raw_margin is required when implied_probs is provided")
		
		# Compute market features
		market_features = self._compute_market_features(implied_probs, raw_margin)
		
		# Compute gate: g = sigmoid(logits + bias)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = self.gate_head(gate_input)
		g = torch.sigmoid(gate_logits + self.gate_bias)
		
		# Convert implied probs to log space
		implied_log = _log_softmax_from_implied(implied_probs)
		
		# Final prediction: log p_mkt + g * r(x)
		pred_logits = implied_log + g * residual_logits
		
		return pred_logits
	
	def get_gate_stats(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor],
		implied_probs: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> Dict[str, np.ndarray]:
		"""Get gate statistics for analysis."""
		self.eval()
		with torch.no_grad():
			h = self.base_model.get_hidden(x, cat_features)
			market_features = self._compute_market_features(implied_probs, raw_margin)
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


# ============================================================================
# CONTEXTUAL GATED RESIDUAL MODEL - BINARY (Over/Under)
# ============================================================================

class GatedResidualModelBinary(nn.Module):
	"""
	Binary version of the gated residual model for over/under prediction.
	
	Same architecture as multiclass but with 1 output:
		g = sigmoid(gate_head([h(x), market_features]))
		pred_logit = logit(p_mkt) + g * r(x)
	"""
	
	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		cat_config: Optional[CategoricalConfig] = None,
		gate_hidden_dim: int = 32,
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
		gate_target_budget: float = 0.2,
	):
		super().__init__()
		self.gate_target_budget = gate_target_budget
		
		# Base model that produces residual logit and hidden representation
		self.base_model = MLPWithHiddenAccess(
			input_dim=input_dim,
			hidden_layers=hidden_layers,
			dropout=dropout,
			norm=norm,
			activation=activation,
			output_dim=1,
			cat_config=cat_config,
		)
		
		# Market features for binary: implied_prob (1) + raw_margin (1) + entropy-like (1)
		# For binary, "entropy" is just -p*log(p) - (1-p)*log(1-p)
		self.market_feature_dim = 3
		gate_input_dim = self.base_model.hidden_dim + self.market_feature_dim
		
		self.gate_head = nn.Sequential(
			nn.Linear(gate_input_dim, gate_hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout * 0.5),
			nn.Linear(gate_hidden_dim, 1),
		)
		
		# Learnable bias in logit space
		init_bias = math.log(gate_target_budget / (1 - gate_target_budget))
		self.gate_bias = nn.Parameter(torch.tensor([init_bias]))
		
		# Store config for serialization
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.cat_config = cat_config
		self.gate_hidden_dim = gate_hidden_dim
	
	def _compute_market_features(
		self, 
		implied_prob: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> torch.Tensor:
		"""Compute market features from implied probability."""
		eps = 1e-6
		p = implied_prob.view(-1, 1)
		
		# Binary entropy: -p*log(p) - (1-p)*log(1-p), normalized to [0,1] (max at p=0.5 is log(2))
		entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
		entropy = entropy / math.log(2)
		
		# Raw margin
		if raw_margin.dim() == 1:
			raw_margin = raw_margin.unsqueeze(-1)
		
		return torch.cat([p, entropy, raw_margin], dim=-1)
	
	def forward(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor] = None,
		implied_prob: Optional[torch.Tensor] = None,
		raw_margin: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		"""
		Forward pass with contextual gating.
		
		Args:
			x: Continuous features (batch, input_dim)
			cat_features: Optional categorical features
			implied_prob: Market implied probability for over (batch,) or (batch, 1)
			raw_margin: Raw margin before normalization (batch,)
		
		Returns:
			Final logit: logit(p_mkt) + g * r(x)
		"""
		# Get hidden representation and residual logit
		h = self.base_model.get_hidden(x, cat_features)
		residual_logit = self.base_model.final_layer(h)  # (batch, 1)
		
		if implied_prob is None:
			return residual_logit.squeeze(-1)
		
		if raw_margin is None:
			raise ValueError("raw_margin is required when implied_prob is provided")
		
		# Compute market features
		market_features = self._compute_market_features(implied_prob, raw_margin)
		
		# Compute gate
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logit = self.gate_head(gate_input)
		g = torch.sigmoid(gate_logit + self.gate_bias)  # (batch, 1)
		
		# Convert implied prob to logit space
		implied_logit = _logits(implied_prob.view(-1, 1))
		
		# Final prediction: logit(p_mkt) + g * r(x)
		pred_logit = implied_logit + g * residual_logit
		
		return pred_logit.squeeze(-1)
	
	def get_gate_stats(
		self,
		x: torch.Tensor,
		cat_features: Optional[torch.Tensor],
		implied_prob: torch.Tensor,
		raw_margin: torch.Tensor,
	) -> Dict[str, float]:
		"""Get gate statistics for analysis."""
		self.eval()
		with torch.no_grad():
			h = self.base_model.get_hidden(x, cat_features)
			market_features = self._compute_market_features(implied_prob, raw_margin)
			gate_input = torch.cat([h, market_features], dim=-1)
			gate_logit = self.gate_head(gate_input)
			g = torch.sigmoid(gate_logit + self.gate_bias).squeeze(-1)
		
		return {
			"gate_values": g.cpu().numpy(),
			"gate_mean": float(g.mean().cpu()),
			"gate_std": float(g.std().cpu()),
			"gate_min": float(g.min().cpu()),
			"gate_max": float(g.max().cpu()),
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
	task_type: TaskType = "binary"  # "binary" for over/under, "multiclass" for result
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
	# Gated model specific
	gate_hidden_dim: int = 32
	gate_target_budget: float = 0.2
	gate_mean_weight: float = 0.01  # Regularization weight for mean gate budget
	gate_sat_weight: float = 0.001  # Anti-saturation regularization
	lambda_repulsion: float = 0.0  # Encourages deviation from implied odds
	lambda_corr: float = 0.0  # Penalizes correlation with implied odds


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert probability to logits with numerical stability."""
	p = torch.clamp(p, eps, 1 - eps)
	return torch.log(p) - torch.log(1 - p)


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""
	Convert implied probabilities to log-softmax values.
	
	Args:
		implied_probs: Shape (batch, 3) with [home, draw, away] probabilities
	
	Returns:
		Log-softmax values (batch, 3)
	"""
	implied_probs = torch.clamp(implied_probs, eps, 1.0 - eps)
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	return torch.log(implied_probs)


# ============================================================================
# LOSS FUNCTIONS - GATED MODELS
# ============================================================================


def _batch_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Compute batch correlation between two tensors."""
	x = x - x.mean()
	y = y - y.mean()
	vx = x.var(unbiased=False) + eps
	vy = y.var(unbiased=False) + eps
	cov = (x * y).mean()
	return cov / torch.sqrt(vx * vy)


def _logits_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Approximate Corr(pred_logits, implied_logits | y) for binary classification."""
	pred = pred_logits.view(-1)
	impl = implied_logits.view(-1)
	y = target.view(-1)
	total_n = pred.shape[0]
	rho_weighted = pred.new_tensor(0.0)
	weighted_sum = 0.0
	
	for r in [0, 1]:
		mask = y == r
		n_r = int(mask.sum().item())
		if n_r > 1:
			pred_r = pred[mask]
			impl_r = impl[mask]
			rho_r = _batch_corr(pred_r, impl_r, eps=eps)
			w_r = n_r / float(total_n)
			rho_weighted = rho_weighted + w_r * rho_r
			weighted_sum += w_r
	
	if weighted_sum == 0:
		return _batch_corr(pred, impl, eps=eps)
	
	return rho_weighted


def _multiclass_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Compute conditional correlation for multiclass: Corr(pred, implied | y).
	
	For multiclass, we compute correlation of predicted vs implied log-probs
	for the true class, conditioned on each outcome class.
	"""
	batch_size = pred_logits.shape[0]
	target = target.view(-1).long()
	
	# Get logits for true class
	pred_true_class = pred_logits.gather(1, target.unsqueeze(1)).squeeze(1)
	impl_true_class = implied_logits.gather(1, target.unsqueeze(1)).squeeze(1)
	
	rho_weighted = pred_logits.new_tensor(0.0)
	weighted_sum = 0.0
	
	# Weight correlation by class frequency
	for r in range(3):
		mask = target == r
		n_r = int(mask.sum().item())
		if n_r > 1:
			pred_r = pred_true_class[mask]
			impl_r = impl_true_class[mask]
			rho_r = _batch_corr(pred_r, impl_r, eps=eps)
			w_r = n_r / float(batch_size)
			rho_weighted = rho_weighted + w_r * rho_r
			weighted_sum += w_r
	
	if weighted_sum == 0:
		return _batch_corr(pred_true_class, impl_true_class, eps=eps)
	
	return rho_weighted


def gated_loss_multiclass(
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
	"""
	Loss function for gated multiclass model.
	
	Uses mean-budget regularization to encourage "mostly small, sometimes large" gates.
	
	Args:
		model: GatedResidualModel instance
		x: Input features
		cat_features: Optional categorical features
		implied_probs: Market implied probabilities
		target: Class labels
		raw_margin: Raw margin before normalization
		gate_mean_weight: Weight for mean-budget regularization
		gate_sat_weight: Weight for anti-saturation regularization
		lambda_repulsion: Weight for repulsion term (encourages deviation from market)
		lambda_corr: Weight for correlation penalty (penalizes correlation with implied)
		eps: Numerical stability epsilon
	
	Returns:
		Loss value
	"""
	# Get predictions (model handles gating internally)
	pred_logits = model(x, cat_features, implied_probs, raw_margin)
	pred_probs = F.softmax(pred_logits, dim=-1)
	
	# Get implied log probs for correlation computation
	implied_log = _log_softmax_from_implied(implied_probs)
	
	# Cross-entropy loss
	target = target.view(-1).long()
	loss = F.cross_entropy(pred_logits, target)
	
	# Repulsion term: encourage deviation from implied odds
	if lambda_repulsion > 0:
		# Negative L2 distance encourages predictions to differ from market
		implied_normalized = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
		repulsion = ((pred_probs - implied_normalized) ** 2).mean()
		loss = loss - lambda_repulsion * repulsion
	
	# Correlation penalty: penalize correlation between pred logits and implied logits
	if lambda_corr > 0:
		# Conditional correlation in logit space
		rho = _multiclass_conditional_corr(pred_logits, implied_log, target, eps=eps)
		# Penalty: (rho + 1)^2 penalizes high positive correlation
		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty
	
	# Gate regularization
	if gate_mean_weight > 0 or gate_sat_weight > 0:
		h = model.base_model.get_hidden(x, cat_features)
		market_features = model._compute_market_features(implied_probs, raw_margin)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logits = model.gate_head(gate_input)
		g = torch.sigmoid(gate_logits + model.gate_bias)
		
		# Mean-budget regularizer: penalize deviation from target mean
		if gate_mean_weight > 0:
			gate_mean_loss = (g.mean() - model.gate_target_budget).pow(2)
			loss = loss + gate_mean_weight * gate_mean_loss
		
		# Anti-saturation: tiny penalty for gates too close to 0 or 1
		if gate_sat_weight > 0:
			sat_loss = (-torch.log(g * (1 - g) + eps)).mean()
			loss = loss + gate_sat_weight * sat_loss
	
	return loss


def gated_loss_binary(
	model: GatedResidualModelBinary,
	x: torch.Tensor,
	cat_features: Optional[torch.Tensor],
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	raw_margin: torch.Tensor,
	gate_mean_weight: float = 0.01,
	gate_sat_weight: float = 0.001,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Loss function for gated binary model (over/under).
	
	Args:
		model: GatedResidualModelBinary instance
		x: Input features
		cat_features: Optional categorical features
		implied_prob: Market implied probability for over
		target: Binary target (0 or 1)
		raw_margin: Raw margin before normalization
		gate_mean_weight: Weight for mean-budget regularization
		gate_sat_weight: Weight for anti-saturation regularization
		lambda_repulsion: Weight for repulsion term (encourages deviation from market)
		lambda_corr: Weight for correlation penalty (penalizes correlation with implied)
		eps: Numerical stability epsilon
	
	Returns:
		Loss value
	"""
	# Get predictions
	pred_logit = model(x, cat_features, implied_prob, raw_margin)
	pred_prob = torch.sigmoid(pred_logit)
	
	# Get implied logit for correlation computation
	implied_logit = _logits(implied_prob.view(-1))
	
	# Binary cross-entropy loss
	target = target.view(-1)
	loss = F.binary_cross_entropy_with_logits(pred_logit, target)
	
	# Repulsion term: encourage deviation from implied odds
	if lambda_repulsion > 0:
		# Negative squared distance encourages predictions to differ from market
		implied_flat = implied_prob.view(-1)
		repulsion = ((pred_prob - implied_flat) ** 2).mean()
		loss = loss - lambda_repulsion * repulsion
	
	# Correlation penalty: penalize correlation between pred logits and implied logits
	if lambda_corr > 0:
		# Conditional correlation in logit space
		rho = _logits_conditional_corr(pred_logit, implied_logit, target, eps=eps)
		# Penalty: (rho + 1)^2 penalizes high positive correlation
		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty
	
	# Gate regularization
	if gate_mean_weight > 0 or gate_sat_weight > 0:
		h = model.base_model.get_hidden(x, cat_features)
		market_features = model._compute_market_features(implied_prob, raw_margin)
		gate_input = torch.cat([h, market_features], dim=-1)
		gate_logit = model.gate_head(gate_input)
		g = torch.sigmoid(gate_logit + model.gate_bias).squeeze(-1)
		
		if gate_mean_weight > 0:
			gate_mean_loss = (g.mean() - model.gate_target_budget).pow(2)
			loss = loss + gate_mean_weight * gate_mean_loss
		
		if gate_sat_weight > 0:
			sat_loss = (-torch.log(g * (1 - g) + eps)).mean()
			loss = loss + gate_sat_weight * sat_loss
	
	return loss
