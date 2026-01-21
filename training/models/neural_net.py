"""
Neural network architecture and loss functions for football match prediction.

Supports two task types:
- Binary classification (over/under 2.5 goals): 1 output, 1 implied odd
- Multiclass classification (home/draw/away): 3 outputs, 3 implied odds

Categorical Features:
- League: embedded (configurable embedding dim, default 3)
- Season stage: one-hot encoded (early/mid/late = 3 categories)
- Promoted status: binary features (home_promoted, away_promoted)
"""


from dataclasses import dataclass, field
from typing import List, Literal, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TaskType = Literal["binary", "multiclass"]


@dataclass
class CategoricalConfig:
	"""Configuration for categorical features."""
	num_leagues: int = 5  # Number of unique leagues
	league_embed_dim: int = 3  # Embedding dimension for leagues
	num_season_stages: int = 3  # early, mid, late
	# Binary features: home_promoted, away_promoted (just pass through)


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
	Embeds categorical features: league (embedding) + season_stage (one-hot) + promoted (binary).
	
	Input tensor layout (cat_features):
		- [:, 0]: league_idx (int, 0 to num_leagues-1)
		- [:, 1]: season_stage_idx (int, 0=early, 1=mid, 2=late)
		- [:, 2]: home_promoted (0 or 1)
		- [:, 3]: away_promoted (0 or 1)
	
	Output: concatenated [league_embed, season_stage_onehot, home_promoted, away_promoted]
	"""
	
	def __init__(self, cat_config: CategoricalConfig):
		super().__init__()
		self.cat_config = cat_config
		self.league_embed = nn.Embedding(cat_config.num_leagues, cat_config.league_embed_dim)
		# Output dim = league_embed_dim + num_season_stages (one-hot) + 2 (binary promoted flags)
		self.output_dim = cat_config.league_embed_dim + cat_config.num_season_stages + 2
	
	def forward(self, cat_features: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			cat_features: (batch, 4) tensor with [league_idx, stage_idx, home_promoted, away_promoted]
		Returns:
			(batch, output_dim) embedded categorical features
		"""
		league_idx = cat_features[:, 0].long()
		stage_idx = cat_features[:, 1].long()
		promoted = cat_features[:, 2:4].float()  # home_promoted, away_promoted
		
		# League embedding
		league_emb = self.league_embed(league_idx)  # (batch, league_embed_dim)
		
		# Season stage one-hot
		stage_onehot = F.one_hot(stage_idx, num_classes=self.cat_config.num_season_stages).float()  # (batch, 3)
		
		# Concatenate all
		return torch.cat([league_emb, stage_onehot, promoted], dim=-1)


class MLP(nn.Module):
	"""
	Flexible MLP with configurable layers, dropout, normalization, and activation.
	
	Supports optional categorical features via embedding layer.
	
	Args:
		input_dim: Number of continuous input features
		hidden_layers: List of hidden layer sizes
		dropout: Dropout probability
		norm: Normalization type ('none', 'bn', 'ln')
		activation: Activation function ('relu', 'silu', 'gelu', 'geglu')
		output_dim: Number of output units (1 for binary, 3 for multiclass)
		cat_config: Optional CategoricalConfig for handling categorical features
	"""

	def __init__(
		self,
		input_dim: int,
		hidden_layers: List[int],
		dropout: float = 0.3,
		norm: str = "none",
		activation: str = "relu",
		output_dim: int = 1,
		cat_config: Optional[CategoricalConfig] = None,
	):
		super().__init__()
		
		# Categorical embedder (optional)
		self.cat_embedder = None
		total_input_dim = input_dim
		if cat_config is not None:
			self.cat_embedder = CategoricalEmbedder(cat_config)
			total_input_dim = input_dim + self.cat_embedder.output_dim
		
		layers = []
		prev = total_input_dim
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
		layers.append(nn.Linear(prev, output_dim))
		self.net = nn.Sequential(*layers)
		self.input_dim = input_dim
		self.hidden_layers = hidden_layers
		self.dropout = dropout
		self.norm = norm
		self.activation = activation
		self.output_dim = output_dim
		self.cat_config = cat_config

	def forward(self, x: torch.Tensor, cat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Forward pass.
		
		Args:
			x: Continuous features (batch, input_dim)
			cat_features: Optional categorical features (batch, 4) with
				[league_idx, stage_idx, home_promoted, away_promoted]
		"""
		if self.cat_embedder is not None:
			if cat_features is None:
				raise ValueError("cat_features required when model has cat_config")
			cat_emb = self.cat_embedder(cat_features)
			x = torch.cat([x, cat_emb], dim=-1)
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
	task_type: TaskType = "binary"  # "binary" for over/under, "multiclass" for result
	cat_config: Optional[CategoricalConfig] = None  # Categorical feature config


# ============================================================================
# LOSS FUNCTIONS - BINARY (Over/Under)
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


def residual_market_loss_corr(
	residuals_logits: torch.Tensor,
	implied_prob: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	conditional_corr: bool = True,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""Loss function for residual market model with correlation penalty (binary classification)."""
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


# ============================================================================
# LOSS FUNCTIONS - MULTICLASS (Home/Draw/Away Result)
# ============================================================================


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""
	Convert implied probabilities to log-softmax values.
	
	Args:
		implied_probs: Shape (batch, 3) with [home, draw, away] probabilities
	
	Returns:
		Log-softmax values (batch, 3)
	"""
	implied_probs = torch.clamp(implied_probs, eps, 1.0 - eps)
	# Normalize to ensure they sum to 1
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	return torch.log(implied_probs)


def multiclass_batch_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Compute average correlation between predicted and implied logits across classes.
	
	Args:
		pred_logits: Shape (batch, 3)
		implied_logits: Shape (batch, 3)
	
	Returns:
		Average correlation across 3 classes
	"""
	n_classes = pred_logits.shape[1]
	total_corr = pred_logits.new_tensor(0.0)
	
	for c in range(n_classes):
		pred_c = pred_logits[:, c]
		impl_c = implied_logits[:, c]
		corr_c = batch_corr(pred_c, impl_c, eps=eps)
		total_corr = total_corr + corr_c
	
	return total_corr / n_classes


def multiclass_conditional_corr(
	pred_logits: torch.Tensor,
	implied_logits: torch.Tensor,
	target: torch.Tensor,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Compute conditional correlation for multiclass: Corr(pred, implied | y).
	
	For multiclass, we compute correlation of predicted vs implied log-probs
	for the true class, conditioned on each outcome class.
	
	Args:
		pred_logits: Shape (batch, 3) - raw logits from model + implied
		implied_logits: Shape (batch, 3) - log of implied probs
		target: Shape (batch,) - class labels 0, 1, 2
	"""
	batch_size = pred_logits.shape[0]
	target = target.view(-1).long()
	
	# Get the predicted and implied values for the true class
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
			rho_r = batch_corr(pred_r, impl_r, eps=eps)
			w_r = n_r / float(batch_size)
			rho_weighted = rho_weighted + w_r * rho_r
			weighted_sum += w_r
	
	if weighted_sum == 0:
		return batch_corr(pred_true_class, impl_true_class, eps=eps)
	
	return rho_weighted


def residual_market_loss_multiclass(
	residual_logits: torch.Tensor,
	implied_probs: torch.Tensor,
	target: torch.Tensor,
	lambda_repulsion: float = 0.0,
	lambda_corr: float = 0.0,
	conditional_corr: bool = True,
	eps: float = 1e-6,
) -> torch.Tensor:
	"""
	Loss function for multiclass (Home/Draw/Away) residual market model.
	
	The model outputs residual logits that are added to log(implied_probs).
	Final prediction: softmax(residual_logits + log(implied_probs))
	
	Args:
		residual_logits: Shape (batch, 3) - model output
		implied_probs: Shape (batch, 3) - implied probs [home, draw, away]
		target: Shape (batch,) - class labels (0=Home, 1=Draw, 2=Away)
		lambda_repulsion: Weight for repulsion penalty (encourages divergence from implied)
		lambda_corr: Weight for correlation penalty
		conditional_corr: Whether to use conditional correlation
		eps: Numerical stability epsilon
	
	Returns:
		Loss value
	"""
	# Convert implied probs to log space and add residuals
	implied_log = _log_softmax_from_implied(implied_probs, eps=eps)
	pred_logits = residual_logits + implied_log
	
	# Base cross-entropy loss
	target = target.view(-1).long()
	base = F.cross_entropy(pred_logits, target)
	loss = base
	
	if lambda_repulsion > 0.0:
		# Repulsion: encourage model to deviate from implied odds
		pred_probs = F.softmax(pred_logits, dim=-1)
		implied_normalized = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
		repulsion = ((pred_probs - implied_normalized) ** 2).mean()
		loss = loss - lambda_repulsion * repulsion
	
	if lambda_corr > 0.0:
		# Correlation penalty
		if conditional_corr:
			rho = multiclass_conditional_corr(pred_logits, implied_log, target, eps=eps)
		else:
			rho = multiclass_batch_corr(pred_logits, implied_log, eps=eps)
		
		corr_penalty = (rho + 1.0) ** 2
		loss = loss + lambda_corr * corr_penalty
	
	return loss
