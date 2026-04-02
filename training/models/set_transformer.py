"""
Set-based neural network architectures for player-level match prediction.

Two variants:
1. DeepSetsModel  — simple MLP-encode-then-pool (baseline)
2. WeightedDeepSetsModel — Deep Sets with learned permutation-invariant pooling
3. SetTransformerModel — self-attention over player embeddings + PMA pooling

Both take variable-size player sets (with padding masks) for home and away
teams and output Home/Draw/Away logits.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.player_feature_engineering import POSITION_VOCABULARY


NUM_POSITIONS = len(POSITION_VOCABULARY)  # 17 real positions
PAD_IDX = 0  # position index 0 = padding
ROLE_NAMES = ["gk", "def", "mid", "att", "sub"]
POSITION_TO_ROLE_NAME = {
	"GK": "gk",
	"DC": "def",
	"DL": "def",
	"DR": "def",
	"DML": "def",
	"DMR": "def",
	"DMC": "def",
	"MC": "mid",
	"ML": "mid",
	"MR": "mid",
	"AMC": "mid",
	"AML": "mid",
	"AMR": "mid",
	"FW": "att",
	"FWL": "att",
	"FWR": "att",
	"Sub": "sub",
}
ROLE_TO_IDX = {name: idx for idx, name in enumerate(ROLE_NAMES)}
POSITION_ROLE_IDS = [ROLE_TO_IDX[POSITION_TO_ROLE_NAME[pos]] for pos in POSITION_VOCABULARY]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PositionEmbedding(nn.Module):
	"""Learnable position embeddings (0 = padding, 1-17 = real positions)."""

	def __init__(self, embed_dim: int = 4):
		super().__init__()
		self.embed = nn.Embedding(NUM_POSITIONS + 1, embed_dim, padding_idx=PAD_IDX)

	def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
		return self.embed(position_ids)


class PlayerEncoder(nn.Module):
	"""Per-player MLP: (continuous_features || position_embed) -> hidden."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.position_embed = PositionEmbedding(position_embed_dim)
		total_in = input_dim + position_embed_dim
		self.mlp = nn.Sequential(
			nn.Linear(total_in, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			features:  (batch, max_players, D_feat)
			positions: (batch, max_players) int

		Returns:
			(batch, max_players, hidden_dim)
		"""
		pos_emb = self.position_embed(positions)
		x = torch.cat([features, pos_emb], dim=-1)
		return self.mlp(x)


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""
	Mean-pool over the player dimension, respecting the padding mask.

	Args:
		x:    (batch, max_players, hidden_dim)
		mask: (batch, max_players) bool — True where a real player exists

	Returns:
		(batch, hidden_dim)
	"""
	mask_f = mask.unsqueeze(-1).float()  # (batch, max_players, 1)
	summed = (x * mask_f).sum(dim=1)     # (batch, hidden_dim)
	counts = mask_f.sum(dim=1).clamp(min=1.0)  # (batch, 1)
	return summed / counts


def masked_max_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""Max-pool over the player dimension, respecting the padding mask."""

	fill_value = torch.finfo(x.dtype).min
	masked_x = x.masked_fill(~mask.unsqueeze(-1), fill_value)
	pooled = masked_x.max(dim=1).values
	empty_rows = ~mask.any(dim=1)
	if empty_rows.any():
		pooled = pooled.clone()
		pooled[empty_rows] = 0.0
	return pooled


def masked_weighted_pool(
	x: torch.Tensor,
	mask: torch.Tensor,
	logits: torch.Tensor,
) -> torch.Tensor:
	"""Weighted permutation-invariant pooling over valid player slots only."""

	mask_f = mask.float()
	masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
	weights = torch.softmax(masked_logits, dim=1) * mask_f
	weight_sums = weights.sum(dim=1, keepdim=True)
	weights = torch.where(
		weight_sums > 0,
		weights / weight_sums.clamp(min=1e-12),
		torch.zeros_like(weights),
	)
	return (x * weights.unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Multihead Attention Block (for Set Transformer)
# ---------------------------------------------------------------------------

class MAB(nn.Module):
	"""Multihead Attention Block: MAB(X, Y) = LayerNorm(H + rFF(H))
	where H = LayerNorm(X + MultiheadAttn(X, Y, Y))."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.ff = nn.Sequential(
			nn.Linear(dim, dim * 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim * 2, dim),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor, y: torch.Tensor,
				key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		"""
		Args:
			x: (batch, n, dim) — queries
			y: (batch, m, dim) — keys/values
			key_padding_mask: (batch, m) bool — True where padded (to be ignored)

		Returns:
			(batch, n, dim)
		"""
		h, _ = self.attn(x, y, y, key_padding_mask=key_padding_mask)
		h = self.norm1(x + h)
		return self.norm2(h + self.ff(h))


class SAB(nn.Module):
	"""Self-Attention Block: SAB(X) = MAB(X, X)."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.mab = MAB(dim, num_heads, dropout)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		return self.mab(x, x, key_padding_mask=key_padding_mask)


class PMA(nn.Module):
	"""Pooling by Multihead Attention: uses k learnable seed vectors to
	aggregate a set into k fixed-size outputs."""

	def __init__(self, dim: int, num_heads: int = 4, num_seeds: int = 1, dropout: float = 0.1):
		super().__init__()
		self.seeds = nn.Parameter(torch.randn(1, num_seeds, dim) * 0.02)
		self.mab = MAB(dim, num_heads, dropout)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		"""
		Args:
			x: (batch, n, dim)
			key_padding_mask: (batch, n) bool — True where padded

		Returns:
			(batch, num_seeds, dim)
		"""
		seeds = self.seeds.expand(x.size(0), -1, -1)
		return self.mab(seeds, x, key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# Team encoder: processes one team's player set -> fixed-size representation
# ---------------------------------------------------------------------------

class DeepSetsTeamEncoder(nn.Module):
	"""Encode-then-pool: MLP per player, then masked mean pooling."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			features:  (batch, max_players, D)
			positions: (batch, max_players)
			mask:      (batch, max_players) bool

		Returns:
			(batch, output_dim)
		"""
		encoded = self.encoder(features, positions)  # (batch, max_players, hidden)
		pooled = masked_mean_pool(encoded, mask)      # (batch, hidden)
		return self.decoder(pooled)                   # (batch, output_dim)


class WeightedDeepSetsTeamEncoder(nn.Module):
	"""Deep Sets with a learned per-player importance score before pooling."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		self.pool_scorer = nn.Linear(hidden_dim, 1)
		nn.init.zeros_(self.pool_scorer.weight)
		nn.init.zeros_(self.pool_scorer.bias)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(features, positions)
		pool_logits = self.pool_scorer(encoded).squeeze(-1)
		pooled = masked_weighted_pool(encoded, mask, pool_logits)
		return self.decoder(pooled)


class StatsDeepSetsTeamEncoder(nn.Module):
	"""Deep Sets with concatenated mean and max pooled summaries."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim * 2 + 1, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(features, positions)
		mean_pooled = masked_mean_pool(encoded, mask)
		max_pooled = masked_max_pool(encoded, mask)
		player_count = mask.float().sum(dim=1, keepdim=True)
		summary = torch.cat([mean_pooled, max_pooled, player_count], dim=-1)
		return self.decoder(summary)


class RoleAwareDeepSetsTeamEncoder(nn.Module):
	"""Deep Sets with separate mean pools for broad football role buckets."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		role_lookup = torch.full((NUM_POSITIONS + 1,), -1, dtype=torch.long)
		for pos_idx, role_idx in enumerate(POSITION_ROLE_IDS, start=1):
			role_lookup[pos_idx] = role_idx
		self.register_buffer("role_lookup", role_lookup)
		self.num_roles = len(ROLE_NAMES)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim * (1 + self.num_roles) + self.num_roles, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(features, positions)
		overall_mean = masked_mean_pool(encoded, mask)
		role_ids = self.role_lookup[positions]
		role_means = []
		role_counts = []
		for role_idx in range(self.num_roles):
			role_mask = mask & (role_ids == role_idx)
			role_means.append(masked_mean_pool(encoded, role_mask))
			role_counts.append(role_mask.float().sum(dim=1, keepdim=True))
		summary = torch.cat([overall_mean, *role_means, *role_counts], dim=-1)
		return self.decoder(summary)


class SetTransformerTeamEncoder(nn.Module):
	"""Self-attention over player embeddings + PMA pooling."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 num_heads: int = 4, num_sab_layers: int = 2,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		self.sab_layers = nn.ModuleList([
			SAB(hidden_dim, num_heads, dropout) for _ in range(num_sab_layers)
		])
		self.pma = PMA(hidden_dim, num_heads, num_seeds=1, dropout=dropout)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor,
				mask: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			features:  (batch, max_players, D)
			positions: (batch, max_players)
			mask:      (batch, max_players) bool

		Returns:
			(batch, output_dim)
		"""
		encoded = self.encoder(features, positions)

		# Attention mask: True = ignore (PyTorch convention)
		attn_mask = ~mask

		for sab in self.sab_layers:
			encoded = sab(encoded, key_padding_mask=attn_mask)

		pooled = self.pma(encoded, key_padding_mask=attn_mask)  # (batch, 1, hidden)
		pooled = pooled.squeeze(1)                               # (batch, hidden)
		return self.decoder(pooled)


# ---------------------------------------------------------------------------
# Full match model: two team encoders -> classification head
# ---------------------------------------------------------------------------

class PlayerMatchModel(nn.Module):
	"""
	Full match-level model using player sets.

	Encodes home and away teams separately via a shared team encoder,
	then combines representations for Home/Draw/Away prediction.

	The model optionally accepts implied probabilities from bookmaker odds
	to use as a baseline (similar to the gated residual approach).
	"""

	def __init__(
		self,
		input_dim: int,
		team_encoder_type: str = "deep_sets",
		hidden_dim: int = 64,
		team_output_dim: int = 32,
		num_heads: int = 4,
		num_sab_layers: int = 2,
		position_embed_dim: int = 4,
		dropout: float = 0.1,
		use_implied: bool = True,
		head_type: str = "mlp",
		mlp_market_features: bool = False,
		linear_residual_head: bool = False,
		gate_hidden_dim: int = 32,
		gate_target_budget: float = 0.2,
		gate_use_market_features: bool = True,
		shared_gate: bool = False,
		linear_gate: bool = False,
		market_feature_stats: int = 3,
		market_logit_scale: float = 1.0,
		learn_market_bias: bool = False,
		learn_market_class_scale: bool = False,
	):
		super().__init__()
		self.use_implied = use_implied
		self.head_type = head_type
		self.mlp_market_features = mlp_market_features
		self.linear_residual_head = linear_residual_head
		self.gate_use_market_features = gate_use_market_features
		self.shared_gate = shared_gate
		self.linear_gate = linear_gate
		self.market_feature_stats = market_feature_stats
		self.market_logit_scale = market_logit_scale
		self.gate_target_budget = gate_target_budget
		self.market_feature_dim = 6 + max(0, market_feature_stats - 3)

		# Shared team encoder
		if team_encoder_type == "deep_sets":
			self.team_encoder = DeepSetsTeamEncoder(
				input_dim, hidden_dim, team_output_dim, position_embed_dim, dropout)
		elif team_encoder_type == "deep_sets_role_pool":
			self.team_encoder = RoleAwareDeepSetsTeamEncoder(
				input_dim, hidden_dim, team_output_dim, position_embed_dim, dropout)
		elif team_encoder_type == "deep_sets_stats":
			self.team_encoder = StatsDeepSetsTeamEncoder(
				input_dim, hidden_dim, team_output_dim, position_embed_dim, dropout)
		elif team_encoder_type == "weighted_deep_sets":
			self.team_encoder = WeightedDeepSetsTeamEncoder(
				input_dim, hidden_dim, team_output_dim, position_embed_dim, dropout)
		elif team_encoder_type == "set_transformer":
			self.team_encoder = SetTransformerTeamEncoder(
				input_dim, hidden_dim, team_output_dim, num_heads, num_sab_layers,
				position_embed_dim, dropout)
		else:
			raise ValueError(f"Unknown team_encoder_type: {team_encoder_type}")

		context_dim = team_output_dim * 3
		if head_type == "mlp":
			head_input_dim = context_dim
			if use_implied:
				if mlp_market_features:
					if market_feature_stats not in {3, 4, 5}:
						raise ValueError(f"Unsupported market_feature_stats: {market_feature_stats}")
					head_input_dim += self.market_feature_dim
				else:
					head_input_dim += 3
			self.head = nn.Sequential(
				nn.Linear(head_input_dim, 64),
				nn.GELU(),
				nn.Dropout(dropout),
				nn.Linear(64, 32),
				nn.GELU(),
				nn.Dropout(dropout),
				nn.Linear(32, 3),
			)
			self.residual_head = None
			self.gate_head = None
			self.gate_bias = None
		elif head_type == "gated_residual":
			if not use_implied:
				raise ValueError("gated_residual head requires use_implied=True")
			if market_feature_stats not in {3, 4, 5}:
				raise ValueError(f"Unsupported market_feature_stats: {market_feature_stats}")
			self.head = None
			if linear_residual_head:
				self.residual_head = nn.Linear(context_dim, 3, bias=False)
			else:
				self.residual_head = nn.Sequential(
					nn.Linear(context_dim, 64),
					nn.GELU(),
					nn.Dropout(dropout),
					nn.Linear(64, 32),
					nn.GELU(),
					nn.Dropout(dropout),
					nn.Linear(32, 3),
				)
			gate_output_dim = 1 if shared_gate else 3
			gate_input_dim = context_dim + (self.market_feature_dim if gate_use_market_features else 0)
			if linear_gate:
				self.gate_head = nn.Linear(gate_input_dim, gate_output_dim)
			else:
				self.gate_head = nn.Sequential(
					nn.Linear(gate_input_dim, gate_hidden_dim),
					nn.GELU(),
					nn.Dropout(dropout * 0.5),
					nn.Linear(gate_hidden_dim, gate_output_dim),
				)
			init_bias = math.log(gate_target_budget / (1.0 - gate_target_budget))
			self.gate_bias = nn.Parameter(torch.full((gate_output_dim,), init_bias))
			if learn_market_bias:
				self.market_bias = nn.Parameter(torch.zeros(3))
			else:
				self.register_buffer("market_bias", torch.zeros(3))
			if learn_market_class_scale:
				self.market_class_scale = nn.Parameter(torch.zeros(3))
			else:
				self.market_class_scale = None
		else:
			raise ValueError(f"Unknown head_type: {head_type}")

	def forward(
		self,
		home_features: torch.Tensor,
		home_positions: torch.Tensor,
		home_mask: torch.Tensor,
		away_features: torch.Tensor,
		away_positions: torch.Tensor,
		away_mask: torch.Tensor,
		implied: torch.Tensor | None = None,
		raw_margin: torch.Tensor | None = None,
		return_components: bool = False,
	) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
		"""
		Args:
			home_features:  (batch, max_players, D)
			home_positions: (batch, max_players) int
			home_mask:      (batch, max_players) bool
			away_features:  (batch, max_players, D)
			away_positions: (batch, max_players) int
			away_mask:      (batch, max_players) bool
			implied:        (batch, 3) float — normalized bookmaker probabilities

		Returns:
			(batch, 3) logits for Home/Draw/Away
		"""
		combined = self.encode_match_context(
			home_features,
			home_positions,
			home_mask,
			away_features,
			away_positions,
			away_mask,
		)

		if self.use_implied and implied is not None:
			if self.head_type == "mlp":
				if self.mlp_market_features:
					if raw_margin is None:
						raise ValueError("raw_margin is required when mlp_market_features=True")
					combined = torch.cat([combined, self._compute_market_features(implied, raw_margin)], dim=-1)
				else:
					combined = torch.cat([combined, implied], dim=-1)
			elif self.head_type == "gated_residual":
				if raw_margin is None:
					raise ValueError("raw_margin is required for gated_residual head")
				anchor_logits = self._compute_implied_logits(implied)
				residual_logits = self.residual_head(combined)
				if raw_margin.ndim == 1:
					raw_margin = raw_margin.unsqueeze(-1)
				gate = self._compute_gate(combined, implied, raw_margin)
				logits = anchor_logits + gate * residual_logits
				if return_components:
					return logits, {
						"gate": gate,
						"anchor_logits": anchor_logits,
						"residual_logits": residual_logits,
					}
				return logits

		if self.head is None:
			raise ValueError("head is not available for the current PlayerMatchModel configuration")
		logits = self.head(combined)
		if return_components:
			return logits, {}
		return logits

	def encode_match_context(
		self,
		home_features: torch.Tensor,
		home_positions: torch.Tensor,
		home_mask: torch.Tensor,
		away_features: torch.Tensor,
		away_positions: torch.Tensor,
		away_mask: torch.Tensor,
	) -> torch.Tensor:
		"""Return the player-derived match context before the prediction head."""

		home_repr = self.team_encoder(home_features, home_positions, home_mask)
		away_repr = self.team_encoder(away_features, away_positions, away_mask)
		diff = home_repr - away_repr

		return torch.cat([home_repr, away_repr, diff], dim=-1)

	@staticmethod
	def _normalized_market_entropy(implied: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
		entropy = -torch.sum(implied * torch.log(implied + eps), dim=-1, keepdim=True)
		return entropy / math.log(implied.size(-1))

	def _compute_market_features(self, implied: torch.Tensor, raw_margin: torch.Tensor) -> torch.Tensor:
		entropy = self._normalized_market_entropy(implied)
		max_prob = implied.max(dim=-1, keepdim=True)[0]
		min_prob = implied.min(dim=-1, keepdim=True)[0]
		sorted_probs = torch.sort(implied, dim=-1, descending=True)[0]
		top2_gap = sorted_probs[:, :1] - sorted_probs[:, 1:2]
		if raw_margin.ndim == 1:
			raw_margin = raw_margin.unsqueeze(-1)
		feature_parts = [implied, entropy, max_prob, raw_margin]
		if self.market_feature_stats >= 4:
			feature_parts.append(min_prob)
		if self.market_feature_stats >= 5:
			feature_parts.append(top2_gap)
		return torch.cat(feature_parts, dim=-1)

	def _compute_implied_logits(self, implied: torch.Tensor) -> torch.Tensor:
		log_implied = implied.clamp(min=1e-6).log() * self.market_logit_scale
		if self.market_class_scale is not None:
			log_implied = log_implied * torch.exp(self.market_class_scale)
		return log_implied + self.market_bias

	def _compute_gate(self, combined: torch.Tensor, implied: torch.Tensor, raw_margin: torch.Tensor) -> torch.Tensor:
		if self.gate_use_market_features:
			gate_input = torch.cat([combined, self._compute_market_features(implied, raw_margin)], dim=-1)
		else:
			gate_input = combined
		gate = torch.sigmoid(self.gate_head(gate_input) + self.gate_bias)
		if self.shared_gate:
			gate = gate.expand(-1, 3)
		return gate
