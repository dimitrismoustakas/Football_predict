"""
Set-based neural network architectures for player-level match prediction.

Two variants:
1. DeepSetsModel  — simple MLP-encode-then-pool (baseline)
2. SetTransformerModel — self-attention over player embeddings + PMA pooling

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
	):
		super().__init__()
		self.use_implied = use_implied

		# Shared team encoder
		if team_encoder_type == "deep_sets":
			self.team_encoder = DeepSetsTeamEncoder(
				input_dim, hidden_dim, team_output_dim, position_embed_dim, dropout)
		elif team_encoder_type == "set_transformer":
			self.team_encoder = SetTransformerTeamEncoder(
				input_dim, hidden_dim, team_output_dim, num_heads, num_sab_layers,
				position_embed_dim, dropout)
		else:
			raise ValueError(f"Unknown team_encoder_type: {team_encoder_type}")

		# Classification head
		# Input: home_repr || away_repr || home_repr - away_repr
		head_input_dim = team_output_dim * 3
		if use_implied:
			head_input_dim += 3  # append implied probabilities

		self.head = nn.Sequential(
			nn.Linear(head_input_dim, 64),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(32, 3),
		)

	def forward(
		self,
		home_features: torch.Tensor,
		home_positions: torch.Tensor,
		home_mask: torch.Tensor,
		away_features: torch.Tensor,
		away_positions: torch.Tensor,
		away_mask: torch.Tensor,
		implied: torch.Tensor | None = None,
	) -> torch.Tensor:
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
		home_repr = self.team_encoder(home_features, home_positions, home_mask)
		away_repr = self.team_encoder(away_features, away_positions, away_mask)
		diff = home_repr - away_repr

		combined = torch.cat([home_repr, away_repr, diff], dim=-1)

		if self.use_implied and implied is not None:
			combined = torch.cat([combined, implied], dim=-1)

		return self.head(combined)
