"""
Set Transformer model for player-level match prediction.
"""

import torch
import torch.nn as nn

from preprocessing.player_feature_engineering import POSITION_VOCABULARY


NUM_POSITIONS = len(POSITION_VOCABULARY)
PAD_IDX = 0


class PositionEmbedding(nn.Module):
	"""Learnable position embeddings (0 = padding, 1-N = real positions)."""

	def __init__(self, embed_dim: int = 4):
		super().__init__()
		self.embed = nn.Embedding(NUM_POSITIONS + 1, embed_dim, padding_idx=PAD_IDX)

	def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
		return self.embed(position_ids)


class PlayerEncoder(nn.Module):
	"""Per-player MLP over numeric features plus position embedding."""

	def __init__(
		self,
		input_dim: int,
		hidden_dim: int = 64,
		position_embed_dim: int = 4,
		dropout: float = 0.1,
	):
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
		pos_emb = self.position_embed(positions)
		return self.mlp(torch.cat([features, pos_emb], dim=-1))


class MAB(nn.Module):
	"""Multihead attention block."""

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

	def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		h, _ = self.attn(x, y, y, key_padding_mask=key_padding_mask)
		h = self.norm1(x + h)
		return self.norm2(h + self.ff(h))


class SAB(nn.Module):
	"""Self-attention block."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.mab = MAB(dim, num_heads, dropout)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		return self.mab(x, x, key_padding_mask=key_padding_mask)


class PMA(nn.Module):
	"""Pooling by multihead attention with a single learnable seed."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.seed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
		self.mab = MAB(dim, num_heads, dropout)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		seed = self.seed.expand(x.size(0), -1, -1)
		return self.mab(seed, x, key_padding_mask=key_padding_mask)


class SetTransformerTeamEncoder(nn.Module):
	"""Self-attention over players followed by PMA pooling."""

	def __init__(
		self,
		input_dim: int,
		hidden_dim: int = 64,
		output_dim: int = 32,
		num_heads: int = 4,
		num_sab_layers: int = 2,
		position_embed_dim: int = 4,
		dropout: float = 0.1,
	):
		super().__init__()
		self.output_dim = output_dim
		self.encoder = PlayerEncoder(
			input_dim=input_dim,
			hidden_dim=hidden_dim,
			position_embed_dim=position_embed_dim,
			dropout=dropout,
		)
		self.sab_layers = nn.ModuleList([
			SAB(hidden_dim, num_heads, dropout) for _ in range(num_sab_layers)
		])
		self.pma = PMA(hidden_dim, num_heads, dropout)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def encode_players(self, features: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(features, positions)
		attn_mask = ~mask
		for sab in self.sab_layers:
			encoded = sab(encoded, key_padding_mask=attn_mask)
		return encoded

	def pool_players(self, encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		if encoded.size(1) == 0:
			return encoded.new_zeros((encoded.size(0), self.output_dim))
		safe_encoded = encoded
		safe_mask = mask
		empty_rows = ~mask.any(dim=1)
		if empty_rows.any():
			safe_encoded = encoded.clone()
			safe_mask = mask.clone()
			safe_encoded[empty_rows, 0] = 0.0
			safe_mask[empty_rows, 0] = True
		pooled = self.pma(safe_encoded, key_padding_mask=~safe_mask).squeeze(1)
		decoded = self.decoder(pooled)
		if empty_rows.any():
			decoded = decoded.clone()
			decoded[empty_rows] = 0.0
		return decoded

	def forward(self, features: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encode_players(features, positions, mask)
		return self.pool_players(encoded, mask)


class CrossTeamInteractionBlock(nn.Module):
	"""Bidirectional cross-attention between home and away player sets."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.home_to_away = MAB(dim, num_heads, dropout)
		self.away_to_home = MAB(dim, num_heads, dropout)

	def forward(
		self,
		home_encoded: torch.Tensor,
		away_encoded: torch.Tensor,
		home_key_padding_mask: torch.Tensor | None = None,
		away_key_padding_mask: torch.Tensor | None = None,
	) -> tuple[torch.Tensor, torch.Tensor]:
		home_updated = self.home_to_away(home_encoded, away_encoded, key_padding_mask=away_key_padding_mask)
		away_updated = self.away_to_home(away_encoded, home_encoded, key_padding_mask=home_key_padding_mask)
		return home_updated, away_updated


class PlayerMatchModel(nn.Module):
	"""Match model built on top of a shared Set Transformer team encoder."""

	def __init__(
		self,
		input_dim: int,
		hidden_dim: int = 64,
		team_output_dim: int = 32,
		num_heads: int = 4,
		num_sab_layers: int = 2,
		position_embed_dim: int = 4,
		num_cross_team_layers: int = 0,
		dropout: float = 0.1,
	):
		super().__init__()
		if num_cross_team_layers < 0:
			raise ValueError(f"num_cross_team_layers must be >= 0, got {num_cross_team_layers}")

		self.team_encoder = SetTransformerTeamEncoder(
			input_dim=input_dim,
			hidden_dim=hidden_dim,
			output_dim=team_output_dim,
			num_heads=num_heads,
			num_sab_layers=num_sab_layers,
			position_embed_dim=position_embed_dim,
			dropout=dropout,
		)
		self.cross_team_layers = nn.ModuleList([
			CrossTeamInteractionBlock(hidden_dim, num_heads, dropout)
			for _ in range(num_cross_team_layers)
		])
		self.head = nn.Sequential(
			nn.Linear(team_output_dim * 3 + 3, 64),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(32, 3),
		)

	def encode_match_context(
		self,
		home_features: torch.Tensor,
		home_positions: torch.Tensor,
		home_mask: torch.Tensor,
		away_features: torch.Tensor,
		away_positions: torch.Tensor,
		away_mask: torch.Tensor,
	) -> torch.Tensor:
		home_encoded = self.team_encoder.encode_players(home_features, home_positions, home_mask)
		away_encoded = self.team_encoder.encode_players(away_features, away_positions, away_mask)
		home_padding_mask = ~home_mask
		away_padding_mask = ~away_mask
		for cross_team_layer in self.cross_team_layers:
			home_encoded, away_encoded = cross_team_layer(
				home_encoded,
				away_encoded,
				home_key_padding_mask=home_padding_mask,
				away_key_padding_mask=away_padding_mask,
			)
		home_repr = self.team_encoder.pool_players(home_encoded, home_mask)
		away_repr = self.team_encoder.pool_players(away_encoded, away_mask)
		return torch.cat([home_repr, away_repr, home_repr - away_repr], dim=-1)

	def forward(
		self,
		home_features: torch.Tensor,
		home_positions: torch.Tensor,
		home_mask: torch.Tensor,
		away_features: torch.Tensor,
		away_positions: torch.Tensor,
		away_mask: torch.Tensor,
		implied: torch.Tensor,
	) -> torch.Tensor:
		combined = self.encode_match_context(
			home_features,
			home_positions,
			home_mask,
			away_features,
			away_positions,
			away_mask,
		)
		return self.head(torch.cat([combined, implied], dim=-1))
