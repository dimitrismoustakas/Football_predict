"""
Set Transformer model for player-level match prediction.
"""

import math

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
		pos_emb = self.position_embed(positions)
		x = torch.cat([features, pos_emb], dim=-1)
		return self.mlp(x)


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
	"""Pooling by multihead attention with one learnable seed."""

	def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
		super().__init__()
		self.seeds = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
		self.mab = MAB(dim, num_heads, dropout)

	def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
		seeds = self.seeds.expand(x.size(0), -1, -1)
		return self.mab(seeds, x, key_padding_mask=key_padding_mask)


class SetTransformerTeamEncoder(nn.Module):
	"""Self-attention over players followed by PMA pooling."""

	def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32,
				 num_heads: int = 4, num_sab_layers: int = 2,
				 position_embed_dim: int = 4, dropout: float = 0.1):
		super().__init__()
		self.encoder = PlayerEncoder(input_dim, hidden_dim, position_embed_dim, dropout)
		self.sab_layers = nn.ModuleList([
			SAB(hidden_dim, num_heads, dropout) for _ in range(num_sab_layers)
		])
		self.pma = PMA(hidden_dim, num_heads, dropout)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_dim, output_dim),
			nn.GELU(),
		)

	def forward(self, features: torch.Tensor, positions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		encoded = self.encoder(features, positions)
		attn_mask = ~mask
		for sab in self.sab_layers:
			encoded = sab(encoded, key_padding_mask=attn_mask)
		pooled = self.pma(encoded, key_padding_mask=attn_mask).squeeze(1)
		return self.decoder(pooled)


class PlayerMatchModel(nn.Module):
	"""Match model built on top of a shared Set Transformer team encoder."""

	def __init__(
		self,
		input_dim: int,
		team_encoder_type: str = "set_transformer",
		hidden_dim: int = 64,
		team_output_dim: int = 32,
		num_heads: int = 4,
		num_sab_layers: int = 2,
		position_embed_dim: int = 4,
		dropout: float = 0.1,
		use_implied: bool = True,
		head_type: str = "mlp",
		gate_hidden_dim: int = 32,
		gate_target_budget: float = 0.2,
		market_feature_stats: int = 3,
		market_logit_scale: float = 1.0,
		learn_market_class_scale: bool = False,
	):
		super().__init__()
		if team_encoder_type != "set_transformer":
			raise ValueError(f"Unknown team_encoder_type: {team_encoder_type}")
		if market_feature_stats not in {3, 4, 5}:
			raise ValueError(f"Unsupported market_feature_stats: {market_feature_stats}")

		self.use_implied = use_implied
		self.head_type = head_type
		self.market_feature_stats = market_feature_stats
		self.market_logit_scale = market_logit_scale
		self.market_feature_dim = 6 + max(0, market_feature_stats - 3)

		self.team_encoder = SetTransformerTeamEncoder(
			input_dim=input_dim,
			hidden_dim=hidden_dim,
			output_dim=team_output_dim,
			num_heads=num_heads,
			num_sab_layers=num_sab_layers,
			position_embed_dim=position_embed_dim,
			dropout=dropout,
		)

		context_dim = team_output_dim * 3
		if head_type == "mlp":
			head_input_dim = context_dim + (3 if use_implied else 0)
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
			self.market_class_scale = None
		elif head_type == "gated_residual":
			if not use_implied:
				raise ValueError("gated_residual head requires use_implied=True")
			self.head = None
			self.residual_head = nn.Sequential(
				nn.Linear(context_dim, 64),
				nn.GELU(),
				nn.Dropout(dropout),
				nn.Linear(64, 32),
				nn.GELU(),
				nn.Dropout(dropout),
				nn.Linear(32, 3),
			)
			self.gate_head = nn.Sequential(
				nn.Linear(context_dim + self.market_feature_dim, gate_hidden_dim),
				nn.GELU(),
				nn.Dropout(dropout * 0.5),
				nn.Linear(gate_hidden_dim, 3),
			)
			init_bias = math.log(gate_target_budget / (1.0 - gate_target_budget))
			self.gate_bias = nn.Parameter(torch.full((3,), init_bias))
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
		return log_implied

	def _compute_gate(self, combined: torch.Tensor, implied: torch.Tensor, raw_margin: torch.Tensor) -> torch.Tensor:
		gate_input = torch.cat([combined, self._compute_market_features(implied, raw_margin)], dim=-1)
		return torch.sigmoid(self.gate_head(gate_input) + self.gate_bias)
