"""
Standalone training script for the player-level Set Transformer model.

Evaluates whether player-set representations add predictive value beyond
team-level aggregates and bookmaker implied probabilities.

Usage:
    python -m training.train_player_model [--encoder deep_sets|deep_sets_role_pool|deep_sets_stats|weighted_deep_sets|set_transformer]
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from preprocessing.lineup_builder import (
	PLAYER_FEATURE_COLS,
	NUM_FEATURES,
	build_projected_squads,
	assemble_squad_tensors,
)
from preprocessing.player_feature_engineering import (
	compute_player_rolling_features,
	load_all_player_data,
)
from training.evaluation.metrics import accuracy_score, log_loss, ranked_probability_score
from training.models.set_transformer import PlayerMatchModel
from training.train_utils import (
	add_targets_and_implied,
	evaluate_implied_baseline,
	filter_min_history,
	generate_rolling_cv_folds,
	get_sorted_seasons,
	load_frame,
)

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
DEFAULT_CONFIG = {
	"encoder_type": "deep_sets",
	"hidden_dim": 64,
	"team_output_dim": 32,
	"num_heads": 4,
	"num_sab_layers": 2,
	"position_embed_dim": 4,
	"dropout": 0.15,
	"use_implied": True,
	"head_type": "mlp",
	"mlp_market_features": False,
	"linear_residual_head": False,
	"gate_hidden_dim": 32,
	"gate_target_budget": 0.2,
	"gate_use_market_features": True,
	"shared_gate": False,
	"linear_gate": False,
	"market_feature_stats": 3,
	"market_logit_scale": 1.0,
	"learn_market_bias": False,
	"learn_market_class_scale": False,
	"gate_mean_weight": 0.0,
	"gate_sat_weight": 0.0,
	"lambda_repulsion": 0.0,
	"lambda_logit_delta": 0.0,
	"lr": 1e-3,
	"weight_decay": 1e-4,
	"batch_size": 256,
	"max_epochs": 50,
	"patience": 5,
	"top_n_players": 16,
	"seed": 42,
}


def build_player_model(config: dict) -> PlayerMatchModel:
	"""Build a player-set model from the experiment config."""

	return PlayerMatchModel(
		input_dim=NUM_FEATURES,
		team_encoder_type=config["encoder_type"],
		hidden_dim=config["hidden_dim"],
		team_output_dim=config["team_output_dim"],
		num_heads=config["num_heads"],
		num_sab_layers=config["num_sab_layers"],
		position_embed_dim=config["position_embed_dim"],
		dropout=config["dropout"],
		use_implied=config["use_implied"],
		head_type=config["head_type"],
		mlp_market_features=config["mlp_market_features"],
		linear_residual_head=config["linear_residual_head"],
		gate_hidden_dim=config["gate_hidden_dim"],
		gate_target_budget=config["gate_target_budget"],
		gate_use_market_features=config["gate_use_market_features"],
		shared_gate=config["shared_gate"],
		linear_gate=config["linear_gate"],
		market_feature_stats=config["market_feature_stats"],
		market_logit_scale=config["market_logit_scale"],
		learn_market_bias=config["learn_market_bias"],
		learn_market_class_scale=config["learn_market_class_scale"],
	)


def clone_squad_tensors(squad_tensors: dict) -> dict:
	"""Clone tensor-like squad inputs so experiments can mutate them safely."""

	cloned = {}
	for key, value in squad_tensors.items():
		if isinstance(value, np.ndarray):
			cloned[key] = value.copy()
		elif isinstance(value, list):
			cloned[key] = list(value)
		else:
			cloned[key] = value
	return cloned


def shuffle_squad_features(squad_tensors: dict, seed: int = 42) -> dict:
	"""Shuffle valid player feature slots across matches while preserving masks."""

	shuffled = clone_squad_tensors(squad_tensors)
	rng = np.random.RandomState(seed)
	for side in ["home", "away"]:
		features = shuffled[f"{side}_players"]
		mask = shuffled[f"{side}_mask"]
		for feature_idx in range(features.shape[2]):
			column = features[:, :, feature_idx].copy()
			valid_values = column[mask].copy()
			rng.shuffle(valid_values)
			column[mask] = valid_values
			features[:, :, feature_idx] = column
	return shuffled


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PlayerMatchDataset(Dataset):
	"""PyTorch dataset for player-set match prediction."""

	def __init__(
		self,
		squad_tensors: dict,
		labels: np.ndarray,
		implied: np.ndarray,
		raw_margin: np.ndarray,
		match_indices: np.ndarray,
	):
		self.home_players = torch.tensor(squad_tensors["home_players"][match_indices], dtype=torch.float32)
		self.away_players = torch.tensor(squad_tensors["away_players"][match_indices], dtype=torch.float32)
		self.home_positions = torch.tensor(squad_tensors["home_positions"][match_indices], dtype=torch.long)
		self.away_positions = torch.tensor(squad_tensors["away_positions"][match_indices], dtype=torch.long)
		self.home_mask = torch.tensor(squad_tensors["home_mask"][match_indices], dtype=torch.bool)
		self.away_mask = torch.tensor(squad_tensors["away_mask"][match_indices], dtype=torch.bool)
		self.labels = torch.tensor(labels, dtype=torch.long)
		self.implied = torch.tensor(implied, dtype=torch.float32)
		self.raw_margin = torch.tensor(raw_margin, dtype=torch.float32)

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		return (
			self.home_players[idx],
			self.home_positions[idx],
			self.home_mask[idx],
			self.away_players[idx],
			self.away_positions[idx],
			self.away_mask[idx],
			self.implied[idx],
			self.labels[idx],
			self.raw_margin[idx],
		)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_match_data(df: pl.DataFrame) -> pl.DataFrame:
	"""Add targets and filter to usable matches."""
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	df = df.filter(
		(pl.col("odds_home") > 1.0) &
		(pl.col("odds_draw") > 1.0) &
		(pl.col("odds_away") > 1.0) &
		pl.col("implied_home").is_finite() &
		pl.col("implied_draw").is_finite() &
		pl.col("implied_away").is_finite()
	)
	return df


def build_squad_data(match_df: pl.DataFrame, top_n: int = 16) -> Tuple[dict, pl.DataFrame]:
	"""
	Build squad tensors aligned with match_df rows.

	Returns:
		squad_tensors: dict with arrays indexed by match position
		aligned_df: match_df reordered to match squad tensor ordering
	"""
	print("Loading raw player data...")
	raw = load_all_player_data()
	print(f"  {len(raw)} player-match records")

	print("Computing per-player rolling features...")
	rolling = compute_player_rolling_features(raw)
	non_null = rolling.drop_nulls(subset=PLAYER_FEATURE_COLS[:1])
	print(f"  {len(non_null)} player-match records with valid rolling features")

	print(f"Building projected squads (top {top_n} per team)...")
	squads = build_projected_squads(raw, rolling, match_df=match_df, top_n=top_n)
	print(f"  {len(squads)} squad entries")

	print("Assembling squad tensors...")
	squad_tensors = assemble_squad_tensors(squads, match_df, max_players=top_n)
	tensor_game_ids = squad_tensors["game_ids"]
	print(f"  {len(tensor_game_ids)} matches with squad tensors")

	# Align match_df to tensor ordering
	game_id_order = pl.DataFrame({"game_id": tensor_game_ids, "_tensor_idx": range(len(tensor_game_ids))})
	aligned_df = match_df.join(game_id_order, on="game_id", how="inner").sort("_tensor_idx")

	return squad_tensors, aligned_df


def split_by_seasons(
	aligned_df: pl.DataFrame,
	train_seasons: List[str],
	val_seasons: List[str],
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
	"""Split aligned data into train/val by season, returning indices and label arrays."""
	season_str = aligned_df["season"].cast(pl.Utf8)

	train_idx = aligned_df.filter(season_str.is_in(train_seasons))["_tensor_idx"].to_numpy()
	val_idx = aligned_df.filter(season_str.is_in(val_seasons))["_tensor_idx"].to_numpy()

	train_df = aligned_df.filter(season_str.is_in(train_seasons))
	val_df = aligned_df.filter(season_str.is_in(val_seasons))

	def extract_arrays(part: pl.DataFrame) -> dict:
		return {
			"y": part["result_label"].to_numpy().astype(int),
			"implied": np.stack([
				part["implied_home"].to_numpy(),
				part["implied_draw"].to_numpy(),
				part["implied_away"].to_numpy(),
			], axis=1).astype(np.float64),
			"raw_margin": part["raw_margin"].to_numpy().astype(np.float64),
			"odds_home": part["odds_home"].to_numpy(),
			"odds_draw": part["odds_draw"].to_numpy(),
			"odds_away": part["odds_away"].to_numpy(),
			"dates": part["date"].to_numpy() if "date" in part.columns else None,
		}

	return train_idx, val_idx, extract_arrays(train_df), extract_arrays(val_df)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	config: dict,
) -> float:
	model.train()
	total_loss = 0.0
	n_batches = 0

	for batch in loader:
		(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask,
		 implied, labels, raw_margin) = [t.to(device) for t in batch]

		optimizer.zero_grad()
		if getattr(model, "head_type", "mlp") == "gated_residual":
			logits, components = model(
				home_feat,
				home_pos,
				home_mask,
				away_feat,
				away_pos,
				away_mask,
				implied,
				raw_margin,
				return_components=True,
			)
			loss = F.cross_entropy(logits, labels)
			gate_mean_weight = float(config.get("gate_mean_weight", 0.0))
			gate_sat_weight = float(config.get("gate_sat_weight", 0.0))
			lambda_repulsion = float(config.get("lambda_repulsion", 0.0))
			lambda_logit_delta = float(config.get("lambda_logit_delta", 0.0))
			if gate_mean_weight > 0:
				loss = loss + gate_mean_weight * (components["gate"].mean() - float(config["gate_target_budget"])).pow(2)
			if gate_sat_weight > 0:
				loss = loss + gate_sat_weight * (-torch.log(components["gate"] * (1.0 - components["gate"]) + 1e-6)).mean()
			if lambda_repulsion > 0:
				implied_norm = implied / implied.sum(dim=-1, keepdim=True).clamp(min=1e-6)
				pred_probs = F.softmax(logits, dim=-1)
				loss = loss - lambda_repulsion * ((pred_probs - implied_norm) ** 2).mean()
			if lambda_logit_delta > 0:
				loss = loss + lambda_logit_delta * (logits - components["anchor_logits"]).pow(2).mean()
		else:
			logits = model(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied, raw_margin)
			loss = F.cross_entropy(logits, labels)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()
		n_batches += 1

	return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> Tuple[float, np.ndarray]:
	"""Evaluate model, returning loss and predicted probabilities."""
	model.eval()
	total_loss = 0.0
	n_batches = 0
	all_probs = []

	for batch in loader:
		(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask,
		 implied, labels, raw_margin) = [t.to(device) for t in batch]

		logits = model(home_feat, home_pos, home_mask, away_feat, away_pos, away_mask, implied, raw_margin)
		# Clamp logits to avoid NaN from attention layers
		logits = logits.clamp(-20, 20)
		loss = F.cross_entropy(logits, labels)
		total_loss += loss.item()
		n_batches += 1
		all_probs.append(F.softmax(logits, dim=-1).cpu().numpy())

	avg_loss = total_loss / max(n_batches, 1)
	probs = np.concatenate(all_probs, axis=0)
	# Safety: replace any residual NaN with uniform
	nan_mask = np.isnan(probs).any(axis=1)
	if nan_mask.any():
		probs[nan_mask] = 1.0 / 3.0
	return avg_loss, probs


def compute_metrics(probs: np.ndarray, data: dict) -> dict:
	"""Compute standard evaluation metrics."""
	y_true = data["y"]
	preds = np.argmax(probs, axis=1)
	acc = accuracy_score(y_true, preds)
	ll = log_loss(y_true, probs, labels=[0, 1, 2])
	rps = ranked_probability_score(y_true, probs)
	y_onehot = np.eye(3)[y_true]
	brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
	return {"accuracy": float(acc), "log_loss": float(ll), "rps": float(rps), "brier": brier}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_player_model(config: dict = None):
	if config is None:
		config = DEFAULT_CONFIG.copy()

	set_seed(config["seed"])
	print(f"{'=' * 60}")
	print(f"  Player Set Model Training ({config['encoder_type']})")
	print(f"{'=' * 60}")
	print(f"Device: {DEVICE}")

	# Load match data
	print(f"\nLoading match data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = prepare_match_data(df)
	print(f"Usable matches: {len(df)}")

	# Build squad tensors
	squad_tensors, aligned_df = build_squad_data(df, top_n=config["top_n_players"])
	print(f"Aligned matches: {len(aligned_df)}")

	# Season splits
	test_season = get_sorted_seasons(aligned_df)[-1]
	folds = generate_rolling_cv_folds(aligned_df, n_folds=3, test_season=test_season)

	# Use last fold for quick evaluation
	train_seasons, val_season = folds[-1]
	print(f"\nTrain seasons: {train_seasons}")
	print(f"Val season: {val_season}")
	print(f"Test season: {test_season}")

	# Prepare data splits
	train_idx, val_idx, train_data, val_data = split_by_seasons(
		aligned_df, train_seasons, [val_season])
	_, test_idx, _, test_data = split_by_seasons(
		aligned_df, train_seasons + [val_season], [test_season])

	print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

	# Baseline
	print("\n--- Implied Baseline ---")
	val_baseline = evaluate_implied_baseline(val_data)
	test_baseline = evaluate_implied_baseline(test_data)
	print(f"Val:  acc={val_baseline['accuracy']:.4f}  log_loss={val_baseline['log_loss']:.5f}  rps={val_baseline['rps']:.5f}")
	print(f"Test: acc={test_baseline['accuracy']:.4f}  log_loss={test_baseline['log_loss']:.5f}  rps={test_baseline['rps']:.5f}")

	# Datasets and loaders
	train_dataset = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
	val_dataset = PlayerMatchDataset(squad_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
	test_dataset = PlayerMatchDataset(squad_tensors, test_data["y"], test_data["implied"], test_data["raw_margin"], test_idx)

	train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

	# Build model
	model = build_player_model(config).to(DEVICE)

	n_params = sum(p.numel() for p in model.parameters())
	print(f"\nModel parameters: {n_params:,}")

	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=config["lr"],
		weight_decay=config["weight_decay"],
	)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=config["max_epochs"], eta_min=config["lr"] * 0.01)

	# Training loop with early stopping
	best_val_loss = float("inf")
	best_epoch = 0
	patience_counter = 0
	best_state = None

	print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>7} | {'Val LL':>8} | {'Val RPS':>8}")
	print("-" * 65)

	for epoch in range(1, config["max_epochs"] + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, config)
		val_loss, val_probs = evaluate(model, val_loader, DEVICE)
		val_metrics = compute_metrics(val_probs, val_data)
		scheduler.step()

		print(f"{epoch:>5} | {train_loss:>10.5f} | {val_loss:>10.5f} | {val_metrics['accuracy']:>7.4f} | {val_metrics['log_loss']:>8.5f} | {val_metrics['rps']:>8.5f}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			patience_counter = 0
			best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
		else:
			patience_counter += 1
			if patience_counter >= config["patience"]:
				print(f"\nEarly stopping at epoch {epoch} (best={best_epoch})")
				break

	# Restore best model and evaluate on test set
	if best_state is not None:
		model.load_state_dict(best_state)
		model.to(DEVICE)

	print(f"\n--- Test Set Evaluation (best epoch {best_epoch}) ---")
	_, test_probs = evaluate(model, test_loader, DEVICE)
	test_metrics = compute_metrics(test_probs, test_data)

	print(f"  Accuracy:  {test_metrics['accuracy']:.4f}  (baseline: {test_baseline['accuracy']:.4f})")
	print(f"  Log Loss:  {test_metrics['log_loss']:.5f}  (baseline: {test_baseline['log_loss']:.5f})")
	print(f"  RPS:       {test_metrics['rps']:.5f}  (baseline: {test_baseline['rps']:.5f})")
	print(f"  Brier:     {test_metrics['brier']:.5f}")

	delta_ll = test_metrics["log_loss"] - test_baseline["log_loss"]
	delta_rps = test_metrics["rps"] - test_baseline["rps"]
	print(f"\n  Delta LL vs implied:  {delta_ll:+.5f} ({'better' if delta_ll < 0 else 'worse'})")
	print(f"  Delta RPS vs implied: {delta_rps:+.5f} ({'better' if delta_rps < 0 else 'worse'})")

	return {
		"best_epoch": best_epoch,
		"test_metrics": test_metrics,
		"test_baseline": test_baseline,
		"config": config,
	}


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train player-level set model")
	parser.add_argument("--encoder", choices=["deep_sets", "deep_sets_role_pool", "deep_sets_stats", "weighted_deep_sets", "set_transformer"], default="deep_sets")
	parser.add_argument("--head-type", choices=["mlp", "gated_residual"], default="mlp")
	parser.add_argument("--mlp-market-features", action="store_true")
	parser.add_argument("--linear-residual-head", action="store_true")
	parser.add_argument("--no-implied", action="store_true", help="Don't use implied probabilities as input")
	parser.add_argument("--hidden-dim", type=int)
	parser.add_argument("--team-output-dim", type=int)
	parser.add_argument("--dropout", type=float)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--weight-decay", type=float)
	parser.add_argument("--gate-hidden-dim", type=int)
	parser.add_argument("--gate-target-budget", type=float)
	parser.add_argument("--player-only-gate", action="store_true")
	parser.add_argument("--shared-gate", action="store_true")
	parser.add_argument("--linear-gate", action="store_true")
	parser.add_argument("--market-feature-stats", type=int, choices=[3, 4, 5])
	parser.add_argument("--market-logit-scale", type=float)
	parser.add_argument("--learn-market-bias", action="store_true")
	parser.add_argument("--learn-market-class-scale", action="store_true")
	parser.add_argument("--gate-mean-weight", type=float)
	parser.add_argument("--gate-sat-weight", type=float)
	parser.add_argument("--lambda-repulsion", type=float)
	parser.add_argument("--lambda-logit-delta", type=float)
	args = parser.parse_args()

	config = DEFAULT_CONFIG.copy()
	config["encoder_type"] = args.encoder
	config["head_type"] = args.head_type
	if args.mlp_market_features:
		config["mlp_market_features"] = True
	if args.linear_residual_head:
		config["linear_residual_head"] = True
	if args.no_implied:
		config["use_implied"] = False
	if args.hidden_dim is not None:
		config["hidden_dim"] = args.hidden_dim
	if args.team_output_dim is not None:
		config["team_output_dim"] = args.team_output_dim
	if args.dropout is not None:
		config["dropout"] = args.dropout
	if args.lr is not None:
		config["lr"] = args.lr
	if args.weight_decay is not None:
		config["weight_decay"] = args.weight_decay
	if args.gate_hidden_dim is not None:
		config["gate_hidden_dim"] = args.gate_hidden_dim
	if args.gate_target_budget is not None:
		config["gate_target_budget"] = args.gate_target_budget
	if args.player_only_gate:
		config["gate_use_market_features"] = False
	if args.shared_gate:
		config["shared_gate"] = True
	if args.linear_gate:
		config["linear_gate"] = True
	if args.market_feature_stats is not None:
		config["market_feature_stats"] = args.market_feature_stats
	if args.market_logit_scale is not None:
		config["market_logit_scale"] = args.market_logit_scale
	if args.learn_market_bias:
		config["learn_market_bias"] = True
	if args.learn_market_class_scale:
		config["learn_market_class_scale"] = True
	if args.gate_mean_weight is not None:
		config["gate_mean_weight"] = args.gate_mean_weight
	if args.gate_sat_weight is not None:
		config["gate_sat_weight"] = args.gate_sat_weight
	if args.lambda_repulsion is not None:
		config["lambda_repulsion"] = args.lambda_repulsion
	if args.lambda_logit_delta is not None:
		config["lambda_logit_delta"] = args.lambda_logit_delta

	train_player_model(config)
