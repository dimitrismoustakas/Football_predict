"""
Canonical training entry point for the player-level Set Transformer model.

Uses the same rolling CV, epoch-selection, and fixed held-out test protocol as
the main model while writing to a separate append-only ledger.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from csv import DictReader, DictWriter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
	NUM_FEATURES,
	build_projected_squads,
	assemble_squad_tensors,
)
from preprocessing.player_feature_engineering import (
	compute_player_rolling_features,
	load_all_player_data,
)
from training.evaluation.metrics import evaluate_profit, ranked_probability_score
from training.models.set_transformer import PlayerMatchModel
from training.train_utils import (
	add_targets_and_implied,
	build_data_snapshot,
	evaluate_implied_baseline,
	filter_min_history,
	generate_rolling_cv_folds,
	load_frame,
	resolve_test_season,
)
from utils.paths import EXPERIMENT_METRICS_DIR, MODELS_DIR, PROJECT_ROOT
from utils.portfolio import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION, evaluate_bankroll_strategy

DEFAULT_PARQUET = Path(os.environ.get("PARQUET_PATH", "data/training/understat_df.parquet"))
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "evaluation.json"
LATEST_SET_TRANSFORMER_METRICS_PATH = MODELS_DIR / "latest_set_transformer_metrics.json"
EXPERIMENT_LOG_PATH = EXPERIMENT_METRICS_DIR / "set_transformer_runs.tsv"
DISPLAY_NAME = "Player Set Transformer"
MODEL_NAME = "set_transformer"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LESS_IS_BETTER_METRICS = {"log_loss", "rps", "brier"}

EXPERIMENT_LOG_COLUMNS = [
	"recorded_at_utc",
	"git_commit",
	"git_branch",
	"cv_log_loss",
	"delta",
	"best_epoch",
	"status",
	"description",
	"cv_rps",
	"val_log_loss",
	"test_log_loss",
	"cv_metrics_json",
	"test_metrics_json",
]

# Training hyperparameters
DEFAULT_CONFIG = {
	"encoder_type": "set_transformer",
	"hidden_dim": 64,
	"team_output_dim": 32,
	"num_heads": 4,
	"num_sab_layers": 2,
	"position_embed_dim": 4,
	"dropout": 0.15,
	"use_implied": True,
	"head_type": "mlp",
	"gate_hidden_dim": 32,
	"gate_target_budget": 0.2,
	"market_feature_stats": 3,
	"market_logit_scale": 1.0,
	"learn_market_class_scale": False,
	"aux_context_loss_weight": 0.0,
	"lr": 1e-3,
	"weight_decay": 1e-4,
	"batch_size": 256,
	"max_epochs": 50,
	"patience": 5,
	"min_selection_epoch": 1,
	"top_n_players": 16,
	"seed": 42,
}


def print_header(text: str):
	print("\n" + "=" * 60)
	print(text)
	print("=" * 60)


def summarize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
	keys = [
		"accuracy",
		"brier",
		"rps",
		"log_loss",
		"corr_with_implied",
		"bankroll_roi",
		"bankroll_bet_count",
		"max_drawdown",
	]
	return {key: metrics[key] for key in keys if key in metrics}


def mean_metric(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
	if not metrics_list:
		return {}
	keys = metrics_list[0].keys()
	return {
		key: float(np.mean([metrics[key] for metrics in metrics_list]))
		for key in keys
		if all(isinstance(metrics.get(key), (int, float)) for metrics in metrics_list)
	}


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def write_json(path: Path, payload: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, "w", encoding="utf-8") as file:
		json.dump(payload, file, indent=2)


def serialize_log_value(value):
	if isinstance(value, (dict, list)):
		return json.dumps(value, separators=(",", ":"), sort_keys=True)
	if value is None:
		return ""
	return value


def append_tsv_row(path: Path, row: dict):
	path.parent.mkdir(parents=True, exist_ok=True)
	serialized = {key: serialize_log_value(row.get(key, "")) for key in EXPERIMENT_LOG_COLUMNS}
	write_header = not path.exists() or path.stat().st_size == 0
	with open(path, "a", encoding="utf-8", newline="") as file:
		writer = DictWriter(file, fieldnames=EXPERIMENT_LOG_COLUMNS, delimiter="\t")
		if write_header:
			writer.writeheader()
		writer.writerow(serialized)


def load_evaluation_config() -> dict:
	raw = load_json(EVALUATION_CONFIG_PATH)
	required = [
		"comparison_metric",
		"training_seed",
		"rolling_cv_n_folds",
		"test_season",
		"test_role",
	]
	missing = [key for key in required if key not in raw]
	if missing:
		raise ValueError(f"Missing evaluation config keys: {missing}")
	config = dict(raw)
	config["comparison_metric"] = str(config["comparison_metric"])
	config["training_seed"] = int(config["training_seed"])
	config["rolling_cv_n_folds"] = int(config["rolling_cv_n_folds"])
	config["test_season"] = str(config["test_season"])
	config["test_role"] = str(config["test_role"])
	if config["test_role"] not in {"held_out", "acceptance"}:
		raise ValueError(f"Unsupported test_role: {config['test_role']}")
	return config


def get_git_value(*args: str) -> str:
	try:
		result = subprocess.run(
			["git", *args],
			cwd=PROJECT_ROOT,
			capture_output=True,
			text=True,
			check=True,
		)
		return result.stdout.strip()
	except Exception:
		return ""


def get_git_metadata() -> dict:
	return {
		"git_branch": get_git_value("rev-parse", "--abbrev-ref", "HEAD"),
		"git_commit": get_git_value("rev-parse", "HEAD"),
	}


def load_experiment_rows(path: Path | str) -> list[dict]:
	path = Path(path)
	if not path.exists() or path.stat().st_size == 0:
		return []
	with open(path, "r", encoding="utf-8", newline="") as file:
		reader = DictReader(file, delimiter="\t")
		return [row for row in reader]


def get_latest_keep_reference(path: Path) -> dict | None:
	for row in reversed(load_experiment_rows(path)):
		if row.get("status") == "keep":
			return row
	return None


def metric_improvement(candidate_value: float, reference_value: float, metric_name: str) -> float:
	if metric_name in LESS_IS_BETTER_METRICS:
		return reference_value - candidate_value
	return candidate_value - reference_value


def compute_delta(
	candidate_objective: float,
	comparison_metric: str,
	reference_row: dict | None,
) -> float | None:
	if reference_row is None:
		return None
	reference_objective = reference_row.get("cv_log_loss")
	if reference_objective in (None, ""):
		return None
	return metric_improvement(
		candidate_value=candidate_objective,
		reference_value=float(reference_objective),
		metric_name=comparison_metric,
	)


def print_data_snapshot(data_snapshot: dict):
	print(f"Data fingerprint: {data_snapshot['data_fingerprint']}")
	print(f"Season rows: {data_snapshot['season_row_counts']}")


def print_delta(delta: float | None, comparison_metric: str):
	if delta is None:
		print("Delta: n/a (no prior keep reference)")
		return
	print(f"Delta_{comparison_metric}: {delta:.6f}")


def split_selection_folds(folds: list[tuple[list[str], str]]) -> tuple[list[tuple[list[str], str]], tuple[list[str], str]]:
	if len(folds) < 2:
		raise ValueError("Need at least 2 rolling folds: one for CV objective and one for epoch selection.")
	return folds[:-1], folds[-1]


def describe_training_split(
	objective_folds: list[tuple[list[str], str]],
	epoch_train_seasons: list[str],
	epoch_selection_season: str,
	all_cv_seasons: list[str],
	test_season: str,
	test_role: str,
):
	for fold_idx, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(f"Objective fold {fold_idx}: train on {train_seasons[0]}..{train_seasons[-1]} | validate on {val_season}")
	print(
		f"Epoch selection: train on {epoch_train_seasons[0]}..{epoch_train_seasons[-1]} | validate on {epoch_selection_season}"
	)
	print(f"Fixed test season ({test_role}): train on {all_cv_seasons[0]}..{all_cv_seasons[-1]} | test on {test_season}")


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
		gate_hidden_dim=config["gate_hidden_dim"],
		gate_target_budget=config["gate_target_budget"],
		market_feature_stats=config["market_feature_stats"],
		market_logit_scale=config["market_logit_scale"],
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
	non_null = rolling.drop_nulls(subset=["xg_per90_r10"])
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


def build_loader(dataset: PlayerMatchDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
	generator = None
	if shuffle:
		generator = torch.Generator()
		generator.manual_seed(seed)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


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
		if config.get("aux_context_loss_weight", 0.0) > 0 and config.get("head_type") == "gated_residual":
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
			loss = loss + config["aux_context_loss_weight"] * F.cross_entropy(components["residual_logits"], labels)
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
	"""Compute full evaluation metrics using the same summary shape as the main model."""
	y_true = data["y"]
	preds = np.argmax(probs, axis=1)
	accuracy = float(np.mean(preds == y_true))
	y_onehot = np.eye(3)[y_true]
	brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
	log_loss_value = float(-np.mean(np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0))))
	rps = float(ranked_probability_score(y_true, probs))
	corr_per_class = []
	for class_idx in range(3):
		corr = np.corrcoef(probs[:, class_idx], data["implied"][:, class_idx])[0, 1]
		corr_per_class.append(0.0 if np.isnan(corr) else float(corr))
	profit_metrics = evaluate_profit(
		probs=probs,
		y_true=y_true,
		odds_home=data["odds_home"],
		odds_draw=data["odds_draw"],
		odds_away=data["odds_away"],
	)
	bankroll_metrics = evaluate_bankroll_strategy(
		probs=probs,
		y_true=y_true,
		odds_home=data["odds_home"],
		odds_draw=data["odds_draw"],
		odds_away=data["odds_away"],
		groups=data.get("dates"),
		kelly_fraction=DEFAULT_KELLY_FRACTION,
		initial_bankroll=DEFAULT_BANKROLL,
	)
	return {
		"accuracy": accuracy,
		"brier": brier,
		"rps": rps,
		"log_loss": log_loss_value,
		"corr_with_implied": float(np.mean(corr_per_class)),
		**profit_metrics,
		**bankroll_metrics,
	}


# ---------------------------------------------------------------------------
# Canonical evaluation
# ---------------------------------------------------------------------------

def train_with_early_stopping_split(
	config: dict,
	squad_tensors: dict,
	train_idx: np.ndarray,
	train_data: dict,
	val_idx: np.ndarray,
	val_data: dict,
	seed: int,
	verbose: bool = True,
) -> tuple[nn.Module, int, float]:
	set_seed(seed)
	min_selection_epoch = max(1, int(config.get("min_selection_epoch", 1)))
	model = build_player_model(config).to(DEVICE)
	train_dataset = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
	val_dataset = PlayerMatchDataset(squad_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
	train_loader = build_loader(train_dataset, config["batch_size"], shuffle=True, seed=seed)
	val_loader = build_loader(val_dataset, config["batch_size"], shuffle=False, seed=seed)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=config["max_epochs"],
		eta_min=config["lr"] * 0.01,
	)

	best_val_loss = float("inf")
	best_epoch = 1
	patience_counter = 0
	best_state = None

	for epoch in range(1, config["max_epochs"] + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, config)
		val_loss, _ = evaluate(model, val_loader, DEVICE)
		scheduler.step()
		if verbose:
			print(f"Epoch {epoch:>2}: train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
		if epoch < min_selection_epoch:
			continue
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			patience_counter = 0
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
		else:
			patience_counter += 1
			if patience_counter >= config["patience"]:
				if verbose:
					print(f"Early stopping at epoch {epoch} (best={best_epoch})")
				break

	if best_state is not None:
		model.load_state_dict(best_state)
		model.to(DEVICE)
	return model, best_epoch, float(best_val_loss)


def train_fixed_epochs_split(
	config: dict,
	squad_tensors: dict,
	train_idx: np.ndarray,
	train_data: dict,
	epochs: int,
	seed: int,
	verbose: bool = True,
) -> nn.Module:
	set_seed(seed)
	model = build_player_model(config).to(DEVICE)
	train_dataset = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
	train_loader = build_loader(train_dataset, config["batch_size"], shuffle=True, seed=seed)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=max(1, epochs),
		eta_min=config["lr"] * 0.01,
	)
	for epoch in range(1, max(1, epochs) + 1):
		train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, config)
		scheduler.step()
		if verbose:
			print(f"Epoch {epoch:>2}: train_loss={train_loss:.5f}")
	return model


def evaluate_on_split(
	model: nn.Module,
	squad_tensors: dict,
	split_idx: np.ndarray,
	split_data: dict,
	batch_size: int,
) -> dict:
	dataset = PlayerMatchDataset(squad_tensors, split_data["y"], split_data["implied"], split_data["raw_margin"], split_idx)
	loader = build_loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
	_, probs = evaluate(model, loader, DEVICE)
	return compute_metrics(probs, split_data)


def prepare_experiment_context(config: dict, evaluation_config: dict) -> dict:
	print(f"\nLoading match data from {DEFAULT_PARQUET}")
	df = load_frame(DEFAULT_PARQUET)
	df = prepare_match_data(df)
	print(f"Usable matches: {len(df)}")

	squad_tensors, aligned_df = build_squad_data(df, top_n=config["top_n_players"])
	print(f"Aligned matches: {len(aligned_df)}")

	test_season = resolve_test_season(aligned_df, evaluation_config["test_season"])
	data_snapshot = build_data_snapshot(aligned_df, test_season)
	print_data_snapshot(data_snapshot)

	print(f"\nGenerating {evaluation_config['rolling_cv_n_folds']}-fold rolling CV splits...")
	folds = generate_rolling_cv_folds(
		aligned_df,
		n_folds=evaluation_config["rolling_cv_n_folds"],
		test_season=test_season,
	)
	print(f"Fixed held-out test season: {test_season} ({evaluation_config['test_role']})")

	objective_folds, epoch_fold = split_selection_folds(folds)
	epoch_train_seasons, epoch_selection_season = epoch_fold
	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})
	objective_val_seasons = [val_season for _, val_season in objective_folds]
	describe_training_split(
		objective_folds,
		epoch_train_seasons,
		epoch_selection_season,
		all_cv_seasons,
		test_season,
		evaluation_config["test_role"],
	)

	return {
		"squad_tensors": squad_tensors,
		"aligned_df": aligned_df,
		"test_season": test_season,
		"data_snapshot": data_snapshot,
		"objective_folds": objective_folds,
		"epoch_train_seasons": epoch_train_seasons,
		"epoch_selection_season": epoch_selection_season,
		"all_cv_seasons": all_cv_seasons,
		"objective_val_seasons": objective_val_seasons,
	}


def evaluate_cv_objective(
	config: dict,
	prepared: dict,
	final_train_epochs: int,
	training_seed: int,
) -> tuple[list[dict], Dict[str, float], Dict[str, float]]:
	fold_metrics = []
	fold_baseline_metrics = []
	for fold_idx, (train_seasons, val_season) in enumerate(prepared["objective_folds"], start=1):
		print(
			f"\n--- CV Objective Fold {fold_idx}/{len(prepared['objective_folds'])}: {train_seasons[0]}..{train_seasons[-1]} -> {val_season} ---"
		)
		train_idx, val_idx, train_data, val_data = split_by_seasons(prepared["aligned_df"], train_seasons, [val_season])
		model = train_fixed_epochs_split(
			config,
			prepared["squad_tensors"],
			train_idx,
			train_data,
			epochs=final_train_epochs,
			seed=training_seed + fold_idx,
			verbose=True,
		)
		baseline_metrics = summarize_metrics(evaluate_implied_baseline(val_data))
		metrics = summarize_metrics(
			evaluate_on_split(
				model,
				prepared["squad_tensors"],
				val_idx,
				val_data,
				batch_size=config["batch_size"],
			)
		)
		fold_baseline_metrics.append(baseline_metrics)
		fold_metrics.append({
			"fold_index": fold_idx,
			"train_start_season": train_seasons[0],
			"train_end_season": train_seasons[-1],
			"val_season": val_season,
			"baseline_metrics": baseline_metrics,
			"metrics": metrics,
		})
	mean_fold_metrics = mean_metric([fold["metrics"] for fold in fold_metrics])
	mean_baseline_metrics = mean_metric(fold_baseline_metrics)
	return fold_metrics, mean_fold_metrics, mean_baseline_metrics


def train_player_model(
	config: dict | None = None,
	description: str = "",
	record_run: bool = True,
	prepared: dict | None = None,
) -> dict:
	if config is None:
		config = dict(DEFAULT_CONFIG)
	else:
		config = dict(config)

	evaluation_config = load_evaluation_config()
	comparison_metric = evaluation_config["comparison_metric"]
	training_seed = evaluation_config["training_seed"]
	config["seed"] = training_seed

	print_header(f"TRAIN PLAYER MODEL: {DISPLAY_NAME}")
	print(f"Device: {DEVICE}")
	print(f"Evaluation config: {EVALUATION_CONFIG_PATH}")
	print(f"Config: {json.dumps(config, sort_keys=True)}")

	if prepared is None:
		prepared = prepare_experiment_context(config, evaluation_config)

	git_metadata = get_git_metadata()

	train_idx, val_idx, train_data, val_data = split_by_seasons(
		prepared["aligned_df"],
		prepared["epoch_train_seasons"],
		[prepared["epoch_selection_season"]],
	)
	early_stop_model, best_epoch, best_val_loss = train_with_early_stopping_split(
		config,
		prepared["squad_tensors"],
		train_idx,
		train_data,
		val_idx,
		val_data,
		seed=training_seed,
		verbose=True,
	)
	print(f"Early stopping best epoch: {best_epoch} (val_loss={best_val_loss:.5f})")
	final_train_epochs = max(1, min(config["max_epochs"], int(best_epoch)))
	print(f"Final retrain epochs: {final_train_epochs} (mode=best)")

	cv_fold_metrics, cv_metrics, cv_baseline_metrics = evaluate_cv_objective(
		config=config,
		prepared=prepared,
		final_train_epochs=final_train_epochs,
		training_seed=training_seed,
	)
	objective_value = float(cv_metrics[comparison_metric])
	print(f"\nCV objective ({comparison_metric}): {objective_value:.5f}")

	print("\n--- Early-stop Model Performance on Epoch-selection Season ---")
	validation_baseline_metrics = evaluate_implied_baseline(val_data)
	validation_metrics = evaluate_on_split(
		early_stop_model,
		prepared["squad_tensors"],
		val_idx,
		val_data,
		batch_size=config["batch_size"],
	)

	all_train_idx, test_idx, all_train_data, test_data = split_by_seasons(
		prepared["aligned_df"],
		prepared["all_cv_seasons"],
		[prepared["test_season"]],
	)
	test_baseline_metrics = evaluate_implied_baseline(test_data)

	print("\n--- Training Final Model ---")
	model = train_fixed_epochs_split(
		config,
		prepared["squad_tensors"],
		all_train_idx,
		all_train_data,
		epochs=final_train_epochs,
		seed=training_seed + 10_000,
		verbose=True,
	)

	print(f"\n--- Model Performance on Fixed Test Set ({evaluation_config['test_role']}) ---")
	test_metrics = evaluate_on_split(
		model,
		prepared["squad_tensors"],
		test_idx,
		test_data,
		batch_size=config["batch_size"],
	)

	reference_row = get_latest_keep_reference(path=EXPERIMENT_LOG_PATH)
	delta = compute_delta(
		candidate_objective=objective_value,
		comparison_metric=comparison_metric,
		reference_row=reference_row,
	)
	print_delta(delta, comparison_metric)

	run_record = {
		"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
		"display_name": DISPLAY_NAME,
		"comparison_metric": comparison_metric,
		"objective_name": f"cv_mean_{comparison_metric}",
		"objective_value": objective_value,
		"model_name": MODEL_NAME,
		"model_config": config,
		"training_entry_point": "training/train_player_model.py",
		"evaluation_config_source": str(EVALUATION_CONFIG_PATH.relative_to(PROJECT_ROOT)),
		"objective_fold_count": len(prepared["objective_folds"]),
		"objective_val_seasons": prepared["objective_val_seasons"],
		"objective_metrics": cv_metrics,
		"objective_baseline_metrics": cv_baseline_metrics,
		"objective_fold_metrics": cv_fold_metrics,
		"epoch_selection_season": prepared["epoch_selection_season"],
		"held_out_test_season": prepared["test_season"],
		"test_role": evaluation_config["test_role"],
		"best_epoch": best_epoch,
		"best_val_loss": float(best_val_loss),
		"data_snapshot": prepared["data_snapshot"],
		"delta": delta,
		"val_metrics": summarize_metrics(validation_metrics),
		"val_baseline_metrics": summarize_metrics(validation_baseline_metrics),
		"test_metrics": summarize_metrics(test_metrics),
		"test_baseline_metrics": summarize_metrics(test_baseline_metrics),
		**git_metadata,
	}

	if record_run:
		write_json(
			LATEST_SET_TRANSFORMER_METRICS_PATH,
			{
				"schema_version": 1,
				"description": "Latest evaluated set-transformer candidate. Runtime-generated; compare with prior kept rows in artifacts/experiment_metrics/set_transformer_runs.tsv.",
				"model": run_record,
			},
		)
		append_tsv_row(
			EXPERIMENT_LOG_PATH,
			{
				"recorded_at_utc": run_record["recorded_at_utc"],
				"git_commit": git_metadata["git_commit"][:7],
				"git_branch": git_metadata["git_branch"],
				"cv_log_loss": f"{run_record['objective_value']:.6f}",
				"delta": f"{delta:.6f}" if delta is not None else "",
				"best_epoch": best_epoch,
				"status": "",
				"description": description,
				"cv_rps": f"{cv_metrics.get('rps', 0):.6f}",
				"val_log_loss": f"{run_record['val_metrics'].get('log_loss', 0):.6f}",
				"test_log_loss": f"{run_record['test_metrics'].get('log_loss', 0):.6f}",
				"cv_metrics_json": run_record["objective_metrics"],
				"test_metrics_json": run_record["test_metrics"],
			},
		)

	print_header("DONE")
	print(f"CV objective ({comparison_metric}): {run_record['objective_value']:.5f}")
	print(f"Best validation loss: {best_val_loss:.5f}")
	print(f"Validation metrics: {run_record['val_metrics']}")
	print(f"Test metrics ({evaluation_config['test_role']}): {run_record['test_metrics']}")
	print(f"Experiment log: {EXPERIMENT_LOG_PATH}")
	return run_record


def main():
	parser = argparse.ArgumentParser(description="Train player-level set transformer model")
	parser.add_argument("--description", type=str, default="", help="Short text description of what this experiment tried")
	parser.add_argument("--encoder", choices=["set_transformer"], default="set_transformer")
	parser.add_argument("--head-type", choices=["mlp", "gated_residual"], default="mlp")
	parser.add_argument("--no-implied", action="store_true", help="Don't use implied probabilities as input")
	parser.add_argument("--hidden-dim", type=int)
	parser.add_argument("--team-output-dim", type=int)
	parser.add_argument("--num-heads", type=int)
	parser.add_argument("--num-sab-layers", type=int)
	parser.add_argument("--dropout", type=float)
	parser.add_argument("--lr", type=float)
	parser.add_argument("--weight-decay", type=float)
	parser.add_argument("--gate-hidden-dim", type=int)
	parser.add_argument("--gate-target-budget", type=float)
	parser.add_argument("--market-feature-stats", type=int, choices=[3, 4, 5])
	parser.add_argument("--market-logit-scale", type=float)
	parser.add_argument("--learn-market-class-scale", action="store_true")
	parser.add_argument("--aux-context-loss-weight", type=float)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--max-epochs", type=int)
	parser.add_argument("--patience", type=int)
	parser.add_argument("--min-selection-epoch", type=int)
	parser.add_argument("--top-n-players", type=int)
	parser.add_argument("--no-ledger", action="store_true", help="Run the canonical evaluation without appending a ledger row")
	args = parser.parse_args()

	config = dict(DEFAULT_CONFIG)
	config["encoder_type"] = args.encoder
	config["head_type"] = args.head_type
	if args.no_implied:
		config["use_implied"] = False
	if args.hidden_dim is not None:
		config["hidden_dim"] = args.hidden_dim
	if args.team_output_dim is not None:
		config["team_output_dim"] = args.team_output_dim
	if args.num_heads is not None:
		config["num_heads"] = args.num_heads
	if args.num_sab_layers is not None:
		config["num_sab_layers"] = args.num_sab_layers
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
	if args.market_feature_stats is not None:
		config["market_feature_stats"] = args.market_feature_stats
	if args.market_logit_scale is not None:
		config["market_logit_scale"] = args.market_logit_scale
	if args.learn_market_class_scale:
		config["learn_market_class_scale"] = True
	if args.aux_context_loss_weight is not None:
		config["aux_context_loss_weight"] = args.aux_context_loss_weight
	if args.batch_size is not None:
		config["batch_size"] = args.batch_size
	if args.max_epochs is not None:
		config["max_epochs"] = args.max_epochs
	if args.patience is not None:
		config["patience"] = args.patience
	if args.min_selection_epoch is not None:
		config["min_selection_epoch"] = args.min_selection_epoch
	if args.top_n_players is not None:
		config["top_n_players"] = args.top_n_players

	train_player_model(config=config, description=args.description, record_run=not args.no_ledger)


if __name__ == "__main__":
	main()
