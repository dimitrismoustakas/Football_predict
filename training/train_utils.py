"""
Training utilities and data preparation functions.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from training.models.neural_net import MLP, TrainConfig, residual_market_loss_corr
from training.evaluation.metrics import accuracy_score, brier_score_loss, log_loss


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================


def load_frame(parquet_path: Path) -> pl.DataFrame:
	"""Load Parquet file into Polars DataFrame."""
	return pl.scan_parquet(str(parquet_path)).collect()


def select_feature_columns(df: pl.DataFrame) -> List[str]:
	"""Select feature columns based on naming convention."""
	cols = df.columns
	feat_cols = [
		c
		for c in cols
		if c.startswith("ovr__")
		and "__r5" in c
		and "__sum__" not in c
		and not c.startswith("ovr__games__")
		and (c.endswith("__h") or c.endswith("__a"))
	]
	# Add Elo features if present
	for ec in ["elo_diff", "elo_sum", "elo_mean"]:
		if ec in cols:
			feat_cols.append(ec)
	return sorted(feat_cols)


def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
	"""Filter to matches where both teams have at least 5 prior games."""
	need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
	missing = [c for c in need_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for history filter: {missing}")
	return df.filter(
		(pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5)
	)


def train_test_season_splits(df: pl.DataFrame) -> Tuple[List[str], str, str]:
	"""Split seasons into train/validation/test sets."""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	if len(seasons) < 3:
		raise ValueError("Need at least 3 seasons to create the requested splits.")
	current = seasons[-1]
	previous = seasons[-2]
	train = seasons[:-2]
	return train, previous, current


def train_test_season_splits_arch_search(df: pl.DataFrame) -> Tuple[List[str], str, str]:
	"""
	Split seasons for architecture search.
	Uses last season of 'normal train' as validation.
	Returns: (train_seasons, val_season, test_season)
	- train: seasons[:-3] (all except last 3)
	- val: seasons[-3] (third from last, i.e., last of normal train)
	- test: seasons[-2] (second from last, held out)
	- seasons[-1] is current/future season, not used
	"""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	if len(seasons) < 4:
		raise ValueError("Need at least 4 seasons for architecture search splits.")
	train = seasons[:-3]
	val = seasons[-3]
	test = seasons[-2]
	return train, val, test


def generate_rolling_cv_folds(
	df: pl.DataFrame, n_folds: int = 3
) -> List[Tuple[List[str], str]]:
	"""
	Generate rolling-origin cross-validation folds for time-series data.
	
	Uses expanding window: train on seasons up to Y, validate on Y+1.
	Reserves last season as held-out test (not included in folds).
	
	Example with 10 seasons [2014..2023] and n_folds=3:
		Fold 0: train=[2014..2020], val=2021
		Fold 1: train=[2014..2021], val=2022
		Fold 2: train=[2014..2022], val=2023
		Test (held out): 2024 (current season, not in folds)
	
	Returns: List of (train_seasons, val_season) tuples
	"""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	
	# Reserve last season as test (current/future), second-to-last for final val
	# CV folds use seasons before that
	if len(seasons) < n_folds + 2:
		raise ValueError(
			f"Need at least {n_folds + 2} seasons for {n_folds}-fold rolling CV. "
			f"Got {len(seasons)} seasons."
		)
	
	# Available seasons for CV (exclude current season which is last)
	available = seasons[:-1]
	
	folds = []
	for i in range(n_folds):
		# Val season: work backwards from second-to-last available
		val_idx = len(available) - n_folds + i
		val_season = available[val_idx]
		train_seasons = available[:val_idx]
		folds.append((train_seasons, val_season))
	
	return folds


def get_test_season(df: pl.DataFrame) -> str:
	"""Get the held-out test season (second to last, as last is current/future)."""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	if len(seasons) < 2:
		raise ValueError("Need at least 2 seasons to have a test season.")
	return seasons[-2]


def build_hidden_layers(
	base_width: int, n_layers: int, shape: str
) -> List[int]:
	"""
	Build hidden layer sizes based on parameterized shape.
	
	Shapes:
		- constant: [base] * n_layers
		- pyramid: [base, base*2, base*4, ...] (growing, capped at 512)
		- inverted: [base*2^(n-1), ..., base*2, base] (shrinking)
		- diamond: expand then contract (for n_layers >= 3)
	
	Args:
		base_width: Base width (e.g., 64, 128, 256)
		n_layers: Number of hidden layers (2-5)
		shape: One of 'constant', 'pyramid', 'inverted', 'diamond'
	
	Returns:
		List of hidden layer sizes
	"""
	if shape == "constant":
		return [base_width] * n_layers
	
	elif shape == "pyramid":
		# Growing: base -> base*2 -> base*4, capped at 512
		layers = []
		width = base_width
		for _ in range(n_layers):
			layers.append(min(width, 512))
			width *= 2
		return layers
	
	elif shape == "inverted":
		# Shrinking: start wide, end narrow
		layers = []
		max_mult = 2 ** (n_layers - 1)
		width = min(base_width * max_mult, 512)
		for i in range(n_layers):
			layers.append(width)
			width = max(base_width, width // 2)
		return layers
	
	elif shape == "diamond":
		# Expand then contract (makes sense for n_layers >= 3)
		if n_layers < 3:
			return [base_width] * n_layers
		
		mid = n_layers // 2
		layers = []
		
		# Expanding phase
		width = base_width
		for i in range(mid):
			layers.append(width)
			width = min(width * 2, 512)
		
		# Peak
		layers.append(width)
		
		# Contracting phase
		remaining = n_layers - mid - 1
		for i in range(remaining):
			width = max(base_width, width // 2)
			layers.append(width)
		
		return layers
	
	else:
		raise ValueError(f"Unknown shape: {shape}. Use 'constant', 'pyramid', 'inverted', or 'diamond'.")


def add_targets_and_implied(df: pl.DataFrame) -> pl.DataFrame:
	"""Add match result and implied probability columns."""
	if "Over" not in df.columns:
		raise ValueError("'Over' column not found; ensure you used build_match_level().")

	need_odds = ["odds_over", "odds_under"]
	missing = [c for c in need_odds if c not in df.columns]
	if missing:
		raise ValueError(f"Missing odds columns for implied probabilities: {missing}")

	df = df.with_columns(
		pl.when(pl.col("home_goals") > pl.col("away_goals"))
		.then(pl.lit("H"))
		.when(pl.col("home_goals") < pl.col("away_goals"))
		.then(pl.lit("A"))
		.otherwise(pl.lit("D"))
		.alias("match_result")
	)

	implied_over = 1 / pl.col("odds_over")
	implied_under = 1 / pl.col("odds_under")
	norm = implied_over + implied_under

	return df.with_columns((implied_over / norm).alias("implied_over_prob"))


def prepare_data(
	df: pl.DataFrame,
	feature_cols: List[str],
	season_list: List[str],
	scaler: StandardScaler = None,
	fit_scaler: bool = False,
) -> Dict[str, np.ndarray]:
	"""Selects data, scales features, and returns a dictionary of arrays."""
	part = df.filter(pl.col("season").cast(pl.Utf8).is_in(list(season_list)))

	req_cols = list(
		set(feature_cols)
		| {"Over", "implied_over_prob", "odds_over", "odds_under", "date"}
	)

	initial_count = len(part)
	part = part.filter(
		(pl.col("odds_over") > 1.0)
		& (pl.col("odds_under") > 1.0)
		& pl.col("implied_over_prob").is_finite()
	).drop_nulls(subset=req_cols)
	final_count = len(part)

	if initial_count != final_count:
		print(
			f"Dropped {initial_count - final_count} rows due to invalid odds/missing data in {season_list}"
		)

	X = part.select(feature_cols).to_pandas().values

	if fit_scaler:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
	elif scaler is not None:
		X = scaler.transform(X)

	return {
		"X": X,
		"y": part.select("Over").to_pandas().values.flatten().astype(int),
		"implied": part.select("implied_over_prob").to_pandas().values.flatten(),
		"odds_over": part.select("odds_over").to_pandas().values.flatten(),
		"odds_under": part.select("odds_under").to_pandas().values.flatten(),
		"dates": part.select("date").to_pandas().values.flatten(),
		"scaler": scaler,
	}


def to_loader(
	data: Dict[str, np.ndarray], 
	batch_size: int, 
	shuffle: bool = True,
	device: torch.device = None,
	num_workers: int = 0,
	pin_memory: bool = None,
) -> DataLoader:
	"""
	Convert data dictionary to PyTorch DataLoader.
	
	Args:
		data: Dict with 'X', 'y', 'implied' arrays
		batch_size: Batch size
		shuffle: Whether to shuffle data
		device: Target device (for pin_memory default)
		num_workers: Number of worker processes for data loading
		pin_memory: Whether to pin memory (defaults to True for CUDA)
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if pin_memory is None:
		pin_memory = device.type == "cuda"
		
	tensor_x = torch.tensor(data["X"], dtype=torch.float32)
	tensor_implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1)
	tensor_y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(1)
	ds = TensorDataset(tensor_x, tensor_implied, tensor_y)
	return DataLoader(
		ds, 
		batch_size=batch_size, 
		shuffle=shuffle, 
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


# ============================================================================
# CV FOLD DATA CACHING
# ============================================================================

# Optimal settings for data loading (benchmark tested)
OPTIMAL_NUM_WORKERS = 4
PIN_MEMORY = torch.cuda.is_available()


def precompute_fold_data(
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
) -> List[Dict[str, Any]]:
	"""
	Precompute scaled train/val data for each CV fold.
	
	Called ONCE before Optuna search to avoid refitting scalers on every trial.
	Each trial only needs to wrap DataLoaders with the appropriate batch size.
	
	Returns:
		List of dicts, one per fold, each containing:
		- X_train, y_train, implied_train: scaled training arrays
		- X_val, y_val, implied_val, odds_over_val, odds_under_val, dates_val
		- scaler: fitted StandardScaler for this fold
		- train_seasons, val_season: for reference
	"""
	fold_data = []
	
	for fold_idx, (train_seasons, val_season) in enumerate(folds):
		print(f"  Fold {fold_idx}: train={train_seasons[0]}..{train_seasons[-1]}, val={val_season}")
		
		# Prepare training data (fits scaler)
		data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
		
		# Prepare validation data (uses fitted scaler)
		data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
		
		fold_data.append({
			"X_train": data_train["X"],
			"y_train": data_train["y"],
			"implied_train": data_train["implied"],
			"X_val": data_val["X"],
			"y_val": data_val["y"],
			"implied_val": data_val["implied"],
			"odds_over_val": data_val["odds_over"],
			"odds_under_val": data_val["odds_under"],
			"dates_val": data_val["dates"],
			"scaler": data_train["scaler"],
			"train_seasons": train_seasons,
			"val_season": val_season,
		})
	
	return fold_data


def fold_data_to_loaders(
	fold: Dict[str, Any],
	batch_size: int,
	device: torch.device = None,
) -> Tuple[DataLoader, DataLoader]:
	"""
	Convert precomputed fold data to DataLoaders with specified batch size.
	
	Fast because data is already scaled - just wraps in tensors.
	Uses optimized num_workers and pin_memory settings.
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	train_data = {
		"X": fold["X_train"],
		"y": fold["y_train"],
		"implied": fold["implied_train"],
	}
	train_loader = to_loader(
		train_data, batch_size, shuffle=True, device=device,
		num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY
	)
	
	val_data = {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
	}
	val_loader = to_loader(
		val_data, batch_size, shuffle=False, device=device,
		num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY
	)
	
	return train_loader, val_loader


def get_val_data_dict(fold: Dict[str, Any]) -> Dict[str, np.ndarray]:
	"""Extract validation data in format expected by evaluate_model."""
	return {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
		"odds_over": fold["odds_over_val"],
		"odds_under": fold["odds_under_val"],
		"dates": fold["dates_val"],
	}


# ============================================================================
# TRAINING
# ============================================================================


def create_scheduler(
	optimizer: torch.optim.Optimizer,
	scheduler_type: str,
	epochs: int = 100,
	steps_per_epoch: int = 1,
	lr: float = 1e-3,
) -> torch.optim.lr_scheduler.LRScheduler:
	"""
	Factory function for learning rate schedulers.
	
	Args:
		optimizer: PyTorch optimizer
		scheduler_type: One of 'plateau', 'cosine', 'onecycle'
		epochs: Total training epochs (for cosine/onecycle)
		steps_per_epoch: Steps per epoch (for onecycle)
		lr: Base learning rate (for onecycle max_lr calculation)
	"""
	if scheduler_type == "plateau":
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode="min", factor=0.5, patience=5
		)
	elif scheduler_type == "cosine":
		return torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=epochs, eta_min=lr * 0.01
		)
	elif scheduler_type == "onecycle":
		return torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=lr * 10,
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
		)
	else:
		raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class EarlyStopping:
	"""Early stopping handler with model state tracking."""

	def __init__(self, patience: int = 7, min_delta: float = 0.0):
		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.best_loss = None
		self.early_stop = False
		self.best_model_state = None

	def __call__(self, val_loss: float, model: nn.Module):
		if self.best_loss is None:
			self.best_loss = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
		elif val_loss > self.best_loss - self.min_delta:
			self.counter += 1
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_loss = val_loss
			self.best_model_state = copy.deepcopy(model.state_dict())
			self.counter = 0

	def load_best_weights(self, model: nn.Module):
		if self.best_model_state:
			model.load_state_dict(self.best_model_state)


def train_model(
	config: TrainConfig,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device = None,
	trial = None,
	verbose: bool = True,
) -> Tuple:
	"""
	Train model with early stopping.
	Returns (model, history, best_val_loss)
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	from .models.neural_net import MLP

	activation = getattr(config, "activation", "relu")
	model = MLP(
		config.input_dim, config.hidden_layers, config.dropout, config.norm, activation
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.lr, weight_decay=config.weight_decay
	)
	
	scheduler_type = getattr(config, "scheduler_type", "plateau")
	scheduler = create_scheduler(
		optimizer,
		scheduler_type,
		epochs=config.epochs,
		steps_per_epoch=len(train_loader),
		lr=config.lr,
	)
	early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4)

	history = {"train_loss": [], "val_loss": []}

	for epoch in range(1, config.epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		for batch_x, batch_implied, batch_y in train_loader:
			batch_x = batch_x.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)

			optimizer.zero_grad()
			residual_logits = model(batch_x)
			loss = residual_market_loss_corr(
				residual_logits,
				batch_implied,
				batch_y,
				lambda_repulsion=config.lambda_repulsion,
				lambda_corr=config.lambda_corr,
			)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(batch_x)
			
			# Step OneCycleLR after each batch
			if scheduler_type == "onecycle":
				scheduler.step()

		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)

		# Validation phase
		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for bx, bi, by in val_loader:
				bx, bi, by = bx.to(device), bi.to(device), by.to(device)
				residual_logits = model(bx)
				loss = residual_market_loss_corr(
					residual_logits,
					bi,
					by,
					lambda_repulsion=config.lambda_repulsion,
					lambda_corr=config.lambda_corr,
				)
				val_loss += loss.item() * len(bx)

		avg_val_loss = val_loss / len(val_loader.dataset)
		history["val_loss"].append(avg_val_loss)

		# Step schedulers that operate per-epoch
		if scheduler_type == "plateau":
			scheduler.step(avg_val_loss)
		elif scheduler_type == "cosine":
			scheduler.step()
		# onecycle is stepped per batch above
		
		early_stopping(avg_val_loss, model)

		# Log to MLflow if in an active run
		if mlflow.active_run():
			mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
			mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

		if verbose and (epoch % 10 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}"
			)

		# Optuna pruning
		if trial is not None:
			trial.report(avg_val_loss, epoch)
			if trial.should_prune():
				import optuna
				raise optuna.TrialPruned()

		if early_stopping.early_stop:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break

	early_stopping.load_best_weights(model)
	return model, history, early_stopping.best_loss

def evaluate_implied_baseline(data: Dict[str, np.ndarray]) -> Dict:
	"""Evaluate market implied probabilities as baseline predictions."""
	implied_probs = data["implied"]
	y_true = data["y"]
	
	preds = (implied_probs >= 0.5).astype(int)
	acc = accuracy_score(y_true, preds)
	brier = brier_score_loss(y_true, implied_probs)
	ll = log_loss(y_true, np.c_[1 - implied_probs, implied_probs], labels=[0, 1])
	
	return {
		"accuracy": float(acc),
		"brier": float(brier),
		"log_loss": float(ll),
	}


def load_existing_model(
	config_path: Path,
	model_path: Path,
	device: torch.device,
) -> Tuple[nn.Module, Dict]:
	"""
	Load existing model from disk if it exists.
	
	Returns (model, config_dict) or (None, None) if not found.
	"""
	if not config_path.exists() or not model_path.exists():
		return None, None
	
	import json
	with open(config_path) as f:
		config_dict = json.load(f)
	
	model = MLP(
		input_dim=config_dict["input_dim"],
		hidden_layers=config_dict["hidden_layers"],
		dropout=config_dict["dropout"],
		norm=config_dict["norm"],
		activation=config_dict["activation"],
	).to(device)
	
	model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
	model.eval()
	
	return model, config_dict