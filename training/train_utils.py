"""
Training utilities and data preparation functions.
"""

import copy
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from training.models.neural_net import TrainConfig, residual_market_loss_corr
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
) -> DataLoader:
	"""Convert data dictionary to PyTorch DataLoader."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	tensor_x = torch.tensor(data["X"], dtype=torch.float32)
	tensor_implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1)
	tensor_y = torch.tensor(data["y"], dtype=torch.float32).unsqueeze(1)
	ds = TensorDataset(tensor_x, tensor_implied, tensor_y)
	kwargs = {"num_workers": 0, "pin_memory": True} if device.type == "cuda" else {}
	return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


# ============================================================================
# TRAINING
# ============================================================================


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

	model = MLP(
		config.input_dim, config.hidden_layers, config.dropout, config.norm
	).to(device)
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.lr, weight_decay=config.weight_decay
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=5
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

		scheduler.step(avg_val_loss)
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
