"""
Training utilities and data preparation functions for match-result prediction.

Categorical features:
- `league_idx`
- `home_promoted`
- `away_promoted`

Continuous features are scaled with `StandardScaler`.
"""

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from training.evaluation.metrics import accuracy_score, log_loss, ranked_probability_score
from training.models.neural_net import GatedResidualModel, TrainConfig, gated_loss

CAT_COLS = ["league_idx", "home_promoted", "away_promoted"]
OPTIMAL_NUM_WORKERS = 0 if sys.platform == "win32" else 4
PIN_MEMORY = torch.cuda.is_available()


def load_frame(parquet_path: Path) -> pl.DataFrame:
	"""Load a parquet file into a Polars frame."""

	return pl.scan_parquet(str(parquet_path)).collect()


def select_feature_columns(df: pl.DataFrame) -> List[str]:
	"""Select the fixed result-model feature subset."""

	cols = set(df.columns)
	feat_cols = [
		"away__deep_against__r5__a",
		"away__deep_against__sum__r5__a",
		"away__deep_for__r5__a",
		"away__deep_for__sum__r5__a",
		"away__npxg_against__r5__a",
		"away__npxg_against__sum__r5__a",
		"away__npxg_for__r5__a",
		"away__npxg_for__sum__r5__a",
		"away__ppda_against__r5__a",
		"away__ppda_against__sum__r5__a",
		"away__ppda_for__r5__a",
		"away__ppda_for__sum__r5__a",
		"away__xg_against__r5__a",
		"away__xg_against__sum__r5__a",
		"away__xg_for__r5__a",
		"away__xg_for__sum__r5__a",
		"away__xgd__r5__a",
		"away__xgd__sum__r5__a",
		"away_minutes_hhi_r15",
		"away_unique_players_r15",
		"away_unique_players_r5_sum",
		"away_xa_hhi_r15",
		"away_xg_hhi_r15",
		"days_since_last_match__a",
		"days_since_last_match__h",
		"elo_diff",
		"elo_diff_r5__a",
		"elo_diff_r5__h",
		"elo_mean",
		"elo_sum",
		"games_last_15_days__a",
		"games_last_15_days__h",
		"home__deep_against__r5__h",
		"home__deep_against__sum__r5__h",
		"home__deep_for__r5__h",
		"home__deep_for__sum__r5__h",
		"home__npxg_against__r5__h",
		"home__npxg_against__sum__r5__h",
		"home__npxg_for__r5__h",
		"home__npxg_for__sum__r5__h",
		"home__ppda_against__r5__h",
		"home__ppda_against__sum__r5__h",
		"home__ppda_for__r5__h",
		"home__ppda_for__sum__r5__h",
		"home__xg_against__r5__h",
		"home__xg_against__sum__r5__h",
		"home__xg_for__r5__h",
		"home__xg_for__sum__r5__h",
		"home__xgd__r5__h",
		"home__xgd__sum__r5__h",
		"home_minutes_hhi_r15",
		"home_unique_players_r15",
		"home_unique_players_r5_sum",
		"home_xa_hhi_r15",
		"home_xg_hhi_r15",
		"opponent_elo_r5__a",
		"opponent_elo_r5__h",
		"opponent_elo_std_r5__a",
		"opponent_elo_std_r5__h",
		"ovr__adj__ga__r10__a",
		"ovr__adj__ga__r10__h",
		"ovr__adj__gf__r10__a",
		"ovr__adj__gf__r10__h",
		"ovr__adj__npxg_against__r10__a",
		"ovr__adj__npxg_against__r10__h",
		"ovr__adj__npxg_for__r10__a",
		"ovr__adj__npxg_for__r10__h",
		"ovr__adj__shots_against__r10__a",
		"ovr__adj__shots_against__r10__h",
		"ovr__adj__shots_for__r10__a",
		"ovr__adj__shots_for__r10__h",
		"ovr__adj__sot_against__r10__a",
		"ovr__adj__sot_against__r10__h",
		"ovr__adj__sot_for__r10__a",
		"ovr__adj__sot_for__r10__h",
		"ovr__adj__xg_against__r10__a",
		"ovr__adj__xg_against__r10__h",
		"ovr__adj__xg_for__r10__a",
		"ovr__adj__xg_for__r10__h",
		"ovr__draw__r5__a",
		"ovr__draw__r5__h",
		"ovr__draw__sum__r5__a",
		"ovr__draw__sum__r5__h",
		"ovr__loss__r5__a",
		"ovr__loss__r5__h",
		"ovr__loss__sum__r5__a",
		"ovr__loss__sum__r5__h",
		"ovr__npxg_against__r5__a",
		"ovr__npxg_against__r5__h",
		"ovr__npxg_against__sum__r5__a",
		"ovr__npxg_against__sum__r5__h",
		"ovr__npxg_for__r5__a",
		"ovr__npxg_for__r5__h",
		"ovr__npxg_for__sum__r5__a",
		"ovr__npxg_for__sum__r5__h",
		"ovr__points__r5__a",
		"ovr__points__r5__h",
		"ovr__points__sum__r5__a",
		"ovr__points__sum__r5__h",
		"ovr__win__r5__a",
		"ovr__win__r5__h",
		"ovr__win__sum__r5__a",
		"ovr__win__sum__r5__h",
		"ovr__xg_against__r5__a",
		"ovr__xg_against__r5__h",
		"ovr__xg_against__sum__r5__a",
		"ovr__xg_against__sum__r5__h",
		"ovr__xg_for__r5__a",
		"ovr__xg_for__r5__h",
		"ovr__xg_for__sum__r5__a",
		"ovr__xg_for__sum__r5__h",
		"ovr__xgd__r5__a",
		"ovr__xgd__r5__h",
		"ovr__xgd__sum__r5__a",
		"ovr__xgd__sum__r5__h",
		"season_progress",
	]
	return [col for col in feat_cols if col in cols]


def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
	"""Filter to matches where both teams have at least five prior games."""

	need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
	missing = [col for col in need_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for history filter: {missing}")
	return df.filter((pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5))


def generate_rolling_cv_folds(df: pl.DataFrame, n_folds: int = 3) -> List[Tuple[List[str], str]]:
	"""Generate expanding-window validation folds and reserve the latest season for test."""

	seasons = (
		df.select(pl.col("season").cast(pl.Utf8)).unique().sort(by="season").to_series().to_list()
	)
	if len(seasons) < n_folds + 2:
		raise ValueError(f"Need at least {n_folds + 2} seasons for {n_folds}-fold rolling CV. Got {len(seasons)}.")

	available = seasons[:-1]
	folds = []
	for fold_idx in range(n_folds):
		val_idx = len(available) - n_folds + fold_idx
		val_season = available[val_idx]
		train_seasons = available[:val_idx]
		folds.append((train_seasons, val_season))
	return folds


def get_test_season(df: pl.DataFrame) -> str:
	"""Return the latest held-out test season."""

	seasons = (
		df.select(pl.col("season").cast(pl.Utf8)).unique().sort(by="season").to_series().to_list()
	)
	if len(seasons) < 2:
		raise ValueError("Need at least 2 seasons to have a test season.")
	return seasons[-1]


def build_hidden_layers(base_width: int, n_layers: int, shape: str) -> List[int]:
	"""Helper for search scripts that build layer stacks by shape."""

	if shape == "constant":
		return [base_width] * n_layers
	if shape == "pyramid":
		layers = []
		width = base_width
		for _ in range(n_layers):
			layers.append(min(width, 512))
			width *= 2
		return layers
	if shape == "inverted":
		layers = []
		width = min(base_width * (2 ** (n_layers - 1)), 512)
		for _ in range(n_layers):
			layers.append(width)
			width = max(base_width, width // 2)
		return layers
	if shape == "diamond":
		if n_layers < 3:
			return [base_width] * n_layers
		mid = n_layers // 2
		layers = []
		width = base_width
		for _ in range(mid):
			layers.append(width)
			width = min(width * 2, 512)
		layers.append(width)
		for _ in range(n_layers - mid - 1):
			width = max(base_width, width // 2)
			layers.append(width)
		return layers
	raise ValueError(f"Unknown shape: {shape}")


def _normalize_implied(odds_cols: List[str], prefix: str) -> Tuple[Dict[str, pl.Expr], pl.Expr]:
	"""Compute normalized implied probabilities."""

	inv_odds = [1 / pl.col(col) for col in odds_cols]
	norm = inv_odds[0]
	for expr in inv_odds[1:]:
		norm = norm + expr

	prob_cols = {}
	for col, expr in zip(odds_cols, inv_odds):
		suffix = col.replace("odds_", "")
		prob_cols[f"{prefix}_{suffix}"] = expr / norm
	return prob_cols, norm


def add_targets_and_implied(df: pl.DataFrame) -> pl.DataFrame:
	"""Add result labels and normalized implied probabilities."""

	if "odds_h" in df.columns and "odds_home" not in df.columns:
		df = df.with_columns([
			pl.col("odds_h").alias("odds_home"),
			pl.col("odds_d").alias("odds_draw"),
			pl.col("odds_a").alias("odds_away"),
		])

	need_odds = ["odds_home", "odds_draw", "odds_away"]
	missing = [col for col in need_odds if col not in df.columns]
	if missing:
		raise ValueError(f"Missing odds columns for result prediction: {missing}")

	prob_cols, norm = _normalize_implied(["odds_home", "odds_draw", "odds_away"], "implied")
	return df.with_columns([
		pl.when(pl.col("home_goals") > pl.col("away_goals"))
		.then(pl.lit(0))
		.when(pl.col("home_goals") == pl.col("away_goals"))
		.then(pl.lit(1))
		.otherwise(pl.lit(2))
		.alias("result_label"),
		prob_cols["implied_home"].alias("implied_home"),
		prob_cols["implied_draw"].alias("implied_draw"),
		prob_cols["implied_away"].alias("implied_away"),
		norm.alias("raw_margin"),
	])


def extract_categorical_features(df: pl.DataFrame) -> np.ndarray:
	"""Extract categorical features as a numpy array."""

	league_idx = df.select("league_idx").to_numpy().flatten().astype(np.int64)
	home_promoted = df.select("home_promoted").to_numpy().flatten().astype(np.int64)
	away_promoted = df.select("away_promoted").to_numpy().flatten().astype(np.int64)
	return np.stack([league_idx, home_promoted, away_promoted], axis=1)


def get_num_leagues(df: pl.DataFrame) -> int:
	"""Get the number of distinct league ids."""

	return df.select("league_idx").unique().height


def _prepare_base(
	df: pl.DataFrame,
	feature_cols: List[str],
	season_list: List[str],
	scaler: StandardScaler,
	fit_scaler: bool,
	req_cols: List[str],
	filter_expr: pl.Expr,
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
	"""Filter rows, scale features, and extract categorical inputs."""

	part = df.filter(pl.col("season").cast(pl.Utf8).is_in(list(season_list)))
	initial_count = len(part)
	filtered = part.filter(filter_expr)
	invalid_count = initial_count - len(filtered)
	part = filtered.drop_nulls(subset=req_cols)
	missing_required_count = len(filtered) - len(part)

	if invalid_count:
		print(f"Dropped {invalid_count} rows due to invalid odds/non-finite implied values in {season_list}")
	if missing_required_count:
		null_counts_row = filtered.select([pl.col(col).is_null().sum().alias(col) for col in req_cols]).to_dicts()[0]
		top_missing = [
			(col, count)
			for col, count in sorted(null_counts_row.items(), key=lambda item: (-item[1], item[0]))
			if count > 0
		][:8]
		print(f"Dropped {missing_required_count} rows due to missing required features in {season_list}: {top_missing}")

	feature_frame = part.select([pl.col(col).cast(pl.Float64).alias(col) for col in feature_cols])
	feature_missing_cells = int(feature_frame.select([pl.col(col).is_null().sum().alias(col) for col in feature_cols]).sum_horizontal().item())
	feature_missing_rows = feature_frame.filter(pl.any_horizontal([pl.col(col).is_null() for col in feature_cols])).height
	if feature_missing_rows:
		print(
			f"Keeping {feature_missing_rows} rows with missing feature values in {season_list} ({feature_missing_cells} missing cells total)"
		)

	X = feature_frame.to_pandas().to_numpy(dtype=np.float64)
	if fit_scaler:
		scaler = StandardScaler()
		X = scaler.fit_transform(X)
	elif scaler is not None:
		X = scaler.transform(X)
	X = np.nan_to_num(X, nan=0.0)
	cat_features = extract_categorical_features(part)
	return part, X, cat_features, scaler


def prepare_data(
	df: pl.DataFrame,
	feature_cols: List[str],
	season_list: List[str],
	scaler: StandardScaler = None,
	fit_scaler: bool = False,
) -> Dict[str, np.ndarray]:
	"""Prepare multiclass result data."""

	req_cols = list({"result_label", "implied_home", "implied_draw", "implied_away", "odds_home", "odds_draw", "odds_away", "date", "raw_margin"} | set(CAT_COLS))
	filter_expr = (
		(pl.col("odds_home") > 1.0)
		& (pl.col("odds_draw") > 1.0)
		& (pl.col("odds_away") > 1.0)
		& pl.col("implied_home").is_finite()
		& pl.col("implied_draw").is_finite()
		& pl.col("implied_away").is_finite()
	)
	part, X, cat_features, scaler = _prepare_base(df, feature_cols, season_list, scaler, fit_scaler, req_cols, filter_expr)
	implied = np.stack([
		part.select("implied_home").to_pandas().values.flatten(),
		part.select("implied_draw").to_pandas().values.flatten(),
		part.select("implied_away").to_pandas().values.flatten(),
	], axis=1)
	return {
		"X": X,
		"y": part.select("result_label").to_pandas().values.flatten().astype(int),
		"implied": implied,
		"cat_features": cat_features,
		"odds_home": part.select("odds_home").to_pandas().values.flatten(),
		"odds_draw": part.select("odds_draw").to_pandas().values.flatten(),
		"odds_away": part.select("odds_away").to_pandas().values.flatten(),
		"raw_margin": part.select("raw_margin").to_pandas().values.flatten(),
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
	"""Convert prepared arrays to a PyTorch dataloader."""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if pin_memory is None:
		pin_memory = device.type == "cuda"

	tensor_x = torch.tensor(data["X"], dtype=torch.float32)
	tensor_cat = torch.tensor(data["cat_features"], dtype=torch.long)
	tensor_implied = torch.tensor(data["implied"], dtype=torch.float32)
	tensor_y = torch.tensor(data["y"], dtype=torch.long)
	tensor_raw_margin = torch.tensor(data["raw_margin"], dtype=torch.float32)
	dataset = TensorDataset(tensor_x, tensor_cat, tensor_implied, tensor_y, tensor_raw_margin)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


def precompute_fold_data(
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
) -> List[Dict[str, Any]]:
	"""Precompute scaled train/validation arrays for each fold."""

	fold_data = []
	for fold_idx, (train_seasons, val_season) in enumerate(folds):
		print(f"  Fold {fold_idx}: train={train_seasons[0]}..{train_seasons[-1]}, val={val_season}")
		data_train = prepare_data(df, feature_cols, train_seasons, fit_scaler=True)
		data_val = prepare_data(df, feature_cols, [val_season], scaler=data_train["scaler"])
		fold_data.append({
			"X_train": data_train["X"],
			"y_train": data_train["y"],
			"implied_train": data_train["implied"],
			"cat_train": data_train["cat_features"],
			"raw_margin_train": data_train["raw_margin"],
			"X_val": data_val["X"],
			"y_val": data_val["y"],
			"implied_val": data_val["implied"],
			"cat_val": data_val["cat_features"],
			"raw_margin_val": data_val["raw_margin"],
			"odds_home_val": data_val["odds_home"],
			"odds_draw_val": data_val["odds_draw"],
			"odds_away_val": data_val["odds_away"],
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
	"""Wrap precomputed fold arrays in dataloaders."""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	train_data = {
		"X": fold["X_train"],
		"y": fold["y_train"],
		"implied": fold["implied_train"],
		"cat_features": fold["cat_train"],
		"raw_margin": fold["raw_margin_train"],
	}
	val_data = {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
		"cat_features": fold["cat_val"],
		"raw_margin": fold["raw_margin_val"],
	}
	return (
		to_loader(train_data, batch_size, shuffle=True, device=device, num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY),
		to_loader(val_data, batch_size, shuffle=False, device=device, num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY),
	)


def get_val_data_dict(fold: Dict[str, Any]) -> Dict[str, np.ndarray]:
	"""Return fold validation data in evaluation format."""

	return {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
		"cat_features": fold["cat_val"],
		"raw_margin": fold["raw_margin_val"],
		"odds_home": fold["odds_home_val"],
		"odds_draw": fold["odds_draw_val"],
		"odds_away": fold["odds_away_val"],
		"dates": fold["dates_val"],
	}


class SchedulerController:
	"""Hide whether a scheduler steps per batch or per epoch."""

	def __init__(self, scheduler, step_unit: str = "epoch", needs_metric: bool = False):
		self.scheduler = scheduler
		self.step_unit = step_unit
		self.needs_metric = needs_metric

	def step_batch(self):
		if self.scheduler is not None and self.step_unit == "batch":
			self.scheduler.step()

	def step_epoch(self, metric: float = None):
		if self.scheduler is None or self.step_unit != "epoch":
			return
		if self.needs_metric:
			if metric is None:
				raise ValueError("Metric required for plateau scheduler step")
			self.scheduler.step(metric)
		else:
			self.scheduler.step()


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
	"""Create the configured optimizer."""

	optimizer_name = getattr(config, "optimizer_name", "adamw").lower()
	betas = (config.beta1, getattr(config, "beta2", 0.999))
	eps = getattr(config, "optimizer_eps", 1e-8)
	if optimizer_name == "adamw":
		return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=betas, eps=eps)
	if optimizer_name == "radam":
		return torch.optim.RAdam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=betas, eps=eps)
	raise ValueError(f"Unsupported optimizer_name={config.optimizer_name}")


def create_scheduler(
	optimizer: torch.optim.Optimizer,
	config: TrainConfig,
	steps_per_epoch: int,
) -> SchedulerController:
	"""Create the configured learning-rate scheduler."""

	min_lr = config.lr * getattr(config, "scheduler_min_lr_ratio", 0.01)
	scheduler_name = getattr(config, "scheduler_name", "cosine").lower()
	if scheduler_name == "none":
		return SchedulerController(None)
	if scheduler_name == "cosine":
		return SchedulerController(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=min_lr))
	if scheduler_name == "warmup_cosine":
		warmup_epochs = max(0, int(getattr(config, "scheduler_warmup_epochs", 0)))
		warmup_epochs = min(warmup_epochs, max(0, config.epochs - 1))
		if warmup_epochs == 0:
			return SchedulerController(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=min_lr))
		remaining_epochs = max(1, config.epochs - warmup_epochs)
		warmup = torch.optim.lr_scheduler.LinearLR(
			optimizer,
			start_factor=max(1e-3, getattr(config, "scheduler_warmup_start_factor", 0.1)),
			total_iters=warmup_epochs,
		)
		cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=min_lr)
		scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
		return SchedulerController(scheduler)
	if scheduler_name == "plateau":
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=getattr(config, "scheduler_plateau_factor", 0.5),
			patience=getattr(config, "scheduler_plateau_patience", 3),
			threshold=getattr(config, "scheduler_plateau_threshold", 1e-4),
			min_lr=min_lr,
		)
		return SchedulerController(scheduler, needs_metric=True)
	if scheduler_name == "onecycle":
		scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=config.lr,
			epochs=max(1, config.epochs),
			steps_per_epoch=max(1, steps_per_epoch),
			pct_start=getattr(config, "onecycle_pct_start", 0.3),
			div_factor=max(1.0, getattr(config, "onecycle_div_factor", 25.0)),
			final_div_factor=max(1.0, getattr(config, "onecycle_final_div_factor", 1000.0)),
			anneal_strategy="cos",
		)
		return SchedulerController(scheduler, step_unit="batch")
	raise ValueError(f"Unsupported scheduler_name={config.scheduler_name}")


class EarlyStopping:
	"""Early stopping handler with best-weight tracking."""

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
		if self.best_model_state is not None:
			model.load_state_dict(self.best_model_state)


def train_model(
	config: TrainConfig,
	train_loader: DataLoader,
	val_loader: DataLoader = None,
	device: torch.device = None,
	trial=None,
	verbose: bool = True,
) -> Tuple[GatedResidualModel, Dict[str, list], float]:
	"""Train the result model with optional early stopping."""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	cat_config = getattr(config, "cat_config", None)
	model = GatedResidualModel(
		input_dim=config.input_dim,
		hidden_layers=config.hidden_layers,
		n_classes=3,
		cat_config=cat_config,
		gate_hidden_dim=getattr(config, "gate_hidden_dim", 32),
		dropout=config.dropout,
		norm=config.norm,
		activation=config.activation,
		gate_target_budget=getattr(config, "gate_target_budget", 0.2),
	).to(device)
	optimizer = create_optimizer(model, config)
	scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
	use_validation = val_loader is not None
	early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4) if use_validation else None
	history = {"train_loss": [], "val_loss": [], "gate_mean": [], "gate_std": []}

	for epoch in range(1, config.epochs + 1):
		model.train()
		total_loss = 0.0
		for batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin in train_loader:
			batch_x = batch_x.to(device)
			batch_cat = batch_cat.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)
			batch_raw_margin = batch_raw_margin.to(device)
			cat_in = batch_cat if cat_config is not None else None

			optimizer.zero_grad(set_to_none=True)
			loss = gated_loss(
				model,
				batch_x,
				cat_in,
				batch_implied,
				batch_y,
				batch_raw_margin,
				gate_mean_weight=getattr(config, "gate_mean_weight", 0.01),
				gate_sat_weight=getattr(config, "gate_sat_weight", 0.001),
				lambda_repulsion=getattr(config, "lambda_repulsion", 0.0),
				lambda_corr=getattr(config, "lambda_corr", 0.0),
			)
			loss.backward()
			if getattr(config, "max_grad_norm", 0.0) and config.max_grad_norm > 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
			optimizer.step()
			scheduler.step_batch()
			total_loss += loss.item() * len(batch_x)

		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)

		if not use_validation:
			scheduler.step_epoch(avg_train_loss)
			if mlflow.active_run():
				mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
				mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=epoch)
			if verbose and (epoch % 10 == 0 or epoch == 1):
				print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f}")
			continue

		model.eval()
		val_loss = 0.0
		all_gates = []
		with torch.no_grad():
			for batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin in val_loader:
				batch_x = batch_x.to(device)
				batch_cat = batch_cat.to(device)
				batch_implied = batch_implied.to(device)
				batch_y = batch_y.to(device)
				batch_raw_margin = batch_raw_margin.to(device)
				cat_in = batch_cat if cat_config is not None else None
				pred_logits = model(batch_x, cat_in, batch_implied, batch_raw_margin)
				loss = F.cross_entropy(pred_logits, batch_y.view(-1).long())
				val_loss += loss.item() * len(batch_x)
				gate_stats = model.get_gate_stats(batch_x, cat_in, batch_implied, batch_raw_margin)
				all_gates.append(gate_stats["gate_values"])

		avg_val_loss = val_loss / len(val_loader.dataset)
		history["val_loss"].append(avg_val_loss)
		all_gates = np.concatenate(all_gates, axis=0)
		gate_mean = all_gates.mean(axis=0).tolist()
		gate_std = all_gates.std(axis=0).tolist()
		history["gate_mean"].append(gate_mean)
		history["gate_std"].append(gate_std)
		scheduler.step_epoch(avg_val_loss)
		early_stopping(avg_val_loss, model)

		if mlflow.active_run():
			mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
			mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
			mlflow.log_metric("lr", float(optimizer.param_groups[0]["lr"]), step=epoch)

		if verbose and (epoch % 10 == 0 or epoch == 1):
			print(
				f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | Gate: [{gate_mean[0]:.3f}, {gate_mean[1]:.3f}, {gate_mean[2]:.3f}]"
			)

		if trial is not None:
			trial.report(min(history["val_loss"]), epoch)
			if trial.should_prune():
				import optuna
				raise optuna.TrialPruned()

		if early_stopping.early_stop:
			if verbose:
				print(f"Early stopping at epoch {epoch}")
			break

	early_stopping.load_best_weights(model) if use_validation else None
	best_loss = early_stopping.best_loss if use_validation else history["train_loss"][-1]
	return model, history, best_loss


def evaluate_implied_baseline(data: Dict[str, np.ndarray]) -> Dict:
	"""Evaluate normalized market implied probabilities as the baseline."""

	implied_probs = data["implied"]
	y_true = data["y"]
	preds = np.argmax(implied_probs, axis=1)
	acc = accuracy_score(y_true, preds)
	y_onehot = np.eye(3)[y_true]
	brier = float(np.mean(np.sum((implied_probs - y_onehot) ** 2, axis=1)))
	ll = log_loss(y_true, implied_probs, labels=[0, 1, 2])
	rps = ranked_probability_score(y_true, implied_probs)
	return {
		"accuracy": float(acc),
		"brier": brier,
		"rps": float(rps),
		"log_loss": float(ll),
	}
