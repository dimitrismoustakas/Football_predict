"""
Training utilities and data preparation functions.

Supports two task types:
- Binary classification (over/under): 1 implied probability
- Multiclass classification (home/draw/away): 3 implied probabilities

Categorical features (via CategoricalEmbedder):
- league_idx: embedded (configurable dim)
- home_promoted, away_promoted: binary

Continuous features (via StandardScaler):
- Rolling stats (ovr__*__r5__h/a)
- Elo features
- Schedule congestion features
- season_progress: [0,1] representing position in season
"""

import sys
import copy
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

from training.models.neural_net import (
	GatedResidualModel,
	GatedResidualModelBinary,
	TrainConfig,
	TaskType,
	gated_loss_multiclass,
	gated_loss_binary,
)
from training.evaluation.metrics import accuracy_score, brier_score_loss, log_loss


# ============================================================================
# CATEGORICAL FEATURE CONSTANTS
# ============================================================================

# Categorical columns expected in the DataFrame
CAT_COLS = ["league_idx", "home_promoted", "away_promoted"]


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================


def load_frame(parquet_path: Path) -> pl.DataFrame:
	"""Load Parquet file into Polars DataFrame."""
	return pl.scan_parquet(str(parquet_path)).collect()


def select_feature_columns(df: pl.DataFrame) -> List[str]:
	"""Select feature columns based on naming convention."""
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
	# Return only columns that exist in the dataframe
	return [c for c in feat_cols if c in cols]


def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
	"""Filter to matches where both teams have at least 5 prior games."""
	need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
	missing = [c for c in need_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for history filter: {missing}")
	return df.filter(
		(pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5)
	)


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
	"""Get the held-out test season (current/latest season, even if incomplete)."""
	seasons = (
		df.select(pl.col("season").cast(pl.Utf8))
		.unique()
		.sort(by="season")
		.to_series()
		.to_list()
	)
	if len(seasons) < 2:
		raise ValueError("Need at least 2 seasons to have a test season.")
	return seasons[-1]


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
	"""Add match result and implied probability columns for over/under."""
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

	prob_cols, norm = _normalize_implied(["odds_over", "odds_under"], "implied")

	return df.with_columns([
		(prob_cols["implied_over"]).alias("implied_over_prob"),
		# Raw margin (overround) - carries info about bookmaker confidence/liquidity
		norm.alias("raw_margin_ou"),
	])


def extract_categorical_features(df: pl.DataFrame) -> np.ndarray:
	"""
	Extract categorical features as a numpy array.
	
	Returns shape (n_rows, 3) with columns:
		[league_idx, home_promoted, away_promoted]
	"""
	league_idx = df.select("league_idx").to_numpy().flatten().astype(np.int64)
	home_promoted = df.select("home_promoted").to_numpy().flatten().astype(np.int64)
	away_promoted = df.select("away_promoted").to_numpy().flatten().astype(np.int64)
	
	return np.stack([league_idx, home_promoted, away_promoted], axis=1)


def get_num_leagues(df: pl.DataFrame) -> int:
	"""Get the number of unique leagues in the dataset."""
	return df.select("league_idx").unique().height


def add_targets_and_implied_result(df: pl.DataFrame) -> pl.DataFrame:
	"""
	Add match result targets and implied probabilities for result prediction.
	
	Requires odds columns: odds_h, odds_d, odds_a (or odds_home, odds_draw, odds_away)
	Adds:
		- result_label: 0=Home, 1=Draw, 2=Away
		- implied_home, implied_draw, implied_away: normalized implied probs
		- odds_home, odds_draw, odds_away: renamed odds columns (if needed)
	"""
	# Support both naming conventions
	if "odds_h" in df.columns and "odds_home" not in df.columns:
		df = df.with_columns([
			pl.col("odds_h").alias("odds_home"),
			pl.col("odds_d").alias("odds_draw"),
			pl.col("odds_a").alias("odds_away"),
		])
	
	need_odds = ["odds_home", "odds_draw", "odds_away"]
	missing = [c for c in need_odds if c not in df.columns]
	if missing:
		raise ValueError(f"Missing odds columns for result prediction: {missing}")
	
	# Create numeric result label
	df = df.with_columns(
		pl.when(pl.col("home_goals") > pl.col("away_goals"))
		.then(pl.lit(0))  # Home win
		.when(pl.col("home_goals") == pl.col("away_goals"))
		.then(pl.lit(1))  # Draw
		.otherwise(pl.lit(2))  # Away win
		.alias("result_label")
	)
	
	prob_cols, norm = _normalize_implied([
		"odds_home",
		"odds_draw",
		"odds_away",
	], "implied")

	return df.with_columns([
		(prob_cols["implied_home"]).alias("implied_home"),
		(prob_cols["implied_draw"]).alias("implied_draw"),
		(prob_cols["implied_away"]).alias("implied_away"),
		# Raw margin (overround) - carries info about bookmaker confidence/liquidity
		norm.alias("raw_margin"),
	])


def _normalize_implied(odds_cols: List[str], prefix: str) -> Tuple[Dict[str, pl.Expr], pl.Expr]:
	"""Compute normalized implied probabilities and return per-col expressions plus norm."""
	inv_odds = [1 / pl.col(col) for col in odds_cols]
	norm = inv_odds[0]
	for expr in inv_odds[1:]:
		norm = norm + expr
	
	prob_cols = {}
	for col, expr in zip(odds_cols, inv_odds):
		suffix = col.replace("odds_", "")
		prob_cols[f"{prefix}_{suffix}"] = expr / norm
	
	return prob_cols, norm


def _prepare_base(
	df: pl.DataFrame,
	feature_cols: List[str],
	season_list: List[str],
	scaler: StandardScaler,
	fit_scaler: bool,
	req_cols: List[str],
	filter_expr: pl.Expr,
) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray, StandardScaler]:
	"""Filter, drop nulls, scale features, and extract categorical features."""
	part = df.filter(pl.col("season").cast(pl.Utf8).is_in(list(season_list)))

	initial_count = len(part)
	filtered = part.filter(filter_expr)
	invalid_count = initial_count - len(filtered)

	part = filtered.drop_nulls(subset=req_cols)
	missing_required_count = len(filtered) - len(part)

	if invalid_count:
		print(
			f"Dropped {invalid_count} rows due to invalid odds/non-finite implied values in {season_list}"
		)

	if missing_required_count:
		null_counts_row = (
			filtered.select([
				pl.col(col).is_null().sum().alias(col)
				for col in req_cols
			])
			.to_dicts()[0]
		)
		top_missing = [
			(col, count)
			for col, count in sorted(
				null_counts_row.items(), key=lambda item: (-item[1], item[0])
			)
			if count > 0
		][:8]
		print(
			f"Dropped {missing_required_count} rows due to missing required features in {season_list}: {top_missing}"
		)

	feature_frame = part.select([
		pl.col(c).cast(pl.Float64).alias(c)
		for c in feature_cols
	])
	feature_missing_cells = int(
		feature_frame.select([
			pl.col(c).is_null().sum().alias(c)
			for c in feature_cols
		]).sum_horizontal().item()
	)
	feature_missing_rows = feature_frame.filter(
		pl.any_horizontal([pl.col(c).is_null() for c in feature_cols])
	).height

	if feature_missing_rows:
		print(
			f"Keeping {feature_missing_rows} rows with missing feature values in {season_list} "
			f"({feature_missing_cells} missing cells total)"
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
	"""Selects data, scales features, extracts categorical features, returns a dictionary of arrays."""
	req_cols = list(
		{"Over", "implied_over_prob", "odds_over", "odds_under", "date", "raw_margin_ou"}
		| set(CAT_COLS)
	)

	filter_expr = (
		(pl.col("odds_over") > 1.0)
		& (pl.col("odds_under") > 1.0)
		& pl.col("implied_over_prob").is_finite()
	)
	part, X, cat_features, scaler = _prepare_base(
		df,
		feature_cols,
		season_list,
		scaler,
		fit_scaler,
		req_cols,
		filter_expr,
	)

	return {
		"X": X,
		"y": part.select("Over").to_pandas().values.flatten().astype(int),
		"implied": part.select("implied_over_prob").to_pandas().values.flatten(),
		"cat_features": cat_features,
		"raw_margin": part.select("raw_margin_ou").to_pandas().values.flatten(),
		"odds_over": part.select("odds_over").to_pandas().values.flatten(),
		"odds_under": part.select("odds_under").to_pandas().values.flatten(),
		"dates": part.select("date").to_pandas().values.flatten(),
		"scaler": scaler,
	}


def prepare_data_result(
	df: pl.DataFrame,
	feature_cols: List[str],
	season_list: List[str],
	scaler: StandardScaler = None,
	fit_scaler: bool = False,
) -> Dict[str, np.ndarray]:
	"""
	Prepare data for result prediction (home/draw/away).
	
	Returns dict with:
		- X: scaled features
		- y: result labels (0=Home, 1=Draw, 2=Away)
		- implied: shape (n, 3) with [home, draw, away] implied probs
		- cat_features: shape (n, 3) with [league_idx, home_promoted, away_promoted]
		- odds_home, odds_draw, odds_away: original odds
		- dates: match dates
		- scaler: fitted StandardScaler
	"""
	req_cols = list(
		{"result_label", "implied_home", "implied_draw", "implied_away",
		   "odds_home", "odds_draw", "odds_away", "date", "raw_margin"}
		| set(CAT_COLS)
	)

	filter_expr = (
		(pl.col("odds_home") > 1.0)
		& (pl.col("odds_draw") > 1.0)
		& (pl.col("odds_away") > 1.0)
		& pl.col("implied_home").is_finite()
		& pl.col("implied_draw").is_finite()
		& pl.col("implied_away").is_finite()
	)
	part, X, cat_features, scaler = _prepare_base(
		df,
		feature_cols,
		season_list,
		scaler,
		fit_scaler,
		req_cols,
		filter_expr,
	)

	# Stack implied probs into shape (n, 3)
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
	task_type: TaskType = "binary",
) -> DataLoader:
	"""
	Convert data dictionary to PyTorch DataLoader.
	
	Args:
		data: Dict with 'X', 'y', 'implied', 'cat_features', 'raw_margin' arrays
		batch_size: Batch size
		shuffle: Whether to shuffle data
		device: Target device (for pin_memory default)
		num_workers: Number of worker processes for data loading
		pin_memory: Whether to pin memory (defaults to True for CUDA)
		task_type: "binary" for over/under, "multiclass" for result
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if pin_memory is None:
		pin_memory = device.type == "cuda"
		
	tensor_x = torch.tensor(data["X"], dtype=torch.float32)
	tensor_cat = torch.tensor(data["cat_features"], dtype=torch.long)
	tensor_raw_margin = torch.tensor(data["raw_margin"], dtype=torch.float32)
	
	if task_type == "binary":
		tensor_implied = torch.tensor(data["implied"], dtype=torch.float32)
		tensor_y = torch.tensor(data["y"], dtype=torch.float32)
		ds = TensorDataset(tensor_x, tensor_cat, tensor_implied, tensor_y, tensor_raw_margin)
	else:
		# multiclass: implied is shape (n, 3), y is class label
		tensor_implied = torch.tensor(data["implied"], dtype=torch.float32)
		tensor_y = torch.tensor(data["y"], dtype=torch.long)
		ds = TensorDataset(tensor_x, tensor_cat, tensor_implied, tensor_y, tensor_raw_margin)
	
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

# Optimal settings for data loading
# NOTE: On Windows, multiprocessing spawn overhead makes num_workers > 0 slow
# for small datasets. Using 0 workers is ~20x faster for our data size.
OPTIMAL_NUM_WORKERS = 0 if sys.platform == "win32" else 4
PIN_MEMORY = torch.cuda.is_available()


def precompute_fold_data(
	df: pl.DataFrame,
	feature_cols: List[str],
	folds: List[Tuple[List[str], str]],
	task_type: TaskType = "binary",
) -> List[Dict[str, Any]]:
	"""
	Precompute scaled train/val data for each CV fold.
	
	Called ONCE before Optuna search to avoid refitting scalers on every trial.
	Each trial only needs to wrap DataLoaders with the appropriate batch size.
	
	Args:
		df: DataFrame with features and targets
		feature_cols: List of feature column names
		folds: List of (train_seasons, val_season) tuples
		task_type: "binary" for over/under, "multiclass" for result
	
	Returns:
		List of dicts, one per fold, each containing:
		- X_train, y_train, implied_train, cat_train: scaled training arrays
		- X_val, y_val, implied_val, cat_val, odds columns, dates_val
		- scaler: fitted StandardScaler for this fold
		- train_seasons, val_season: for reference
		- task_type: the task type for this fold data
	"""
	fold_data = []
	prepare_fn = prepare_data if task_type == "binary" else prepare_data_result
	
	for fold_idx, (train_seasons, val_season) in enumerate(folds):
		print(f"  Fold {fold_idx}: train={train_seasons[0]}..{train_seasons[-1]}, val={val_season}")
		
		# Prepare training data (fits scaler)
		data_train = prepare_fn(df, feature_cols, train_seasons, fit_scaler=True)
		
		# Prepare validation data (uses fitted scaler)
		data_val = prepare_fn(df, feature_cols, [val_season], scaler=data_train["scaler"])
		
		fold_dict = {
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
			"dates_val": data_val["dates"],
			"scaler": data_train["scaler"],
			"train_seasons": train_seasons,
			"val_season": val_season,
			"task_type": task_type,
		}
		
		# Add task-specific odds columns
		if task_type == "binary":
			fold_dict["odds_over_val"] = data_val["odds_over"]
			fold_dict["odds_under_val"] = data_val["odds_under"]
		else:
			fold_dict["odds_home_val"] = data_val["odds_home"]
			fold_dict["odds_draw_val"] = data_val["odds_draw"]
			fold_dict["odds_away_val"] = data_val["odds_away"]
		
		fold_data.append(fold_dict)
	
	return fold_data


def fold_data_to_loaders(
	fold: Dict[str, Any],
	batch_size: int,
	device: torch.device = None,
	task_type: TaskType = None,
) -> Tuple[DataLoader, DataLoader]:
	"""
	Convert precomputed fold data to DataLoaders with specified batch size.
	
	Fast because data is already scaled - just wraps in tensors.
	Uses optimized num_workers and pin_memory settings.
	
	Args:
		fold: Precomputed fold data dict
		batch_size: Batch size for DataLoaders
		device: Target device
		task_type: Override task type (defaults to fold's task_type)
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	# Use fold's task_type if not overridden
	if task_type is None:
		task_type = fold.get("task_type", "binary")
	
	train_data = {
		"X": fold["X_train"],
		"y": fold["y_train"],
		"implied": fold["implied_train"],
		"cat_features": fold["cat_train"],
		"raw_margin": fold["raw_margin_train"],
	}
	train_loader = to_loader(
		train_data, batch_size, shuffle=True, device=device,
		num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY,
		task_type=task_type,
	)
	
	val_data = {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
		"cat_features": fold["cat_val"],
		"raw_margin": fold["raw_margin_val"],
	}
	val_loader = to_loader(
		val_data, batch_size, shuffle=False, device=device,
		num_workers=OPTIMAL_NUM_WORKERS, pin_memory=PIN_MEMORY,
		task_type=task_type,
	)
	
	return train_loader, val_loader


def get_val_data_dict(fold: Dict[str, Any]) -> Dict[str, np.ndarray]:
	"""Extract validation data in format expected by evaluate_model."""
	task_type = fold.get("task_type", "binary")
	
	result = {
		"X": fold["X_val"],
		"y": fold["y_val"],
		"implied": fold["implied_val"],
		"cat_features": fold["cat_val"],
		"dates": fold["dates_val"],
	}
	
	if task_type == "binary":
		result["odds_over"] = fold["odds_over_val"]
		result["odds_under"] = fold["odds_under_val"]
	else:
		result["odds_home"] = fold["odds_home_val"]
		result["odds_draw"] = fold["odds_draw_val"]
		result["odds_away"] = fold["odds_away_val"]
	
	return result


# ============================================================================
# TRAINING
# ============================================================================


def create_scheduler(
	optimizer: torch.optim.Optimizer,
	epochs: int = 100,
	lr: float = 1e-3,
) -> torch.optim.lr_scheduler.LRScheduler:
	"""
	Create a cosine annealing learning rate scheduler.
	
	Args:
		optimizer: PyTorch optimizer
		epochs: Total training epochs
		lr: Base learning rate (for eta_min calculation)
	"""
	return torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer, T_max=epochs, eta_min=lr * 0.01
	)


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
	val_loader: DataLoader = None,
	device: torch.device = None,
	trial = None,
	verbose: bool = True,
) -> Tuple:
	"""
	Train a gated residual model with optional early stopping.
	
	Supports both binary (over/under) and multiclass (result) tasks based on config.task_type.
	
	Args:
		config: Training configuration
		train_loader: DataLoader for training data (must include raw_margin)
		val_loader: DataLoader for validation data. If None, trains for exactly config.epochs
		            without early stopping (useful for final retraining after hyperparameter search).
		device: Device to train on
		trial: Optuna trial for pruning (only used when val_loader is provided)
		verbose: Whether to print progress
	
	Returns (model, history, best_val_loss)
	        If val_loader is None, best_val_loss will be the final training loss.
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	task_type = getattr(config, "task_type", "binary")
	cat_config = getattr(config, "cat_config", None)
	gate_hidden_dim = getattr(config, "gate_hidden_dim", 32)
	gate_target_budget = getattr(config, "gate_target_budget", 0.2)
	gate_mean_weight = getattr(config, "gate_mean_weight", 0.01)
	gate_sat_weight = getattr(config, "gate_sat_weight", 0.001)
	lambda_repulsion = getattr(config, "lambda_repulsion", 0.0)
	lambda_corr = getattr(config, "lambda_corr", 0.0)
	
	# Determine if we're doing validation/early-stopping
	use_validation = val_loader is not None
	
	# Create gated model based on task type
	if task_type == "binary":
		model = GatedResidualModelBinary(
			input_dim=config.input_dim,
			hidden_layers=config.hidden_layers,
			cat_config=cat_config,
			gate_hidden_dim=gate_hidden_dim,
			dropout=config.dropout,
			norm=config.norm,
			activation=config.activation,
			gate_target_budget=gate_target_budget,
		).to(device)
		loss_fn = gated_loss_binary
	else:
		model = GatedResidualModel(
			input_dim=config.input_dim,
			hidden_layers=config.hidden_layers,
			n_classes=3,
			cat_config=cat_config,
			gate_hidden_dim=gate_hidden_dim,
			dropout=config.dropout,
			norm=config.norm,
			activation=config.activation,
			gate_target_budget=gate_target_budget,
		).to(device)
		loss_fn = gated_loss_multiclass
	
	optimizer = torch.optim.AdamW(
		model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(config.beta1, 0.999)
	)
	
	scheduler = create_scheduler(
		optimizer,
		epochs=config.epochs,
		lr=config.lr,
	)
	
	# Only use early stopping when we have validation data
	early_stopping = EarlyStopping(patience=config.patience, min_delta=1e-4) if use_validation else None

	history = {"train_loss": [], "val_loss": [], "gate_mean": [], "gate_std": []}

	for epoch in range(1, config.epochs + 1):
		# Training phase
		model.train()
		total_loss = 0.0
		for batch_x, batch_cat, batch_implied, batch_y, batch_raw_margin in train_loader:
			batch_x = batch_x.to(device)
			batch_cat = batch_cat.to(device)
			batch_implied = batch_implied.to(device)
			batch_y = batch_y.to(device)
			batch_raw_margin = batch_raw_margin.to(device)
			cat_in = batch_cat if cat_config is not None else None

			optimizer.zero_grad()
			loss = loss_fn(
				model,
				batch_x,
				cat_in,
				batch_implied,
				batch_y,
				batch_raw_margin,
				gate_mean_weight=gate_mean_weight,
				gate_sat_weight=gate_sat_weight,
				lambda_repulsion=lambda_repulsion,
				lambda_corr=lambda_corr,
			)
			loss.backward()
			optimizer.step()
			total_loss += loss.item() * len(batch_x)

		avg_train_loss = total_loss / len(train_loader.dataset)
		history["train_loss"].append(avg_train_loss)

		# Validation phase (only if we have validation data)
		if use_validation:
			model.eval()
			val_loss = 0.0
			all_gates = []
			with torch.no_grad():
				for bx, bc, bi, by, b_raw_margin in val_loader:
					bx = bx.to(device)
					bc = bc.to(device)
					bi = bi.to(device)
					by = by.to(device)
					b_raw_margin = b_raw_margin.to(device)
					
					# Get loss without gate regularization for fair comparison
					cat_in = bc if cat_config is not None else None
					pred_logits = model(bx, cat_in, bi, b_raw_margin)
					if task_type == "binary":
						loss = F.binary_cross_entropy_with_logits(pred_logits, by)
					else:
						loss = F.cross_entropy(pred_logits, by.view(-1).long())
					val_loss += loss.item() * len(bx)
					
					# Collect gate values
					gate_stats = model.get_gate_stats(bx, cat_in, bi, b_raw_margin)
					all_gates.append(gate_stats["gate_values"])

			avg_val_loss = val_loss / len(val_loader.dataset)
			history["val_loss"].append(avg_val_loss)
			
			# Track gate statistics
			all_gates = np.concatenate(all_gates, axis=0)
			if task_type == "binary":
				gate_mean = float(all_gates.mean())
				gate_std = float(all_gates.std())
			else:
				gate_mean = all_gates.mean(axis=0).tolist()
				gate_std = all_gates.std(axis=0).tolist()
			history["gate_mean"].append(gate_mean)
			history["gate_std"].append(gate_std)
			
			# Step scheduler
			scheduler.step()
			
			early_stopping(avg_val_loss, model)

			# Log to MLflow if in an active run
			if mlflow.active_run():
				mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
				mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

			if verbose and (epoch % 10 == 0 or epoch == 1):
				if task_type == "binary":
					print(
						f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
						f"Gate: {gate_mean:.3f} +/- {gate_std:.3f}"
					)
				else:
					print(
						f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | "
						f"Gate: [{gate_mean[0]:.3f}, {gate_mean[1]:.3f}, {gate_mean[2]:.3f}]"
					)

			# Optuna pruning
			if trial is not None:
				trial.report(min(history["val_loss"]), epoch)
				if trial.should_prune():
					import optuna
					raise optuna.TrialPruned()

			if early_stopping.early_stop:
				if verbose:
					print(f"Early stopping at epoch {epoch}")
				break
		else:
			# No validation: step scheduler
			scheduler.step()
			
			# Log to MLflow if in an active run
			if mlflow.active_run():
				mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

			if verbose and (epoch % 10 == 0 or epoch == 1):
				print(f"Epoch {epoch:03d} | Train: {avg_train_loss:.5f}")

	if use_validation:
		early_stopping.load_best_weights(model)
		return model, history, early_stopping.best_loss
	else:
		return model, history, history["train_loss"][-1]


def evaluate_implied_baseline(data: Dict[str, np.ndarray], task_type: TaskType = "binary") -> Dict:
	"""
	Evaluate market implied probabilities as baseline predictions.
	
	Args:
		data: Dict with 'implied' and 'y' arrays
		task_type: "binary" for over/under, "multiclass" for result
	"""
	from training.evaluation.metrics import ranked_probability_score
	
	implied_probs = data["implied"]
	y_true = data["y"]
	
	if task_type == "binary":
		preds = (implied_probs >= 0.5).astype(int)
		acc = accuracy_score(y_true, preds)
		brier = brier_score_loss(y_true, implied_probs)
		ll = log_loss(y_true, np.c_[1 - implied_probs, implied_probs], labels=[0, 1])
		return {
			"accuracy": float(acc),
			"brier": float(brier),
			"log_loss": float(ll),
		}
	else:
		# Multiclass: implied is shape (n, 3)
		preds = np.argmax(implied_probs, axis=1)
		acc = accuracy_score(y_true, preds)
		# Brier for multiclass: mean squared error of one-hot vs probs
		n_classes = 3
		y_onehot = np.eye(n_classes)[y_true]
		brier = np.mean(np.sum((implied_probs - y_onehot) ** 2, axis=1))
		ll = log_loss(y_true, implied_probs, labels=[0, 1, 2])
		rps = ranked_probability_score(y_true, implied_probs)
		return {
			"accuracy": float(acc),
			"brier": float(brier),
			"rps": float(rps),
			"log_loss": float(ll),
		}