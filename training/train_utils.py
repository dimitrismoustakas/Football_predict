"""
Training utilities and data preparation functions for match-result prediction.

Categorical features:
- `league_idx`
- `home_promoted`
- `away_promoted`

Continuous features are scaled with `StandardScaler`.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from training.evaluation.metrics import accuracy_score, log_loss, ranked_probability_score
from utils.portfolio import DEFAULT_BUDGET_STRATEGY, DEFAULT_KELLY_FRACTION, evaluate_budget_strategy
from utils.paths import PROJECT_ROOT

CAT_COLS = ["league_idx", "home_promoted", "away_promoted"]
RESULT_FEATURES_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json"
FINGERPRINT_PRIORITY_COLS = [
	"season",
	"date",
	"match_id",
	"game_id",
	"league",
	"league_id",
	"team",
	"home_team",
	"away_team",
	"team_id",
	"home_team_id",
	"away_team_id",
	"home_goals",
	"away_goals",
	"odds_home",
	"odds_draw",
	"odds_away",
]


def load_frame(parquet_path: Path) -> pl.DataFrame:
	"""Load a parquet file into a Polars frame."""

	return pl.scan_parquet(str(parquet_path)).collect()


def load_feature_manifest(feature_path: Path = RESULT_FEATURES_PATH) -> List[str]:
	"""Load the ordered feature manifest for the canonical result model."""

	with open(feature_path, "r", encoding="utf-8") as file:
		feature_cols = json.load(file)
	if not isinstance(feature_cols, list) or not all(isinstance(col, str) for col in feature_cols):
		raise ValueError(f"Invalid feature manifest: {feature_path}")
	return feature_cols


def select_feature_columns(df: pl.DataFrame, feature_path: Path = RESULT_FEATURES_PATH) -> List[str]:
	"""Select the fixed result-model feature subset."""

	manifest_cols = load_feature_manifest(feature_path)
	cols = set(df.columns)
	missing = [col for col in manifest_cols if col not in cols]
	if missing:
		raise ValueError(f"Missing feature columns from manifest {feature_path}: {missing}")
	return manifest_cols


def filter_min_history(df: pl.DataFrame) -> pl.DataFrame:
	"""Filter to matches where both teams have at least five prior games."""

	need_cols = ["ovr__games__r5__h", "ovr__games__r5__a"]
	missing = [col for col in need_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns for history filter: {missing}")
	return df.filter((pl.col("ovr__games__r5__h") >= 5) & (pl.col("ovr__games__r5__a") >= 5))


def get_sorted_seasons(df: pl.DataFrame) -> List[str]:
	"""Return sorted season labels as strings."""

	return df.select(pl.col("season").cast(pl.Utf8)).unique().sort(by="season").to_series().to_list()


def resolve_test_season(df: pl.DataFrame, configured_test_season: str | None = None) -> str:
	"""Return the configured fixed test season, defaulting to the latest season when unset."""

	seasons = get_sorted_seasons(df)
	if len(seasons) < 2:
		raise ValueError("Need at least 2 seasons to have a test season.")
	if configured_test_season is None:
		return seasons[-1]

	test_season = str(configured_test_season)
	if test_season not in seasons:
		raise ValueError(f"Configured test season {test_season} not found in data. Available seasons: {seasons}")
	if seasons.index(test_season) == 0:
		raise ValueError(f"Configured test season {test_season} has no prior seasons available for training.")
	return test_season


def generate_rolling_cv_folds(
	df: pl.DataFrame,
	n_folds: int = 3,
	test_season: str | None = None,
) -> List[Tuple[List[str], str]]:
	"""Generate expanding-window validation folds using only seasons before the fixed test season."""

	seasons = get_sorted_seasons(df)
	resolved_test_season = resolve_test_season(df, test_season)
	test_idx = seasons.index(resolved_test_season)
	available = seasons[:test_idx]
	if len(available) < n_folds + 1:
		raise ValueError(
			f"Need at least {n_folds + 1} seasons before test season {resolved_test_season} for {n_folds}-fold rolling CV. "
			f"Got {len(available)}."
		)

	folds = []
	for fold_idx in range(n_folds):
		val_idx = len(available) - n_folds + fold_idx
		val_season = available[val_idx]
		train_seasons = available[:val_idx]
		folds.append((train_seasons, val_season))
	return folds


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
		"home_goals": part.select("home_goals").to_pandas().values.flatten().astype(np.float64),
		"away_goals": part.select("away_goals").to_pandas().values.flatten().astype(np.float64),
		"feature_cols": list(feature_cols),
		"implied": implied,
		"cat_features": cat_features,
		"odds_home": part.select("odds_home").to_pandas().values.flatten(),
		"odds_draw": part.select("odds_draw").to_pandas().values.flatten(),
		"odds_away": part.select("odds_away").to_pandas().values.flatten(),
		"raw_margin": part.select("raw_margin").to_pandas().values.flatten(),
		"dates": part.select("date").to_pandas().values.flatten(),
		"scaler": scaler,
	}


def _stable_hash_value(value: Any) -> str:
	"""Serialize scalar values consistently for snapshot fingerprinting."""

	if value is None:
		return "<null>"
	if isinstance(value, float):
		return format(value, ".12g")
	return str(value)


def compute_frame_fingerprint(df: pl.DataFrame, fingerprint_cols: List[str] | None = None) -> str:
	"""Compute a stable content fingerprint for a Polars frame."""

	if df.height == 0:
		return hashlib.sha256(b"empty").hexdigest()

	if fingerprint_cols is None:
		fingerprint_cols = [col for col in FINGERPRINT_PRIORITY_COLS if col in df.columns]
	if not fingerprint_cols:
		fingerprint_cols = list(df.columns)

	sort_cols = [col for col in ["season", "date", "match_id", "game_id", "team", "home_team", "away_team"] if col in fingerprint_cols]
	if not sort_cols:
		sort_cols = fingerprint_cols

	fingerprint_df = df.select(fingerprint_cols).sort(sort_cols)
	hasher = hashlib.sha256()
	for row in fingerprint_df.iter_rows(named=False):
		hasher.update("|".join(_stable_hash_value(value) for value in row).encode("utf-8"))
		hasher.update(b"\n")
	return hasher.hexdigest()


def build_data_snapshot(df: pl.DataFrame, test_season: str) -> Dict[str, Any]:
	"""Build versioning metadata for the canonical evaluation dataset."""

	season_counts_df = (
		df.with_columns(pl.col("season").cast(pl.Utf8).alias("season"))
		.group_by("season")
		.len()
		.sort("season")
	)
	season_row_counts = {row["season"]: int(row["len"]) for row in season_counts_df.to_dicts()}
	test_df = df.filter(pl.col("season").cast(pl.Utf8) == test_season)
	return {
		"row_count": int(df.height),
		"season_row_counts": season_row_counts,
		"data_fingerprint": compute_frame_fingerprint(df),
		"test_season": test_season,
		"test_row_count": int(test_df.height),
	}


def to_loader(
	data: Dict[str, np.ndarray],
	batch_size: int,
	shuffle: bool = True,
	device: torch.device = None,
	seed: int | None = None,
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
	generator = None
	if seed is not None:
		generator = torch.Generator()
		generator.manual_seed(seed)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		generator=generator,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)


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
	budget_metrics = evaluate_budget_strategy(
		probs=implied_probs,
		y_true=y_true,
		odds_home=data["odds_home"],
		odds_draw=data["odds_draw"],
		odds_away=data["odds_away"],
		groups=data.get("dates"),
		strategy=DEFAULT_BUDGET_STRATEGY,
		kelly_fraction=DEFAULT_KELLY_FRACTION,
	)
	return {
		"accuracy": float(acc),
		"brier": brier,
		"rps": float(rps),
		"log_loss": float(ll),
		**budget_metrics,
	}
