"""
Run a canonical-style evaluation and anti-leakage test suite for the player-set model.

The current default target is the Deep Sets encoder because the Set Transformer
variant has not been run locally yet.
"""

from __future__ import annotations

import argparse
from csv import DictReader, DictWriter
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.lineup_builder import _compute_player_state_history, assemble_squad_tensors, build_projected_squads
from preprocessing.player_feature_engineering import compute_player_rolling_features, load_all_player_data, prepare_player_data
from training.evaluation.metrics import evaluate_model, evaluate_profit, ranked_probability_score
from training.inference import forward_model
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.train_player_model import (
	DEFAULT_CONFIG,
	PlayerMatchDataset,
	build_player_model,
	build_squad_data,
	evaluate,
	prepare_match_data,
	set_seed,
	shuffle_squad_features,
	split_by_seasons,
	train_one_epoch,
)
from training.train_utils import (
	CAT_COLS,
	build_data_snapshot,
	evaluate_implied_baseline,
	extract_categorical_features,
	generate_rolling_cv_folds,
	load_frame,
	prepare_data,
	resolve_test_season,
)
from utils.paths import PROJECT_ROOT, TRACKED_ASSETS_DIR
from utils.portfolio import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION, evaluate_bankroll_strategy

DEFAULT_PARQUET = PROJECT_ROOT / "data" / "training" / "understat_df.parquet"
DEFAULT_OUTPUT_PATH = TRACKED_ASSETS_DIR / "tmp" / "player_model_evaluation.json"
EVALUATION_CONFIG_PATH = PROJECT_ROOT / "training" / "configs" / "main_models" / "evaluation.json"
LATEST_MAIN_METRICS_PATH = PROJECT_ROOT / "artifacts" / "models" / "latest_main_model_metrics.json"
DEEP_SETS_EXPERIMENT_LOG_PATH = PROJECT_ROOT / "artifacts" / "experiment_metrics" / "deep_sets_runs.tsv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


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


def load_tsv_rows(path: Path) -> list[dict]:
	if not path.exists() or path.stat().st_size == 0:
		return []
	with open(path, "r", encoding="utf-8", newline="") as file:
		return list(DictReader(file, delimiter="\t"))


def find_latest_kept_row(rows: list[dict]) -> dict | None:
	for row in reversed(rows):
		if row.get("status") == "keep":
			return row
	return None


def run_git_command(*args: str) -> str:
	result = subprocess.run(
		args,
		cwd=PROJECT_ROOT,
		check=True,
		capture_output=True,
		text=True,
	)
	return result.stdout.strip()


def append_deep_sets_experiment_log(summary: dict, description: str, ledger_path: Path) -> dict:
	if "," in description:
		raise ValueError("Deep Sets ledger descriptions must not contain commas.")
	rows = load_tsv_rows(ledger_path)
	reference_row = find_latest_kept_row(rows)
	objective_metrics = summary["player_model"]["objective_metrics"]
	test_metrics = summary["player_model"]["test_metrics"]
	validation_metrics = summary["player_model"]["validation_metrics"]
	cv_log_loss = float(objective_metrics["log_loss"])
	reference_cv = float(reference_row["cv_log_loss"]) if reference_row and reference_row.get("cv_log_loss") else None
	row = {
		"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
		"git_commit": run_git_command("git", "rev-parse", "--short", "HEAD"),
		"git_branch": run_git_command("git", "branch", "--show-current"),
		"cv_log_loss": cv_log_loss,
		"delta": (reference_cv - cv_log_loss) if reference_cv is not None else "",
		"best_epoch": int(summary["player_model"]["best_epoch"]),
		"status": "",
		"description": description,
		"cv_rps": float(objective_metrics["rps"]),
		"val_log_loss": float(validation_metrics["log_loss"]),
		"test_log_loss": float(test_metrics["log_loss"]),
		"cv_metrics_json": objective_metrics,
		"test_metrics_json": test_metrics,
	}
	append_tsv_row(ledger_path, row)
	return {
		"ledger_path": str(ledger_path),
		"reference_cv_log_loss": reference_cv,
		"delta_vs_latest_keep": (reference_cv - cv_log_loss) if reference_cv is not None else None,
		"description": description,
	}


def summarize_metrics(metrics: dict) -> dict:
	keys = [
		"accuracy",
		"brier",
		"rps",
		"log_loss",
		"corr_with_implied",
		"n_bets",
		"percent_bets",
		"bet_win_rate",
		"bankroll_roi",
		"bankroll_bet_count",
		"max_drawdown",
	]
	return {key: metrics[key] for key in keys if key in metrics}


def mean_metric(metrics_list: list[dict]) -> dict:
	if not metrics_list:
		return {}
	keys = metrics_list[0].keys()
	return {
		key: float(np.mean([metrics[key] for metrics in metrics_list]))
		for key in keys
		if all(isinstance(metrics.get(key), (int, float)) for metrics in metrics_list)
	}


def clip_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	clipped = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
	return clipped / clipped.sum(axis=1, keepdims=True)


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
	probs = clip_probs(probs)
	preds = np.argmax(probs, axis=1)
	one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
	accuracy = float(np.mean(preds == y_true))
	brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
	logloss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))
	return {
		"accuracy": accuracy,
		"brier": brier,
		"log_loss": logloss,
	}


def adaptive_bin_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
	quantiles = np.linspace(0.0, 1.0, n_bins + 1)
	edges = np.quantile(confidences, quantiles)
	edges[0] = 0.0
	edges[-1] = 1.0
	edges = np.maximum.accumulate(edges)
	unique_edges = np.unique(edges)
	if len(unique_edges) < 2:
		return np.array([0.0, 1.0], dtype=float)
	unique_edges[0] = 0.0
	unique_edges[-1] = 1.0
	return unique_edges


def classwise_adaptive_ece(y_true: np.ndarray, probs: np.ndarray, class_index: int, n_bins: int = 15) -> dict:
	confidences = np.asarray(probs[:, class_index], dtype=float)
	target = (y_true == class_index).astype(float)
	edges = adaptive_bin_edges(confidences, n_bins=n_bins)
	bins = []
	weighted_gap = 0.0
	for idx in range(len(edges) - 1):
		left = float(edges[idx])
		right = float(edges[idx + 1])
		if idx == len(edges) - 2:
			mask = (confidences >= left) & (confidences <= right)
		else:
			mask = (confidences >= left) & (confidences < right)
		count = int(mask.sum())
		if count == 0:
			continue
		mean_conf = float(confidences[mask].mean())
		emp_freq = float(target[mask].mean())
		gap = abs(emp_freq - mean_conf)
		weight = count / len(confidences)
		weighted_gap += weight * gap
		bins.append({
			"bin_index": len(bins) + 1,
			"left": left,
			"right": right,
			"count": count,
			"mean_confidence": mean_conf,
			"empirical_frequency": emp_freq,
			"gap": float(gap),
		})
	return {
		"class_index": int(class_index),
		"sample_count": int(len(confidences)),
		"bin_count": int(len(bins)),
		"adaptive_ece": float(weighted_gap),
		"max_gap": float(max((row["gap"] for row in bins), default=0.0)),
		"bins": bins,
	}


def summarize_calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
	probs = clip_probs(probs)
	class_rows = [classwise_adaptive_ece(y_true, probs, class_index=i, n_bins=n_bins) for i in range(probs.shape[1])]
	return {
		"n_bins_requested": int(n_bins),
		"macro_adaptive_ece": float(np.mean([row["adaptive_ece"] for row in class_rows])),
		"weighted_adaptive_ece": float(np.average([row["adaptive_ece"] for row in class_rows])),
		"classes": class_rows,
	}


def build_calibration_comparison(y_true: np.ndarray, player_probs: np.ndarray, implied_probs: np.ndarray, main_probs: np.ndarray) -> dict:
	player_metrics = multiclass_metrics(y_true, player_probs)
	implied_metrics = multiclass_metrics(y_true, implied_probs)
	main_metrics = multiclass_metrics(y_true, main_probs)
	player_calibration = summarize_calibration(y_true, player_probs)
	implied_calibration = summarize_calibration(y_true, implied_probs)
	main_calibration = summarize_calibration(y_true, main_probs)
	return {
		"player": {
			"metrics": player_metrics,
			"calibration": player_calibration,
		},
		"implied": {
			"metrics": implied_metrics,
			"calibration": implied_calibration,
		},
		"main": {
			"metrics": main_metrics,
			"calibration": main_calibration,
		},
		"comparison": {
			"aligned_sample_count": int(len(y_true)),
			"player_minus_implied_log_loss": float(player_metrics["log_loss"] - implied_metrics["log_loss"]),
			"player_minus_main_log_loss": float(player_metrics["log_loss"] - main_metrics["log_loss"]),
			"player_minus_implied_brier": float(player_metrics["brier"] - implied_metrics["brier"]),
			"player_minus_main_brier": float(player_metrics["brier"] - main_metrics["brier"]),
			"player_minus_implied_macro_adaptive_ece": float(player_calibration["macro_adaptive_ece"] - implied_calibration["macro_adaptive_ece"]),
			"player_minus_main_macro_adaptive_ece": float(player_calibration["macro_adaptive_ece"] - main_calibration["macro_adaptive_ece"]),
		}
	}


def evaluate_player_probs(probs: np.ndarray, data: dict) -> dict:
	y_true = data["y"]
	preds = np.argmax(probs, axis=1)
	accuracy = float(accuracy_score(y_true, preds))
	one_hot = np.eye(3)[y_true]
	brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
	ll = float(log_loss(y_true, probs, labels=[0, 1, 2]))
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
		"log_loss": ll,
		"corr_with_implied": float(np.mean(corr_per_class)),
		**profit_metrics,
		**bankroll_metrics,
	}


def build_loader(dataset: PlayerMatchDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
	generator = torch.Generator()
	generator.manual_seed(seed)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def shuffled_implied_probs(implied: np.ndarray, seed: int) -> np.ndarray:
	shuffled = np.asarray(implied, dtype=float).copy()
	rng = np.random.RandomState(seed)
	rng.shuffle(shuffled)
	return shuffled


def train_player_with_early_stopping(
	config: dict,
	squad_tensors: dict,
	train_idx: np.ndarray,
	train_data: dict,
	val_idx: np.ndarray,
	val_data: dict,
	seed_offset: int = 0,
) -> tuple[torch.nn.Module, int, float]:
	set_seed(config["seed"] + seed_offset)
	model = build_player_model(config).to(DEVICE)
	train_dataset = PlayerMatchDataset(squad_tensors, train_data["y"], train_data["implied"], train_data["raw_margin"], train_idx)
	val_dataset = PlayerMatchDataset(squad_tensors, val_data["y"], val_data["implied"], val_data["raw_margin"], val_idx)
	train_loader = build_loader(train_dataset, config["batch_size"], shuffle=True, seed=config["seed"] + seed_offset)
	val_loader = build_loader(val_dataset, config["batch_size"], shuffle=False, seed=config["seed"] + seed_offset)
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
		train_one_epoch(model, train_loader, optimizer, DEVICE, config)
		val_loss, _ = evaluate(model, val_loader, DEVICE)
		scheduler.step()
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch
			patience_counter = 0
			best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
		else:
			patience_counter += 1
			if patience_counter >= config["patience"]:
				break
	if best_state is not None:
		model.load_state_dict(best_state)
		model.to(DEVICE)
	return model, best_epoch, float(best_val_loss)


def train_player_fixed_epochs(
	config: dict,
	squad_tensors: dict,
	train_idx: np.ndarray,
	train_data: dict,
	epochs: int,
	labels: np.ndarray | None = None,
	seed_offset: int = 0,
) -> torch.nn.Module:
	set_seed(config["seed"] + seed_offset)
	model = build_player_model(config).to(DEVICE)
	train_labels = train_data["y"] if labels is None else labels
	train_dataset = PlayerMatchDataset(squad_tensors, train_labels, train_data["implied"], train_data["raw_margin"], train_idx)
	train_loader = build_loader(train_dataset, config["batch_size"], shuffle=True, seed=config["seed"] + seed_offset)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=max(1, epochs),
		eta_min=config["lr"] * 0.01,
	)
	for _ in range(max(1, epochs)):
		train_one_epoch(model, train_loader, optimizer, DEVICE, config)
		scheduler.step()
	return model


def predict_player_model_probs_on_split(
	model: torch.nn.Module,
	squad_tensors: dict,
	data: dict,
	indices: np.ndarray,
	implied_override: np.ndarray | None = None,
) -> np.ndarray:
	feed_implied = data["implied"] if implied_override is None else implied_override
	dataset = PlayerMatchDataset(squad_tensors, data["y"], feed_implied, data["raw_margin"], indices)
	loader = build_loader(dataset, batch_size=512, shuffle=False, seed=0)
	_, probs = evaluate(model, loader, DEVICE)
	return probs


def evaluate_player_model_on_split(
	model: torch.nn.Module,
	squad_tensors: dict,
	data: dict,
	indices: np.ndarray,
	implied_override: np.ndarray | None = None,
) -> dict:
	probs = predict_player_model_probs_on_split(model, squad_tensors, data, indices, implied_override=implied_override)
	return summarize_metrics(evaluate_player_probs(probs, data))


def predict_main_model_probs(test_data: dict, model: torch.nn.Module) -> np.ndarray:
	X = torch.tensor(test_data["X"], dtype=torch.float32)
	cat_features = torch.tensor(test_data["cat_features"], dtype=torch.long)
	implied = torch.tensor(test_data["implied"], dtype=torch.float32)
	raw_margin = torch.tensor(test_data["raw_margin"], dtype=torch.float32)
	with torch.no_grad():
		logits = forward_model(model, X, cat_features, implied, raw_margin)
		return torch.softmax(logits, dim=-1).cpu().numpy()


def extract_aligned_game_ids(aligned_df: pl.DataFrame, indices: np.ndarray) -> np.ndarray:
	return aligned_df.filter(pl.col("_tensor_idx").is_in(indices)).sort("_tensor_idx")["game_id"].to_numpy()


def prepare_main_model_test_data(df: pl.DataFrame, feature_cols: list[str], scaler, test_season: str) -> tuple[dict, np.ndarray]:
	req_cols = list({
		"game_id",
		"result_label",
		"implied_home",
		"implied_draw",
		"implied_away",
		"odds_home",
		"odds_draw",
		"odds_away",
		"date",
		"raw_margin",
	} | set(CAT_COLS))
	part = df.filter(pl.col("season").cast(pl.Utf8) == test_season)
	part = part.filter(
		(pl.col("odds_home") > 1.0)
		& (pl.col("odds_draw") > 1.0)
		& (pl.col("odds_away") > 1.0)
		& pl.col("implied_home").is_finite()
		& pl.col("implied_draw").is_finite()
		& pl.col("implied_away").is_finite()
	)
	part = part.drop_nulls(subset=req_cols)
	part = part.drop_nulls(subset=feature_cols)
	feature_frame = part.select([pl.col(col).cast(pl.Float64).alias(col) for col in feature_cols])
	X = feature_frame.to_pandas().to_numpy(dtype=np.float64)
	X = scaler.transform(X)
	data = {
		"X": X,
		"y": part["result_label"].to_numpy().astype(int),
		"implied": np.stack([
			part["implied_home"].to_numpy(),
			part["implied_draw"].to_numpy(),
			part["implied_away"].to_numpy(),
		], axis=1).astype(np.float64),
		"cat_features": extract_categorical_features(part),
		"odds_home": part["odds_home"].to_numpy(),
		"odds_draw": part["odds_draw"].to_numpy(),
		"odds_away": part["odds_away"].to_numpy(),
		"raw_margin": part["raw_margin"].to_numpy().astype(np.float64),
		"dates": part["date"].to_numpy(),
	}
	return data, part["game_id"].to_numpy()


def align_row_indices(
	primary_game_ids: np.ndarray,
	secondary_game_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	secondary_row_by_game = {game_id: idx for idx, game_id in enumerate(secondary_game_ids)}
	primary_rows = [idx for idx, game_id in enumerate(primary_game_ids) if game_id in secondary_row_by_game]
	secondary_rows = [secondary_row_by_game[primary_game_ids[idx]] for idx in primary_rows]
	return np.asarray(primary_rows, dtype=int), np.asarray(secondary_rows, dtype=int)


def slice_data_rows(data: dict, rows: np.ndarray) -> dict:
	sliced = {}
	n_rows = len(data["y"])
	for key, value in data.items():
		if isinstance(value, np.ndarray) and value.shape[0] == n_rows:
			sliced[key] = value[rows]
		else:
			sliced[key] = value
	return sliced


def compute_overlap(raw_player_data: pl.DataFrame, squads: pl.DataFrame, season: str) -> dict:
	prepared = prepare_player_data(raw_player_data)
	actual = prepared.filter(pl.col("season").cast(pl.Utf8) == season)
	projected = squads.filter(pl.col("season").cast(pl.Utf8) == season)
	overlaps = []
	for (game_id, team_id), group_df in actual.group_by(["game_id", "team_id"]):
		actual_players = set(group_df["player_id"].to_list())
		projected_players = projected.filter((pl.col("game_id") == game_id) & (pl.col("team_id") == team_id))
		if len(projected_players) == 0 or not actual_players:
			continue
		predicted_players = set(projected_players["player_id"].to_list())
		overlaps.append(len(actual_players & predicted_players) / len(actual_players))
	if not overlaps:
		return {"mean_overlap": 0.0, "perfect_overlap_rate": 0.0, "sample_count": 0}
	overlap_array = np.asarray(overlaps, dtype=float)
	return {
		"mean_overlap": float(overlap_array.mean()),
		"perfect_overlap_rate": float(np.mean(overlap_array == 1.0)),
		"sample_count": int(len(overlap_array)),
	}


def build_frozen_squads(raw_player_data: pl.DataFrame, rolling: pl.DataFrame, top_n: int) -> pl.DataFrame:
	_ = rolling
	prepared = prepare_player_data(raw_player_data)
	player_state_history = _compute_player_state_history(raw_player_data).sort(["league", "team_id", "player_id", "date"])
	seasons = sorted(prepared["season"].cast(pl.Utf8).unique().to_list())
	all_frozen = []
	for season_idx, season in enumerate(seasons):
		if season_idx == 0:
			continue
		previous_season = seasons[season_idx - 1]
		previous_data = prepared.filter(pl.col("season").cast(pl.Utf8) == previous_season)
		previous_minutes = previous_data.group_by(["league", "team_id", "player_id"]).agg(
			pl.col("minutes").sum().alias("previous_season_minutes")
		)
		previous_minutes = previous_minutes.sort(
			["league", "team_id", "previous_season_minutes"],
			descending=[False, False, True],
		)
		previous_minutes = previous_minutes.with_columns(
			pl.col("previous_season_minutes")
			.rank(method="ordinal", descending=True)
			.over(["league", "team_id"])
			.alias("squad_rank")
		)
		top_players = previous_minutes.filter(pl.col("squad_rank") <= top_n)
		season_match_keys = prepared.filter(pl.col("season").cast(pl.Utf8) == season).select(
			["league", "season", "team_id", "game_id", "date"]
		).unique()
		frozen = season_match_keys.join(
			top_players.select(["league", "team_id", "player_id", "squad_rank"]),
			on=["league", "team_id"],
			how="inner",
		)
		frozen = frozen.sort(["league", "team_id", "player_id", "date"])
		frozen = frozen.join_asof(
			player_state_history,
			on="date",
			by=["league", "team_id", "player_id"],
			strategy="backward",
			allow_exact_matches=False,
		)
		frozen = frozen.filter(pl.col("avg_minutes_r10").is_not_null())
		all_frozen.append(frozen)
	return pl.concat(all_frozen, how="diagonal")


def evaluate_main_model_reference(df: pl.DataFrame, test_season: str) -> dict:
	bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=torch.device("cpu"))
	test_data = prepare_data(df, bundle.feature_cols, [test_season], scaler=bundle.scaler)
	metrics = summarize_metrics(evaluate_model(bundle.model, test_data, device=torch.device("cpu"), verbose=False))
	latest_main_metrics = load_json(LATEST_MAIN_METRICS_PATH)["model"]
	return {
		"bundle_git_commit": latest_main_metrics.get("git_commit"),
		"bundle_git_branch": latest_main_metrics.get("git_branch"),
		"recorded_objective_metrics": latest_main_metrics.get("objective_metrics", {}),
		"recorded_test_metrics": latest_main_metrics.get("test_metrics", {}),
		"live_test_metrics": metrics,
		"data_snapshot": latest_main_metrics.get("data_snapshot", {}),
	}


def run_top_n_sensitivity(
	config: dict,
	raw_player_data: pl.DataFrame,
	rolling: pl.DataFrame,
	match_df: pl.DataFrame,
	train_seasons: list[str],
	test_season: str,
	best_epoch: int,
	top_n_values: list[int],
) -> list[dict]:
	rows = []
	for top_n in top_n_values:
		squads = build_projected_squads(raw_player_data, rolling, match_df=match_df, top_n=top_n)
		tensors = assemble_squad_tensors(squads, match_df, max_players=top_n)
		game_id_order = pl.DataFrame({"game_id": tensors["game_ids"], "_tensor_idx": range(len(tensors["game_ids"]))})
		aligned_df = match_df.join(game_id_order, on="game_id", how="inner").sort("_tensor_idx")
		train_idx, _, train_data, _ = split_by_seasons(aligned_df, train_seasons, [])
		_, test_idx, _, test_data = split_by_seasons(aligned_df, train_seasons, [test_season])
		candidate_config = dict(config)
		candidate_config["top_n_players"] = top_n
		model = train_player_fixed_epochs(candidate_config, tensors, train_idx, train_data, epochs=best_epoch, seed_offset=200 + top_n)
		metrics = evaluate_player_model_on_split(model, tensors, test_data, test_idx)
		rows.append({
			"top_n": top_n,
			"overlap": compute_overlap(raw_player_data, squads, season=test_season),
			"test_metrics": metrics,
			"baseline_metrics": summarize_metrics(evaluate_implied_baseline(test_data)),
		})
	return rows


def build_summary(args) -> dict:
	config = dict(DEFAULT_CONFIG)
	config["encoder_type"] = args.encoder
	config["top_n_players"] = args.top_n_players
	config["max_epochs"] = args.max_epochs
	config["batch_size"] = args.batch_size
	config["use_implied"] = not args.no_implied
	config["head_type"] = args.head_type
	config["mlp_market_features"] = bool(args.mlp_market_features)
	config["linear_residual_head"] = bool(args.linear_residual_head)
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

	evaluation_config = load_json(EVALUATION_CONFIG_PATH)
	set_seed(config["seed"])
	df = load_frame(args.parquet_path)
	df = prepare_match_data(df)
	data_snapshot = build_data_snapshot(df, resolve_test_season(df, evaluation_config["test_season"]))
	squad_tensors, aligned_df = build_squad_data(df, top_n=config["top_n_players"])
	test_season = resolve_test_season(aligned_df, evaluation_config["test_season"])
	folds = generate_rolling_cv_folds(aligned_df, n_folds=evaluation_config["rolling_cv_n_folds"], test_season=test_season)
	objective_folds = folds[:-1]
	epoch_train_seasons, epoch_selection_season = folds[-1]
	all_cv_seasons = sorted({season for train_seasons, val_season in folds for season in [*train_seasons, val_season]})

	train_idx, val_idx, train_data, val_data = split_by_seasons(aligned_df, epoch_train_seasons, [epoch_selection_season])
	_, test_idx, _, test_data = split_by_seasons(aligned_df, all_cv_seasons, [test_season])
	early_stop_model, best_epoch, best_val_loss = train_player_with_early_stopping(
		config,
		squad_tensors,
		train_idx,
		train_data,
		val_idx,
		val_data,
	)
	validation_metrics = evaluate_player_model_on_split(early_stop_model, squad_tensors, val_data, val_idx)
	validation_baseline_metrics = summarize_metrics(evaluate_implied_baseline(val_data))

	objective_rows = []
	objective_metrics = []
	objective_baselines = []
	for fold_index, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		fold_train_idx, fold_val_idx, fold_train_data, fold_val_data = split_by_seasons(aligned_df, train_seasons, [val_season])
		fold_model = train_player_fixed_epochs(
			config,
			squad_tensors,
			fold_train_idx,
			fold_train_data,
			epochs=best_epoch,
			seed_offset=fold_index,
		)
		fold_metrics = evaluate_player_model_on_split(fold_model, squad_tensors, fold_val_data, fold_val_idx)
		fold_baseline = summarize_metrics(evaluate_implied_baseline(fold_val_data))
		objective_metrics.append(fold_metrics)
		objective_baselines.append(fold_baseline)
		objective_rows.append({
			"fold_index": fold_index,
			"train_seasons": train_seasons,
			"val_season": val_season,
			"metrics": fold_metrics,
			"baseline_metrics": fold_baseline,
		})

	all_train_idx, _, all_train_data, _ = split_by_seasons(aligned_df, all_cv_seasons, [])
	final_model = train_player_fixed_epochs(
		config,
		squad_tensors,
		all_train_idx,
		all_train_data,
		epochs=best_epoch,
		seed_offset=10_000,
	)
	test_metrics = evaluate_player_model_on_split(final_model, squad_tensors, test_data, test_idx)
	test_probs = predict_player_model_probs_on_split(final_model, squad_tensors, test_data, test_idx)
	test_game_ids = extract_aligned_game_ids(aligned_df, test_idx)
	test_baseline_metrics = summarize_metrics(evaluate_implied_baseline(test_data))
	main_bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=torch.device("cpu"))
	main_test_data, main_test_game_ids = prepare_main_model_test_data(df, main_bundle.feature_cols, main_bundle.scaler, test_season)
	main_test_probs = predict_main_model_probs(main_test_data, main_bundle.model)
	main_reference = evaluate_main_model_reference(df, test_season=test_season)
	player_rows, main_rows = align_row_indices(test_game_ids, main_test_game_ids)
	aligned_test_data = slice_data_rows(test_data, player_rows)
	aligned_main_test_data = slice_data_rows(main_test_data, main_rows)
	aligned_y = aligned_test_data["y"]
	aligned_player_probs = test_probs[player_rows]
	aligned_implied_probs = aligned_test_data["implied"]
	aligned_main_probs = main_test_probs[main_rows]
	if not np.array_equal(aligned_y, aligned_main_test_data["y"]):
		raise ValueError("Aligned player and main test labels do not match by game_id.")
	calibration_comparison = build_calibration_comparison(aligned_y, aligned_player_probs, aligned_implied_probs, aligned_main_probs)
	common_player_metrics = summarize_metrics(evaluate_player_probs(aligned_player_probs, aligned_test_data))
	common_implied_metrics = summarize_metrics(evaluate_implied_baseline(aligned_test_data))
	common_main_metrics = summarize_metrics(evaluate_player_probs(aligned_main_probs, aligned_main_test_data))
	shuffled_implied_test = shuffled_implied_probs(test_data["implied"], seed=config["seed"])
	shuffled_implied_test_metrics = evaluate_player_model_on_split(
		final_model,
		squad_tensors,
		test_data,
		test_idx,
		implied_override=shuffled_implied_test,
	)

	feature_permutation_tensors = shuffle_squad_features(squad_tensors, seed=config["seed"])
	permuted_model = train_player_fixed_epochs(
		config,
		feature_permutation_tensors,
		all_train_idx,
		all_train_data,
		epochs=best_epoch,
		seed_offset=20_000,
	)
	permutation_metrics = evaluate_player_model_on_split(permuted_model, feature_permutation_tensors, test_data, test_idx)

	rng = np.random.RandomState(config["seed"])
	shuffled_labels = all_train_data["y"].copy()
	rng.shuffle(shuffled_labels)
	label_shuffle_model = train_player_fixed_epochs(
		config,
		squad_tensors,
		all_train_idx,
		all_train_data,
		epochs=best_epoch,
		labels=shuffled_labels,
		seed_offset=30_000,
	)
	label_shuffle_metrics = evaluate_player_model_on_split(label_shuffle_model, squad_tensors, test_data, test_idx)

	raw_player_data = load_all_player_data()
	rolling = compute_player_rolling_features(raw_player_data)
	frozen_squads = build_frozen_squads(raw_player_data, rolling, top_n=config["top_n_players"])
	frozen_tensors = assemble_squad_tensors(frozen_squads, df, max_players=config["top_n_players"])
	frozen_game_order = pl.DataFrame({"game_id": frozen_tensors["game_ids"], "_tensor_idx": range(len(frozen_tensors["game_ids"]))})
	frozen_aligned_df = df.join(frozen_game_order, on="game_id", how="inner").sort("_tensor_idx")
	frozen_train_idx, _, frozen_train_data, _ = split_by_seasons(frozen_aligned_df, all_cv_seasons, [])
	_, frozen_test_idx, _, frozen_test_data = split_by_seasons(frozen_aligned_df, all_cv_seasons, [test_season])
	frozen_model = train_player_fixed_epochs(
		config,
		frozen_tensors,
		frozen_train_idx,
		frozen_train_data,
		epochs=best_epoch,
		seed_offset=40_000,
	)
	frozen_metrics = evaluate_player_model_on_split(frozen_model, frozen_tensors, frozen_test_data, frozen_test_idx)

	top_n_results = run_top_n_sensitivity(
		config,
		raw_player_data,
		rolling,
		df,
		train_seasons=all_cv_seasons,
		test_season=test_season,
		best_epoch=best_epoch,
		top_n_values=[11, config["top_n_players"], 28],
	)

	dynamic_gain = test_baseline_metrics["log_loss"] - test_metrics["log_loss"]
	permuted_gain = test_baseline_metrics["log_loss"] - permutation_metrics["log_loss"]
	max_top_n_log_loss = max(row["test_metrics"]["log_loss"] for row in top_n_results)

	return {
		"player_model": {
			"config": config,
			"data_snapshot": data_snapshot,
			"best_epoch": best_epoch,
			"best_val_loss": best_val_loss,
			"epoch_selection_season": epoch_selection_season,
			"objective_folds": objective_rows,
			"objective_metrics": mean_metric(objective_metrics),
			"objective_baseline_metrics": mean_metric(objective_baselines),
			"validation_metrics": validation_metrics,
			"validation_baseline_metrics": validation_baseline_metrics,
			"test_metrics": test_metrics,
			"test_baseline_metrics": test_baseline_metrics,
		},
		"comparison": {
			"main_reference": main_reference,
			"common_test_set": {
				"sample_count": int(len(aligned_y)),
				"player_metrics": common_player_metrics,
				"implied_metrics": common_implied_metrics,
				"main_metrics": common_main_metrics,
				"player_minus_implied_log_loss": float(common_player_metrics["log_loss"] - common_implied_metrics["log_loss"]),
				"player_minus_main_log_loss": float(common_player_metrics["log_loss"] - common_main_metrics["log_loss"]),
				"player_minus_implied_bankroll_roi": float(common_player_metrics["bankroll_roi"] - common_implied_metrics["bankroll_roi"]),
				"player_minus_main_bankroll_roi": float(common_player_metrics["bankroll_roi"] - common_main_metrics["bankroll_roi"]),
				"main_minus_implied_log_loss": float(common_main_metrics["log_loss"] - common_implied_metrics["log_loss"]),
				"main_minus_implied_bankroll_roi": float(common_main_metrics["bankroll_roi"] - common_implied_metrics["bankroll_roi"]),
			},
			"calibration": calibration_comparison,
			"objective_log_loss_delta_vs_main_recorded": float(mean_metric(objective_metrics)["log_loss"] - main_reference["recorded_objective_metrics"]["log_loss"]),
			"test_log_loss_delta_vs_main_live": float(test_metrics["log_loss"] - main_reference["live_test_metrics"]["log_loss"]),
			"test_bankroll_roi_delta_vs_main_live": float(test_metrics["bankroll_roi"] - main_reference["live_test_metrics"]["bankroll_roi"]),
		},
		"anti_leakage": {
			"shuffled_implied_inference": {
				"test_metrics": shuffled_implied_test_metrics,
			},
			"feature_permutation": {
				"test_metrics": permutation_metrics,
				"signal_retained_fraction_vs_dynamic": float(permuted_gain / dynamic_gain) if abs(dynamic_gain) > 1e-12 else None,
			},
			"label_shuffle": {
				"test_metrics": label_shuffle_metrics,
			},
			"frozen_squads": {
				"test_metrics": frozen_metrics,
				"dynamic_test_metrics": test_metrics,
				"dynamic_overlap": compute_overlap(raw_player_data, build_projected_squads(raw_player_data, rolling, match_df=df, top_n=config["top_n_players"]), season=test_season),
				"frozen_overlap": compute_overlap(raw_player_data, frozen_squads, season=test_season),
			},
			"top_n_sensitivity": top_n_results,
		},
		"checks": {
			"beats_implied_on_test_log_loss": bool(test_metrics["log_loss"] < test_baseline_metrics["log_loss"]),
			"permutation_destroys_most_signal": bool(permuted_gain < 0.5 * dynamic_gain) if dynamic_gain > 0 else False,
			"label_shuffle_fails_to_beat_implied": bool(label_shuffle_metrics["log_loss"] >= test_baseline_metrics["log_loss"]),
			"highest_top_n_still_beats_implied": bool(max_top_n_log_loss < test_baseline_metrics["log_loss"]),
		},
	}


def print_summary(summary: dict):
	player = summary["player_model"]
	comparison = summary["comparison"]
	common = comparison["common_test_set"]
	leakage = summary["anti_leakage"]
	print("=" * 72)
	print("PLAYER MODEL EVALUATION")
	print("=" * 72)
	print(f"Encoder: {player['config']['encoder_type']}")
	print(f"Best epoch: {player['best_epoch']} | Best val loss: {player['best_val_loss']:.5f}")
	print(f"Objective CV log_loss: {player['objective_metrics']['log_loss']:.6f}")
	print(f"Objective baseline log_loss: {player['objective_baseline_metrics']['log_loss']:.6f}")
	print(f"Held-out test log_loss: {player['test_metrics']['log_loss']:.6f}")
	print(f"Held-out implied log_loss: {player['test_baseline_metrics']['log_loss']:.6f}")
	print(f"Held-out bankroll ROI: {player['test_metrics']['bankroll_roi']:.4f}")
	print(f"Held-out bet count: {player['test_metrics']['bankroll_bet_count']}")
	print("")
	print("Calibration vs implied and current best main model")
	print(f"  Player macro aECE: {comparison['calibration']['player']['calibration']['macro_adaptive_ece']:.6f}")
	print(f"  Implied macro aECE: {comparison['calibration']['implied']['calibration']['macro_adaptive_ece']:.6f}")
	print(f"  Main macro aECE: {comparison['calibration']['main']['calibration']['macro_adaptive_ece']:.6f}")
	print(f"  Player minus implied macro aECE: {comparison['calibration']['comparison']['player_minus_implied_macro_adaptive_ece']:+.6f}")
	print(f"  Player minus main macro aECE: {comparison['calibration']['comparison']['player_minus_main_macro_adaptive_ece']:+.6f}")
	print("")
	print("Common held-out test set comparison")
	print(f"  Common sample count: {common['sample_count']}")
	print(f"  Implied log_loss: {common['implied_metrics']['log_loss']:.6f}")
	print(f"  Main log_loss: {common['main_metrics']['log_loss']:.6f}")
	print(f"  Player log_loss: {common['player_metrics']['log_loss']:.6f}")
	print(f"  Implied bankroll ROI: {common['implied_metrics']['bankroll_roi']:.4f}")
	print(f"  Main bankroll ROI: {common['main_metrics']['bankroll_roi']:.4f}")
	print(f"  Player bankroll ROI: {common['player_metrics']['bankroll_roi']:.4f}")
	print(f"  Player minus implied log_loss: {common['player_minus_implied_log_loss']:+.6f}")
	print(f"  Player minus main log_loss: {common['player_minus_main_log_loss']:+.6f}")
	print(f"  Player minus implied ROI: {common['player_minus_implied_bankroll_roi']:+.4f}")
	print(f"  Player minus main ROI: {common['player_minus_main_bankroll_roi']:+.4f}")
	print("")
	print("Comparison vs current best gated model")
	print(f"  Main recorded objective log_loss: {comparison['main_reference']['recorded_objective_metrics']['log_loss']:.6f}")
	print(f"  Main live held-out test log_loss: {comparison['main_reference']['live_test_metrics']['log_loss']:.6f}")
	print(f"  Player minus main objective log_loss: {comparison['objective_log_loss_delta_vs_main_recorded']:+.6f}")
	print(f"  Player minus main held-out test log_loss: {comparison['test_log_loss_delta_vs_main_live']:+.6f}")
	print(f"  Player minus main held-out ROI: {comparison['test_bankroll_roi_delta_vs_main_live']:+.4f}")
	print("")
	print("Leak checks")
	print(f"  Shuffled implied at inference log_loss: {leakage['shuffled_implied_inference']['test_metrics']['log_loss']:.6f}")
	print(f"  Feature permutation test log_loss: {leakage['feature_permutation']['test_metrics']['log_loss']:.6f}")
	print(f"  Feature permutation retained signal fraction: {leakage['feature_permutation']['signal_retained_fraction_vs_dynamic']}")
	print(f"  Label shuffle test log_loss: {leakage['label_shuffle']['test_metrics']['log_loss']:.6f}")
	print(f"  Frozen squad test log_loss: {leakage['frozen_squads']['test_metrics']['log_loss']:.6f}")
	for row in leakage["top_n_sensitivity"]:
		print(
			f"  Top-N {row['top_n']:>2}: overlap={row['overlap']['mean_overlap']:.3f} "
			f"test_log_loss={row['test_metrics']['log_loss']:.6f}"
		)
	print("")
	print("Checks")
	for key, value in summary["checks"].items():
		print(f"  {key}: {value}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
	parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--encoder", choices=["deep_sets", "deep_sets_role_pool", "deep_sets_stats", "weighted_deep_sets", "set_transformer"], default="deep_sets")
	parser.add_argument("--head-type", choices=["mlp", "gated_residual"], default="mlp")
	parser.add_argument("--mlp-market-features", action="store_true")
	parser.add_argument("--linear-residual-head", action="store_true")
	parser.add_argument("--top-n-players", type=int, default=16)
	parser.add_argument("--max-epochs", type=int, default=35)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--no-implied", action="store_true")
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
	parser.add_argument("--description", type=str, help="Short text for appending a tracked Deep Sets experiment run")
	parser.add_argument("--ledger-path", type=Path, default=DEEP_SETS_EXPERIMENT_LOG_PATH)
	args = parser.parse_args()

	summary = build_summary(args)
	if args.description:
		if not str(args.encoder).startswith("deep_sets"):
			raise ValueError("Tracked experiment logging is currently reserved for Deep Sets family runs.")
		summary["experiment_log"] = append_deep_sets_experiment_log(summary, args.description, args.ledger_path)
	args.output_path.parent.mkdir(parents=True, exist_ok=True)
	with open(args.output_path, "w", encoding="utf-8") as file:
		json.dump(summary, file, indent=2)
	print_summary(summary)
	if "experiment_log" in summary:
		log_info = summary["experiment_log"]
		print("")
		print(f"Appended Deep Sets run to {log_info['ledger_path']}")
		print(f"Reference keep cv_log_loss: {log_info['reference_cv_log_loss']}")
		print(f"Delta vs latest keep: {log_info['delta_vs_latest_keep']}")
	print(f"\nWrote summary to {args.output_path}")


if __name__ == "__main__":
	main()
