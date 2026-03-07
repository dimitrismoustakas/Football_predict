"""
Feature Selection Search using Cross-Validation

Searches over feature family combinations with a FIXED architecture to find
the optimal feature set. This avoids the combinatorial explosion of searching
both architecture and features simultaneously.

Strategy:
1. Fix the architecture to a known good configuration (from architecture_search)
2. Use Optuna to explore feature family combinations
3. Evaluate each combination using rolling CV (same as architecture_search)

Usage:
	uv run python training/feature_selection_search.py
"""

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import numpy as np
import optuna
import polars as pl

from training.feature_families import (
	get_features_for_families,
	get_family_columns,
	get_all_families,
)
from training.architecture_search_core import (
	DEVICE,
	MODELS_DIR,
	N_CV_FOLDS,
	TASK_CONFIG,
	set_seed,
	print_header,
)
from training.models import TrainConfig, CategoricalConfig
from training.train_utils import (
	filter_min_history,
	fold_data_to_loaders,
	generate_rolling_cv_folds,
	get_num_leagues,
	load_frame,
	precompute_fold_data,
	train_model,
)

TaskType = Literal["binary", "multiclass"]

# ============================================================================
# CONFIGURATION
# ============================================================================

N_TRIALS = 200
MAX_EPOCHS = 30
PATIENCE = 10

# Core families that are always included (mandatory)
CORE_FAMILIES = ["xg_ovr", "elo"]

# Optional families to search combinations of
OPTIONAL_FAMILIES = [
	"xg_venue",
	"goals_ovr",
	"goals_venue",
	"shots_ovr",
	"shots_venue",
	"pressing_ovr",
	"pressing_venue",
	"form_ovr",
	"form_venue",
	"adjusted",
	"schedule",
	"season_context",
	"player_agg",
	"squad_concentration",
]


# ============================================================================
# UTILITIES
# ============================================================================


def load_fixed_architecture(task_type: TaskType) -> Dict[str, Any]:
	"""Load architecture from existing config file, or use defaults."""
	config_paths = {
		"binary": MODELS_DIR / "architecture_config.json",
		"multiclass": MODELS_DIR / "result_architecture_config.json",
	}
	
	defaults = {
		"binary": {
			"hidden_layers": [256, 256, 128],
			"dropout": 0.15,
			"norm": "ln",
			"lr": 1e-3,
			"weight_decay": 1e-4,
			"activation": "gelu",
			"beta1": 0.9,
			"batch_size": 256,
		},
		"multiclass": {
			"hidden_layers": [512, 256, 128, 128],
			"dropout": 0.29,
			"norm": "none",
			"lr": 2.15e-3,
			"weight_decay": 2.9e-4,
			"activation": "relu",
			"beta1": 0.81,
			"batch_size": 256,
		},
	}
	
	config_path = config_paths[task_type]
	if config_path.exists():
		with open(config_path) as f:
			config = json.load(f)
		print(f"Loaded architecture from {config_path}")
		return {k: config[k] for k in defaults[task_type].keys()}
	
	print(f"No config found at {config_path}, using defaults")
	return defaults[task_type]


def get_available_families(df: pl.DataFrame) -> List[str]:
	"""Get list of families that have at least one column in the dataframe."""
	available = []
	for family_name in get_all_families():
		cols = get_family_columns(family_name)
		if any(c in df.columns for c in cols):
			available.append(family_name)
	return available


def prefilter_nulls_for_all_families(df: pl.DataFrame, families: List[str]) -> pl.DataFrame:
	"""
	Drop rows with nulls in ANY feature from ANY of the specified families.
	
	This ensures all trials work with the same dataset, making comparisons fair.
	Without this, different feature combinations would have different sample sizes.
	
	Args:
		df: Input dataframe
		families: All families that will be searched over (core + optional)
	
	Returns:
		DataFrame with nulls dropped for union of all family features
	"""
	# Get union of all possible features
	all_features = select_best_window_per_feature(get_features_for_families(df, families))
	
	print(f"Pre-filtering nulls across {len(all_features)} features from {len(families)} families...")
	n_before = len(df)
	df = df.drop_nulls(subset=all_features)
	n_dropped = n_before - len(df)
	
	if n_dropped > 0:
		print(f"Dropped {n_dropped} rows ({n_dropped/n_before:.1%}) with missing feature values")
	else:
		print("No rows dropped (no missing feature values)")
	
	return df


def select_best_window_per_feature(feature_cols: List[str]) -> List[str]:
	"""
	For each base feature, select the finest available window.
	
	Strategy:
	- Prefer r5 if available
	- Fall back to r10 if r5 doesn't exist
	- Fall back to r15 if neither r5 nor r10 exist
	- Keep non-rolling features (elo, schedule, etc.) that don't have window indicators
	
	Examples:
	  - For xg_ovr family (has r3/r5/r10): Keep only r5 versions
	  - For adjusted family (only has r10): Keep r10 versions
	  - For player_agg family (only has r15): Keep r15 versions
	  - For elo/schedule (no windows): Keep as-is
	"""
	# Group features by their base pattern (remove window suffix)
	from collections import defaultdict
	feature_groups = defaultdict(list)
	
	for col in feature_cols:
		# Extract base pattern by removing window indicator (r3, r5, r10, r15)
		# Pattern: scope__stat__[sum__]r{window}__side → scope__stat__[sum__]__side
		if "__r" in col:
			# Find window pattern
			import re
			match = re.search(r'__r(\d+)', col)
			if match:
				window = int(match.group(1))
				# Create base key without window
				base_key = col.replace(f"__r{window}", "__rX")
				feature_groups[base_key].append((window, col))
		else:
			# Non-rolling feature - keep as-is
			feature_groups[col].append((0, col))
	
	# Select best window for each base feature
	selected = []
	for base_key, features in feature_groups.items():
		# Sort by window (ascending) - prefer smallest available window
		features.sort(key=lambda x: x[0])
		
		# For rolling features, prefer: r5 > r10 > r15 > r3
		if features[0][0] > 0:  # Has windows
			windows = [w for w, _ in features]
			if 5 in windows:
				selected.append([col for w, col in features if w == 5][0])
			elif 10 in windows:
				selected.append([col for w, col in features if w == 10][0])
			elif 15 in windows:
				selected.append([col for w, col in features if w == 15][0])
			elif 3 in windows:
				selected.append([col for w, col in features if w == 3][0])
		else:
			# Non-rolling feature
			selected.append(features[0][1])
	
	return selected


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_feature_set(
	df: pl.DataFrame,
	folds: List[Tuple[List[str], str]],
	feature_cols: List[str],
	task_type: TaskType,
	arch_config: Dict[str, Any],
	verbose: bool = False,
) -> Tuple[float, List[float]]:
	"""Evaluate a feature set using CV with fixed architecture."""
	# Reuse existing infrastructure - handles categoricals, scaling, etc.
	fold_data = precompute_fold_data(df, feature_cols, folds, task_type=task_type)
	input_dim = fold_data[0]["X_train"].shape[1]
	
	# Categorical config if available
	cat_config = None
	if "league_idx" in df.columns:
		num_leagues = get_num_leagues(df)
		cat_config = CategoricalConfig(num_leagues=num_leagues, league_embed_dim=4)
	
	config = TrainConfig(
		input_dim=input_dim,
		hidden_layers=arch_config["hidden_layers"],
		dropout=arch_config["dropout"],
		norm=arch_config["norm"],
		lr=arch_config["lr"],
		weight_decay=arch_config["weight_decay"],
		lambda_repulsion=0.0,
		lambda_corr=0.0,
		activation=arch_config["activation"],
		beta1=arch_config["beta1"],
		epochs=MAX_EPOCHS,
		patience=PATIENCE,
		batch_size=arch_config["batch_size"],
		task_type=task_type,
		cat_config=cat_config,
	)
	
	fold_losses = []
	for fold in fold_data:
		train_loader, val_loader = fold_data_to_loaders(fold, config.batch_size, task_type=task_type)
		_, _, best_val_loss = train_model(config, train_loader, val_loader, device=DEVICE, verbose=verbose)
		fold_losses.append(best_val_loss)
	
	return float(np.mean(fold_losses)), fold_losses


# ============================================================================
# SEARCH
# ============================================================================


def create_objective(
	df: pl.DataFrame,
	folds: List[Tuple[List[str], str]],
	task_type: TaskType,
	available_optional: List[str],
	arch_config: Dict[str, Any],
):
	"""Create Optuna objective for feature family selection."""
	
	def objective(trial: optuna.Trial) -> float:
		# Always include core families
		selected = list(CORE_FAMILIES)
		
		# Binary decision for each optional family
		for family in available_optional:
			if trial.suggest_categorical(f"include_{family}", [True, False]):
				selected.append(family)
		
		# Get best window for each feature (r5 preferred, fallback to r10/r15)
		feature_cols = select_best_window_per_feature(get_features_for_families(df, selected))
		
		trial.set_user_attr("families", selected)
		trial.set_user_attr("n_features", len(feature_cols))
		
		set_seed(42)
		mean_loss, fold_losses = evaluate_feature_set(df, folds, feature_cols, task_type, arch_config)
		trial.set_user_attr("fold_losses", fold_losses)
		
		return mean_loss
	
	return objective


def run_search(
	df: pl.DataFrame,
	folds: List[Tuple[List[str], str]],
	task_type: TaskType,
	available_optional: List[str],
	n_trials: int = N_TRIALS,
) -> optuna.Study:
	"""Run Optuna feature selection search."""
	arch_config = load_fixed_architecture(task_type)
	
	print_header(f"FEATURE SELECTION ({task_type})")
	print(f"Core families: {CORE_FAMILIES}")
	print(f"Optional families: {available_optional}")
	print(f"Search space: 2^{len(available_optional)} = {2**len(available_optional)} combinations")
	print(f"Trials: {n_trials}")
	
	study = optuna.create_study(direction="minimize", study_name=f"feature_selection_{task_type}")
	objective = create_objective(df, folds, task_type, available_optional, arch_config)
	study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
	
	return study


def run_exhaustive(
	df: pl.DataFrame,
	folds: List[Tuple[List[str], str]],
	task_type: TaskType,
	available_optional: List[str],
) -> List[Dict[str, Any]]:
	"""Exhaustive search over all feature combinations (for small search spaces)."""
	arch_config = load_fixed_architecture(task_type)
	
	# Generate all 2^n combinations
	all_combos = [[]]
	for r in range(1, len(available_optional) + 1):
		all_combos.extend(combinations(available_optional, r))
	
	print_header(f"EXHAUSTIVE FEATURE SELECTION ({task_type})")
	print(f"Testing {len(all_combos)} combinations...")
	
	results = []
	for i, combo in enumerate(all_combos):
		selected = list(CORE_FAMILIES) + list(combo)
		feature_cols = select_best_window_per_feature(get_features_for_families(df, selected))
		
		print(f"[{i+1}/{len(all_combos)}] {selected} ({len(feature_cols)} features)")
		
		set_seed(42)
		mean_loss, fold_losses = evaluate_feature_set(df, folds, feature_cols, task_type, arch_config)
		
		results.append({
			"families": selected,
			"n_features": len(feature_cols),
			"mean_loss": mean_loss,
			"fold_losses": fold_losses,
		})
		print(f"  -> Loss: {mean_loss:.5f}")
	
	return sorted(results, key=lambda x: x["mean_loss"])


# ============================================================================
# RESULTS
# ============================================================================


def print_results(study: optuna.Study):
	"""Print search results summary."""
	completed = sorted(
		[t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
		key=lambda t: t.value
	)
	
	print_header("RESULTS")
	print("\nTop 10 Feature Combinations:")
	for i, trial in enumerate(completed[:10]):
		families = trial.user_attrs.get("families", [])
		n = trial.user_attrs.get("n_features", 0)
		print(f"{i+1}. Loss: {trial.value:.5f} | {n} features | {families}")
	
	# Feature importance (frequency in top 20%)
	top_n = max(1, len(completed) // 5)
	counts = {}
	for trial in completed[:top_n]:
		for f in trial.user_attrs.get("families", []):
			counts[f] = counts.get(f, 0) + 1
	
	print(f"\nFamily Importance (top {top_n} trials):")
	for family, count in sorted(counts.items(), key=lambda x: -x[1]):
		pct = count / top_n
		print(f"  {family:20} {pct:5.0%} {'█' * int(pct * 20)}")


def save_best(study: optuna.Study, df: pl.DataFrame, task_type: TaskType):
	"""Save best feature configuration."""
	best = study.best_trial
	families = best.user_attrs.get("families", [])
	feature_cols = select_best_window_per_feature(get_features_for_families(df, families))
	
	output = {
		"families": families,
		"n_features": len(feature_cols),
		"mean_loss": best.value,
		"feature_cols": feature_cols,
	}
	
	path = MODELS_DIR / f"best_features_{task_type}.json"
	with open(path, "w") as f:
		json.dump(output, f, indent=2)
	print(f"\nSaved to {path}")


# ============================================================================
# MAIN
# ============================================================================


def run_pipeline(task_type: TaskType = "multiclass"):
	"""Run feature selection pipeline."""
	print_header("FEATURE SELECTION SEARCH")
	print(f"Task: {task_type}")
	print(f"Device: {DEVICE}")
	
	# Load and prepare data
	df = load_frame(Path("data/training/understat_df.parquet"))
	df = TASK_CONFIG[task_type]["add_targets_fn"](df)
	df = filter_min_history(df)
	
	# Drop rows with missing odds
	odds_cols = TASK_CONFIG[task_type]["odds_cols"]
	n_before = len(df)
	df = df.drop_nulls(subset=odds_cols)
	print(f"Dropped {n_before - len(df)} rows with missing odds")
	print(f"Data: {df.shape}")
	
	# Get available families
	available = get_available_families(df)
	available_optional = [f for f in OPTIONAL_FAMILIES if f in available]
	print(f"Available optional: {available_optional}")
	
	# Pre-filter nulls for ALL families (core + optional) to ensure consistent sample size
	all_families = list(set(CORE_FAMILIES + available_optional))
	df = prefilter_nulls_for_all_families(df, all_families)
	print(f"Data after null filtering: {df.shape}")
	
	# Generate CV folds
	folds = generate_rolling_cv_folds(df, N_CV_FOLDS)
	print(f"Folds: {len(folds)}")
	
	# MLflow setup
	mlflow.set_experiment(f"feature_selection_{task_type}")
	
	# Choose search strategy
	search_space = 2 ** len(available_optional)
	
	with mlflow.start_run(run_name=f"feature_search_{task_type}"):
		mlflow.log_params({
			"task": task_type,
			"core_families": CORE_FAMILIES,
			"optional_families": available_optional,
			"search_space": search_space,
		})
		
		if search_space <= 64:
			results = run_exhaustive(df, folds, task_type, available_optional)
			best = results[0]
			
			print_header("RESULTS")
			for i, r in enumerate(results[:10]):
				print(f"{i+1}. Loss: {r['mean_loss']:.5f} | {r['n_features']} features | {r['families']}")
			
			# Save best
			feature_cols = select_best_window_per_feature(get_features_for_families(df, best["families"]))
			path = MODELS_DIR / f"best_features_{task_type}.json"
			with open(path, "w") as f:
				json.dump({**best, "feature_cols": feature_cols}, f, indent=2)
			print(f"\nSaved to {path}")
			
			mlflow.log_metric("best_loss", best["mean_loss"])
		else:
			study = run_search(df, folds, task_type, available_optional, n_trials=N_TRIALS)
			print_results(study)
			save_best(study, df, task_type)
			mlflow.log_metric("best_loss", study.best_value)


if __name__ == "__main__":
	run_pipeline(task_type="multiclass")
