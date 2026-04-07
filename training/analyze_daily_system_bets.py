from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.analyze_ev_threshold_strategy import (
	_fold_prediction_bundle,
	_prepare_df,
	_prepare_test_selection,
)
from training.evaluation.ev_threshold import (
	evaluate_selection_bankroll_strategy,
	selection_bankroll_path,
	subtract_metric_dicts,
)
from training.evaluation.system_bets import (
	evaluate_system_bankroll_strategy,
	system_bankroll_path,
)
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.train_main_model import (
	DEFAULT_PARQUET,
	load_evaluation_config,
	load_training_config,
	resolve_final_training_epochs,
	split_selection_folds,
)
from training.train_utils import (
	generate_rolling_cv_folds,
	get_num_leagues,
	resolve_test_season,
	select_feature_columns,
)
from utils.paths import PROJECT_ROOT, TRACKED_ASSETS_DIR
from utils.portfolio import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION

DEFAULT_OUTPUT_DIR = TRACKED_ASSETS_DIR / "tmp" / "daily_system_bets_review"
SYSTEM_NAMES = ("2/3", "2/3/4")


def _mean_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
	if not rows:
		return {}
	keys = rows[0].keys()
	return {
		key: float(np.mean([float(row[key]) for row in rows]))
		for key in keys
	}


def _serialize_wealth_path(path: dict) -> dict:
	return {
		"starting_bankroll": float(path["starting_bankroll"]),
		"final_bankroll": float(path["final_bankroll"]),
		"bankroll_roi": float(path["bankroll_roi"]),
		"bankroll_bet_count": int(path["bankroll_bet_count"]),
		"max_drawdown": float(path["max_drawdown"]),
		"groups": [str(group) for group in path["groups"]],
		"bankroll_after_group": [float(value) for value in path["bankroll_after_group"]],
		"group_rows": path["group_rows"],
	}


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--kelly-fraction", type=float, default=DEFAULT_KELLY_FRACTION)
	args = parser.parse_args()

	training_config = load_training_config()
	evaluation_config = load_evaluation_config()
	bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=torch.device("cpu"))
	final_train_epochs = int(
		bundle.metadata.get("selection_summary", {}).get("final_train_epochs")
		or bundle.metadata.get("final_epochs")
		or resolve_final_training_epochs(
			training_config["max_epochs"],
			int(bundle.metadata.get("selection_summary", {}).get("best_epoch", 1)),
		)
	)

	print(f"Using final_train_epochs={final_train_epochs}")
	print(f"Loading data from {args.parquet_path}")
	df = _prepare_df(args.parquet_path)
	test_season = resolve_test_season(df, evaluation_config["test_season"])
	feature_cols = select_feature_columns(df, PROJECT_ROOT / "training" / "configs" / "main_models" / "result_features.json")
	num_leagues = get_num_leagues(df)
	folds = generate_rolling_cv_folds(df, n_folds=evaluation_config["rolling_cv_n_folds"], test_season=test_season)
	objective_folds, _ = split_selection_folds(folds)

	print(f"Objective folds: {len(objective_folds)} | test season: {test_season}")
	fold_runs = []
	for fold_index, (train_seasons, val_season) in enumerate(objective_folds, start=1):
		print(
			f"\n=== Daily System Fold {fold_index}/{len(objective_folds)}: "
			f"{train_seasons[0]}..{train_seasons[-1]} -> {val_season} ==="
		)
		fold_runs.append(
			_fold_prediction_bundle(
				df=df,
				feature_cols=feature_cols,
				training_config=training_config,
				train_seasons=train_seasons,
				val_season=val_season,
				final_train_epochs=final_train_epochs,
				training_seed=evaluation_config["training_seed"],
				num_leagues=num_leagues,
				fold_index=fold_index,
				grouping_mode="day",
			)
		)

	test_run = _prepare_test_selection(bundle, df, feature_cols, test_season=test_season, grouping_mode="day")
	baseline_cv_metrics = _mean_metric_dict([fold["baseline"] for fold in fold_runs])
	baseline_test_metrics = evaluate_selection_bankroll_strategy(
		selection=test_run["selection"],
		y_true=test_run["y_true"],
		groups=test_run["groups"],
		kelly_fraction=args.kelly_fraction,
		initial_bankroll=DEFAULT_BANKROLL,
	)
	baseline_test_path = selection_bankroll_path(
		selection=test_run["selection"],
		y_true=test_run["y_true"],
		groups=test_run["groups"],
		kelly_fraction=args.kelly_fraction,
		initial_bankroll=DEFAULT_BANKROLL,
	)

	results = {
		"grouping_mode": "day",
		"kelly_fraction": float(args.kelly_fraction),
		"test_season": str(test_season),
		"baseline": {
			"cv_mean": baseline_cv_metrics,
			"held_out_test": {
				"metrics": baseline_test_metrics,
				"path": _serialize_wealth_path(baseline_test_path),
			},
		},
		"systems": {},
	}

	for system_name in SYSTEM_NAMES:
		fold_metrics = []
		for fold in fold_runs:
			fold_metrics.append(
				evaluate_system_bankroll_strategy(
					selection=fold["selection"],
					y_true=fold["y_true"],
					system_name=system_name,
					groups=fold["groups"],
					kelly_fraction=args.kelly_fraction,
					initial_bankroll=DEFAULT_BANKROLL,
				)
			)
		test_metrics = evaluate_system_bankroll_strategy(
			selection=test_run["selection"],
			y_true=test_run["y_true"],
			system_name=system_name,
			groups=test_run["groups"],
			kelly_fraction=args.kelly_fraction,
			initial_bankroll=DEFAULT_BANKROLL,
		)
		test_path = system_bankroll_path(
			selection=test_run["selection"],
			y_true=test_run["y_true"],
			system_name=system_name,
			groups=test_run["groups"],
			kelly_fraction=args.kelly_fraction,
			initial_bankroll=DEFAULT_BANKROLL,
		)
		results["systems"][system_name] = {
			"cv_mean": _mean_metric_dict(fold_metrics),
			"cv_delta_minus_baseline": subtract_metric_dicts(_mean_metric_dict(fold_metrics), baseline_cv_metrics),
			"held_out_test": {
				"metrics": test_metrics,
				"delta_minus_baseline": subtract_metric_dicts(test_metrics, baseline_test_metrics),
				"path": _serialize_wealth_path(test_path),
			},
		}

	args.output_dir.mkdir(parents=True, exist_ok=True)
	output_path = args.output_dir / "summary.json"
	output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

	print("\n=== Daily Strategy Comparison ===")
	print(f"Baseline held-out ROI: {baseline_test_metrics['bankroll_roi']:.6f}")
	for system_name in SYSTEM_NAMES:
		metrics = results["systems"][system_name]["held_out_test"]["metrics"]
		delta = results["systems"][system_name]["held_out_test"]["delta_minus_baseline"]
		print(
			f"{system_name} held-out ROI: {metrics['bankroll_roi']:.6f} "
			f"(delta {delta['bankroll_roi']:+.6f}) | drawdown {metrics['max_drawdown']:.6f}"
		)
	print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
	main()