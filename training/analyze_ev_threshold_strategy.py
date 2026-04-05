from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from training.evaluation.ev_threshold import (
	apply_ev_threshold,
	build_group_labels,
	evaluate_selection_bankroll_strategy,
	fit_monotone_ev_threshold,
	selection_bankroll_path,
	selection_to_bet_records,
	subtract_metric_dicts,
	summarize_raw_ev_bins,
)
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.train_main_model import (
	DEFAULT_PARQUET,
	DEVICE,
	build_train_config,
	load_evaluation_config,
	load_training_config,
	prepare_phase_loaders,
	resolve_final_training_epochs,
	set_seed,
	split_selection_folds,
)
from training.training_loop import train_fixed_epochs
from training.train_utils import (
	add_targets_and_implied,
	filter_min_history,
	generate_rolling_cv_folds,
	get_num_leagues,
	load_frame,
	prepare_data,
	resolve_test_season,
	select_feature_columns,
)
from utils.paths import PROJECT_ROOT, TRACKED_ASSETS_DIR
from utils.portfolio import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION, select_best_result_value

DEFAULT_OUTPUT_DIR = TRACKED_ASSETS_DIR / "tmp" / "ev_threshold_strategy_review"
DEFAULT_RAW_EV_BINS = np.array([0.0, 0.01, 0.02, 0.03, 0.05, 0.10, np.inf], dtype=float)
DEFAULT_THRESHOLD_GRID = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.10], dtype=float)
DEFAULT_PLOT_THRESHOLDS = np.array([0.01, 0.03], dtype=float)
GROUPING_MODES = ("day", "split_week")


def _mean_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
	if not rows:
		return {}
	keys = rows[0].keys()
	return {
		key: float(np.mean([float(row[key]) for row in rows]))
		for key in keys
	}


def _parse_kelly_fractions(raw: str) -> list[float]:
	values = []
	for item in str(raw).split(","):
		text = item.strip()
		if not text:
			continue
		value = float(text)
		if value <= 0.0:
			raise ValueError("Kelly fractions must be positive.")
		values.append(value)
	if not values:
		raise ValueError("Provide at least one Kelly fraction.")
	return values


def _parse_threshold_grid(raw: str) -> np.ndarray:
	values = []
	for item in str(raw).split(","):
		text = item.strip()
		if not text:
			continue
		value = float(text)
		if value < 0.0:
			raise ValueError("Thresholds must be non-negative.")
		values.append(value)
	if not values:
		raise ValueError("Provide at least one threshold.")
	values.append(0.0)
	return np.asarray(sorted(set(values)), dtype=float)


def _grouping_label(mode: str) -> str:
	return {
		"day": "daily",
		"split_week": "split-week",
	}[mode]


def _predict_probabilities_from_prepared_data(model, data: dict, device: torch.device) -> np.ndarray:
	"""Run the model on arrays returned by prepare_data/prepare_phase_loaders."""

	model.eval()
	x_tensor = torch.tensor(data["X"], dtype=torch.float32, device=device)
	cat_tensor = torch.tensor(data["cat_features"], dtype=torch.long, device=device)
	implied_tensor = torch.tensor(data["implied"], dtype=torch.float32, device=device)
	raw_margin_tensor = torch.tensor(data["raw_margin"], dtype=torch.float32, device=device)
	with torch.no_grad():
		logits = model(x_tensor, cat_tensor, implied_tensor, raw_margin_tensor)
	return torch.softmax(logits, dim=-1).cpu().numpy()


def _prepare_df(parquet_path: Path):
	df = load_frame(parquet_path)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	return df


def _evaluate_fold_selection(selection: dict[str, np.ndarray], y_true: np.ndarray, groups: np.ndarray) -> dict[str, float]:
	return evaluate_selection_bankroll_strategy(
		selection=selection,
		y_true=y_true,
		groups=groups,
		kelly_fraction=DEFAULT_KELLY_FRACTION,
		initial_bankroll=DEFAULT_BANKROLL,
	)


def _fold_prediction_bundle(
	df,
	feature_cols: list[str],
	training_config: dict,
	train_seasons: list[str],
	val_season: str,
	final_train_epochs: int,
	training_seed: int,
	num_leagues: int,
	fold_index: int,
	grouping_mode: str,
) -> dict:
	set_seed(training_seed, deterministic=True)
	data_train, data_val, train_loader, _ = prepare_phase_loaders(
		df,
		feature_cols,
		training_config["batch_size"],
		train_seasons,
		[val_season],
		training_seed + fold_index,
	)
	fold_config = build_train_config(
		training_config,
		data_train["X"].shape[1],
		epochs=final_train_epochs,
		num_leagues=num_leagues,
	)
	fold_model, _, _ = train_fixed_epochs(fold_config, train_loader, device=DEVICE, verbose=True)
	probs = _predict_probabilities_from_prepared_data(fold_model, data_val, device=DEVICE)
	odds_matrix = np.stack([data_val["odds_home"], data_val["odds_draw"], data_val["odds_away"]], axis=1)
	selection = select_best_result_value(probs, odds_matrix, implied_probs=data_val["implied"])
	group_labels = build_group_labels(data_val["dates"], mode=grouping_mode)
	return {
		"fold_index": fold_index,
		"train_seasons": train_seasons,
		"val_season": val_season,
		"selection": selection,
		"y_true": np.asarray(data_val["y"], dtype=int),
		"groups": group_labels,
		"records": selection_to_bet_records(selection, data_val["y"]),
		"baseline": _evaluate_fold_selection(selection, data_val["y"], group_labels),
	}


def _pool_record_fields(folds: list[dict], exclude_fold_index: int | None = None) -> tuple[np.ndarray, np.ndarray]:
	raw_evs: list[np.ndarray] = []
	realized_rois: list[np.ndarray] = []
	for fold in folds:
		if exclude_fold_index is not None and int(fold["fold_index"]) == int(exclude_fold_index):
			continue
		if fold["records"]["raw_ev"].size == 0:
			continue
		raw_evs.append(np.asarray(fold["records"]["raw_ev"], dtype=float))
		realized_rois.append(np.asarray(fold["records"]["realized_roi"], dtype=float))
	if not raw_evs:
		return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
	return np.concatenate(raw_evs), np.concatenate(realized_rois)


def _serialize_threshold(fitted) -> dict[str, float | int | list[dict[str, float]]]:
	support = np.asarray(fitted.support_raw_ev, dtype=float)
	roi = np.asarray(fitted.support_realized_roi, dtype=float)
	if support.size <= 20:
		curve = [
			{
				"raw_ev": float(raw_ev),
				"fitted_realized_roi": float(realized_roi),
			}
			for raw_ev, realized_roi in zip(support, roi)
		]
	else:
		sample_indices = np.linspace(0, support.size - 1, 20, dtype=int)
		curve = [
			{
				"raw_ev": float(support[index]),
				"fitted_realized_roi": float(roi[index]),
			}
			for index in sample_indices
		]
	return {
		"threshold_raw_ev": None if not np.isfinite(fitted.threshold) else float(fitted.threshold),
		"support_count": int(support.size),
		"curve_preview": curve,
	}


def _selection_summary(selection: dict[str, np.ndarray]) -> dict[str, float | int]:
	positive_mask = np.asarray(selection["positive_mask"], dtype=bool)
	return {
		"positive_count": int(positive_mask.sum()),
		"mean_raw_ev": float(np.asarray(selection["best_ev"], dtype=float)[positive_mask].mean()) if positive_mask.any() else 0.0,
		"mean_full_kelly": float(np.asarray(selection["full_kelly"], dtype=float)[positive_mask].mean()) if positive_mask.any() else 0.0,
	}


def _prepare_test_selection(bundle, df, feature_cols: list[str], test_season: str, grouping_mode: str) -> dict:
	data_test = prepare_data(df, feature_cols, [test_season], scaler=None, fit_scaler=False)
	from training.inference import predict_probabilities  # local import to avoid double-scaling on CV folds

	probs = predict_probabilities(
		model=bundle.model,
		scaler=bundle.scaler,
		X_raw=data_test["X"],
		device=torch.device("cpu"),
		cat_features=data_test["cat_features"],
		implied_probs=data_test["implied"],
		raw_margin=data_test["raw_margin"],
	)
	odds_matrix = np.stack([data_test["odds_home"], data_test["odds_draw"], data_test["odds_away"]], axis=1)
	selection = select_best_result_value(probs, odds_matrix, implied_probs=data_test["implied"])
	group_labels = build_group_labels(data_test["dates"], mode=grouping_mode)
	return {
		"selection": selection,
		"y_true": np.asarray(data_test["y"], dtype=int),
		"groups": group_labels,
		"dates": np.asarray(data_test["dates"]),
		"records": selection_to_bet_records(selection, data_test["y"]),
	}


def _evaluate_fixed_threshold_grid(fold_runs: list[dict], thresholds: np.ndarray) -> list[dict]:
	rows = []
	for threshold in np.asarray(thresholds, dtype=float):
		fold_metrics = []
		fold_counts = []
		for fold in fold_runs:
			thresholded_selection = apply_ev_threshold(fold["selection"], float(threshold))
			fold_metrics.append(_evaluate_fold_selection(thresholded_selection, fold["y_true"], fold["groups"]))
			fold_counts.append(_selection_summary(thresholded_selection)["positive_count"])
		rows.append({
			"threshold": float(threshold),
			"mean_metrics": _mean_metric_dict(fold_metrics),
			"mean_positive_count": float(np.mean(fold_counts)),
		})
	return rows


def _evaluate_fixed_threshold_grid_for_fraction(
	fold_runs: list[dict],
	thresholds: np.ndarray,
	kelly_fraction: float,
	test_run: dict | None = None,
	test_baseline_metrics: dict[str, float] | None = None,
) -> list[dict]:
	rows = []
	for threshold in np.asarray(thresholds, dtype=float):
		fold_metrics = []
		fold_counts = []
		for fold in fold_runs:
			thresholded_selection = apply_ev_threshold(fold["selection"], float(threshold))
			fold_metrics.append(
				evaluate_selection_bankroll_strategy(
					selection=thresholded_selection,
					y_true=fold["y_true"],
					groups=fold["groups"],
					kelly_fraction=kelly_fraction,
					initial_bankroll=DEFAULT_BANKROLL,
				)
			)
			fold_counts.append(_selection_summary(thresholded_selection)["positive_count"])
		rows.append({
			"threshold": float(threshold),
			"mean_metrics": _mean_metric_dict(fold_metrics),
			"mean_positive_count": float(np.mean(fold_counts)),
		})
		if test_run is not None:
			test_selection = apply_ev_threshold(test_run["selection"], float(threshold))
			test_metrics = evaluate_selection_bankroll_strategy(
				selection=test_selection,
				y_true=test_run["y_true"],
				groups=test_run["groups"],
				kelly_fraction=kelly_fraction,
				initial_bankroll=DEFAULT_BANKROLL,
			)
			rows[-1]["held_out_test"] = {
				"metrics": test_metrics,
				"selection": _selection_summary(test_selection),
			}
			if test_baseline_metrics is not None:
				rows[-1]["held_out_test"]["delta_minus_baseline"] = subtract_metric_dicts(
					test_metrics,
					test_baseline_metrics,
				)
	return rows


def _best_grid_row(rows: list[dict], metric_path: tuple[str, ...] = ("mean_metrics", "bankroll_roi")) -> dict:
	def _row_metric(row: dict) -> float:
		value = row
		for key in metric_path:
			value = value[key]
		return float(value)

	return max(
		rows,
		key=lambda row: (
			_row_metric(row),
			-float(row["threshold"]),
		),
	)


def _serialize_wealth_path(path: dict) -> dict[str, float | int | list[str] | list[float]]:
	return {
		"starting_bankroll": float(path["starting_bankroll"]),
		"groups": [str(group) for group in path["groups"]],
		"bankroll_after_group": [float(value) for value in path["bankroll_after_group"]],
		"bankroll_roi": float(path["bankroll_roi"]),
		"bankroll_bet_count": int(path["bankroll_bet_count"]),
		"max_drawdown": float(path["max_drawdown"]),
		"final_bankroll": float(path["final_bankroll"]),
	}


def _plot_tick_labels(groups: list[str], max_ticks: int = 10) -> tuple[list[int], list[str]]:
	if not groups:
		return [0], ["start"]

	indices = np.linspace(0, len(groups) - 1, num=min(max_ticks, len(groups)), dtype=int)
	positions = [0]
	labels = ["start"]
	for index in indices.tolist():
		position = int(index) + 1
		if position == positions[-1]:
			continue
		positions.append(position)
		labels.append(str(groups[index]))
	return positions, labels


def _plot_wealth_paths(curves: list[dict], output_path: Path, title: str) -> str:
	if not curves:
		raise ValueError("Need at least one curve to plot.")

	fig, ax = plt.subplots(figsize=(12, 6))
	for curve in curves:
		path = curve["path"]
		y_values = np.asarray(path["bankroll_after_group"], dtype=float)
		x_values = np.arange(y_values.size + 1)
		wealth = np.concatenate([[float(path["starting_bankroll"])], y_values])
		ax.plot(x_values, wealth, linewidth=2.0, label=str(curve["label"]))

	positions, labels = _plot_tick_labels(curves[0]["path"]["groups"])
	ax.set_xticks(positions, labels, rotation=35, ha="right")
	ax.set_xlabel("Test-set window")
	ax.set_ylabel("Bankroll")
	ax.set_title(title)
	ax.grid(True, alpha=0.25)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=180)
	plt.close(fig)
	return str(output_path.resolve())


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument(
		"--kelly-fractions",
		type=str,
		default=str(DEFAULT_KELLY_FRACTION),
		help="Comma-separated Kelly fractions to evaluate for the fixed-threshold sweep.",
	)
	parser.add_argument(
		"--threshold-grid",
		type=str,
		default=",".join(f"{value:g}" for value in DEFAULT_THRESHOLD_GRID.tolist()),
		help="Comma-separated non-negative raw-EV thresholds for the fixed-threshold sweep.",
	)
	parser.add_argument(
		"--grouping-mode",
		type=str,
		choices=GROUPING_MODES,
		default="day",
		help="How to group fixtures into bankroll decision slates.",
	)
	args = parser.parse_args()
	kelly_fractions = _parse_kelly_fractions(args.kelly_fractions)
	threshold_grid = _parse_threshold_grid(args.threshold_grid)

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
	print(f"Grouping mode: {_grouping_label(args.grouping_mode)}")
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
			f"\n=== EV Threshold Objective Fold {fold_index}/{len(objective_folds)}: "
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
				grouping_mode=args.grouping_mode,
			)
		)

	cv_rows = []
	for fold in fold_runs:
		train_raw_ev, train_realized_roi = _pool_record_fields(fold_runs, exclude_fold_index=fold["fold_index"])
		if train_raw_ev.size == 0:
			raise RuntimeError("No positive-EV CV records available to fit a threshold.")
		fitted = fit_monotone_ev_threshold(train_raw_ev, train_realized_roi)
		thresholded_selection = apply_ev_threshold(fold["selection"], fitted.threshold)
		threshold_metrics = _evaluate_fold_selection(thresholded_selection, fold["y_true"], fold["groups"])
		cv_rows.append({
			"fold_index": int(fold["fold_index"]),
			"val_season": str(fold["val_season"]),
			"threshold_fit_record_count": int(train_raw_ev.size),
			"threshold": None if not np.isfinite(fitted.threshold) else float(fitted.threshold),
			"baseline": fold["baseline"],
			"thresholded": threshold_metrics,
			"delta_thresholded_minus_baseline": subtract_metric_dicts(threshold_metrics, fold["baseline"]),
			"baseline_selection": _selection_summary(fold["selection"]),
			"thresholded_selection": _selection_summary(thresholded_selection),
			"threshold_fit": _serialize_threshold(fitted),
		})

	all_cv_raw_ev, all_cv_realized_roi = _pool_record_fields(fold_runs, exclude_fold_index=None)
	final_threshold_fit = fit_monotone_ev_threshold(all_cv_raw_ev, all_cv_realized_roi)

	test_run = _prepare_test_selection(bundle, df, feature_cols, test_season=test_season, grouping_mode=args.grouping_mode)
	test_baseline = _evaluate_fold_selection(test_run["selection"], test_run["y_true"], test_run["groups"])
	test_thresholded_selection = apply_ev_threshold(test_run["selection"], final_threshold_fit.threshold)
	test_thresholded = _evaluate_fold_selection(test_thresholded_selection, test_run["y_true"], test_run["groups"])
	grid_rows = _evaluate_fixed_threshold_grid_for_fraction(
		fold_runs=fold_runs,
		thresholds=threshold_grid,
		kelly_fraction=DEFAULT_KELLY_FRACTION,
		test_run=test_run,
		test_baseline_metrics=test_baseline,
	)
	best_grid_row = _best_grid_row(grid_rows)
	best_grid_test_row = _best_grid_row(grid_rows, metric_path=("held_out_test", "metrics", "bankroll_roi"))
	test_best_grid_selection = apply_ev_threshold(test_run["selection"], best_grid_row["threshold"])
	test_best_grid = _evaluate_fold_selection(test_best_grid_selection, test_run["y_true"], test_run["groups"])
	kelly_sweep = []
	baseline_wealth_curves = []
	fixed_threshold_wealth_curves = {
		float(threshold): []
		for threshold in DEFAULT_PLOT_THRESHOLDS
	}
	for kelly_fraction in kelly_fractions:
		baseline_metrics = evaluate_selection_bankroll_strategy(
			selection=test_run["selection"],
			y_true=test_run["y_true"],
			groups=test_run["groups"],
			kelly_fraction=kelly_fraction,
			initial_bankroll=DEFAULT_BANKROLL,
		)
		sweep_grid_rows = _evaluate_fixed_threshold_grid_for_fraction(
			fold_runs=fold_runs,
			thresholds=threshold_grid,
			kelly_fraction=kelly_fraction,
			test_run=test_run,
			test_baseline_metrics=baseline_metrics,
		)
		best_sweep_row = _best_grid_row(sweep_grid_rows)
		best_test_row = _best_grid_row(sweep_grid_rows, metric_path=("held_out_test", "metrics", "bankroll_roi"))
		test_selection = apply_ev_threshold(test_run["selection"], best_sweep_row["threshold"])
		test_metrics = best_sweep_row["held_out_test"]["metrics"]
		test_path = selection_bankroll_path(
			selection=test_selection,
			y_true=test_run["y_true"],
			groups=test_run["groups"],
			kelly_fraction=kelly_fraction,
			initial_bankroll=DEFAULT_BANKROLL,
		)
		baseline_path = selection_bankroll_path(
			selection=test_run["selection"],
			y_true=test_run["y_true"],
			groups=test_run["groups"],
			kelly_fraction=kelly_fraction,
			initial_bankroll=DEFAULT_BANKROLL,
		)
		baseline_wealth_curves.append({
			"kelly_fraction": float(kelly_fraction),
			"label": f"k={kelly_fraction:.2f}",
			"path": baseline_path,
		})
		row_by_threshold = {
			float(row["threshold"]): row
			for row in sweep_grid_rows
		}
		for threshold in DEFAULT_PLOT_THRESHOLDS:
			fixed_threshold = float(threshold)
			if fixed_threshold not in row_by_threshold:
				continue
			fixed_selection = apply_ev_threshold(test_run["selection"], fixed_threshold)
			fixed_path = selection_bankroll_path(
				selection=fixed_selection,
				y_true=test_run["y_true"],
				groups=test_run["groups"],
				kelly_fraction=kelly_fraction,
				initial_bankroll=DEFAULT_BANKROLL,
			)
			fixed_threshold_wealth_curves[fixed_threshold].append({
				"kelly_fraction": float(kelly_fraction),
				"threshold": fixed_threshold,
				"label": f"k={kelly_fraction:.2f}, EV >= {fixed_threshold:.1%}",
				"path": fixed_path,
			})
		kelly_sweep.append({
			"kelly_fraction": float(kelly_fraction),
			"grid_rows": sweep_grid_rows,
			"best_by_mean_roi": best_sweep_row,
			"best_by_test_roi": best_test_row,
			"held_out_test": {
				"baseline": baseline_metrics,
				"baseline_wealth_path": _serialize_wealth_path(baseline_path),
				"best_threshold": float(best_sweep_row["threshold"]),
				"thresholded": test_metrics,
				"thresholded_wealth_path": _serialize_wealth_path(test_path),
				"delta_thresholded_minus_baseline": subtract_metric_dicts(test_metrics, baseline_metrics),
				"thresholded_selection": _selection_summary(test_selection),
			},
		})

	args.output_dir.mkdir(parents=True, exist_ok=True)
	baseline_plot_path = args.output_dir / "testset_wealth_by_kelly_fraction.png"
	fixed_threshold_plot_paths = {
		0.01: args.output_dir / "testset_wealth_by_kelly_fraction_threshold_1pct.png",
		0.03: args.output_dir / "testset_wealth_by_kelly_fraction_threshold_3pct.png",
	}
	wealth_paths_path = args.output_dir / "testset_wealth_paths_by_kelly_fraction.json"
	plot_paths = {
		"baseline_by_kelly_fraction": _plot_wealth_paths(
			curves=baseline_wealth_curves,
			output_path=baseline_plot_path,
			title="Held-out test wealth progression by Kelly fraction",
		),
	}
	for threshold, curves in fixed_threshold_wealth_curves.items():
		plot_paths[f"fixed_threshold_{int(round(threshold * 100))}pct_by_kelly_fraction"] = _plot_wealth_paths(
			curves=curves,
			output_path=fixed_threshold_plot_paths[threshold],
			title=f"Held-out test wealth progression by Kelly fraction with fixed EV >= {threshold:.1%}",
		)
	with open(wealth_paths_path, "w", encoding="utf-8") as file:
		json.dump(
			{
				"baseline_by_kelly_fraction": [
					{
						"kelly_fraction": float(curve["kelly_fraction"]),
						"path": _serialize_wealth_path(curve["path"]),
					}
					for curve in baseline_wealth_curves
				],
				"fixed_threshold_by_kelly_fraction": {
					f"{int(round(threshold * 100))}pct": [
						{
							"kelly_fraction": float(curve["kelly_fraction"]),
							"threshold": float(curve["threshold"]),
							"path": _serialize_wealth_path(curve["path"]),
						}
						for curve in curves
					]
					for threshold, curves in fixed_threshold_wealth_curves.items()
				},
			},
			file,
			indent=2,
		)

	report = {
		"config": {
			"parquet_path": str(args.parquet_path.resolve()),
			"output_dir": str(args.output_dir.resolve()),
			"training_config": "training/configs/main_models/result.json",
			"evaluation_config": "training/configs/main_models/evaluation.json",
			"bundle_config": str(RESULT_MODEL_BUNDLE_PATHS.config_path.resolve()),
			"test_season": test_season,
			"objective_fold_count": len(objective_folds),
			"final_train_epochs": final_train_epochs,
			"kelly_fraction": DEFAULT_KELLY_FRACTION,
			"initial_bankroll": DEFAULT_BANKROLL,
			"grouping_mode": args.grouping_mode,
			"threshold_grid": [float(value) for value in threshold_grid],
			"kelly_fractions_sweep": kelly_fractions,
		},
		"cv_learning_pool": {
			"record_count": int(all_cv_raw_ev.size),
			"raw_ev_bin_summary": summarize_raw_ev_bins(all_cv_raw_ev, all_cv_realized_roi, DEFAULT_RAW_EV_BINS),
			"final_threshold_fit": _serialize_threshold(final_threshold_fit),
		},
		"cv_nested_evaluation": {
			"baseline_mean": _mean_metric_dict([row["baseline"] for row in cv_rows]),
			"thresholded_mean": _mean_metric_dict([row["thresholded"] for row in cv_rows]),
			"delta_mean_thresholded_minus_baseline": _mean_metric_dict(
				[row["delta_thresholded_minus_baseline"] for row in cv_rows]
			),
			"folds": cv_rows,
		},
		"cv_fixed_threshold_grid": {
			"rows": grid_rows,
			"best_by_mean_roi": best_grid_row,
			"best_by_test_roi": best_grid_test_row,
		},
		"kelly_fraction_sweep": kelly_sweep,
		"held_out_test": {
			"baseline": test_baseline,
			"isotonic_thresholded": test_thresholded,
			"delta_isotonic_thresholded_minus_baseline": subtract_metric_dicts(test_thresholded, test_baseline),
			"baseline_selection": _selection_summary(test_run["selection"]),
			"isotonic_thresholded_selection": _selection_summary(test_thresholded_selection),
			"best_fixed_threshold": {
				"threshold": float(best_grid_row["threshold"]),
				"metrics": test_best_grid,
				"delta_minus_baseline": subtract_metric_dicts(test_best_grid, test_baseline),
				"selection": _selection_summary(test_best_grid_selection),
			},
			"best_fixed_threshold_by_test_roi": {
				"threshold": float(best_grid_test_row["threshold"]),
				"metrics": best_grid_test_row["held_out_test"]["metrics"],
				"delta_minus_baseline": best_grid_test_row["held_out_test"]["delta_minus_baseline"],
				"selection": best_grid_test_row["held_out_test"]["selection"],
			},
			"fixed_threshold_grid": {
				"rows": grid_rows,
				"best_by_mean_roi": best_grid_row,
				"best_by_test_roi": best_grid_test_row,
			},
			"test_raw_ev_bin_summary": summarize_raw_ev_bins(
				test_run["records"]["raw_ev"],
				test_run["records"]["realized_roi"],
				DEFAULT_RAW_EV_BINS,
			),
			"threshold_fit": _serialize_threshold(final_threshold_fit),
		},
		"artifacts": {
			"testset_wealth_plots": plot_paths,
			"testset_wealth_paths_json": str(wealth_paths_path.resolve()),
		},
	}

	report_path = args.output_dir / "ev_threshold_strategy_report.json"
	with open(report_path, "w", encoding="utf-8") as file:
		json.dump(report, file, indent=2)

	print("\n=== EV Threshold Strategy Summary ===")
	threshold_value = report["held_out_test"]["threshold_fit"]["threshold_raw_ev"]
	if threshold_value is None:
		print("Learned raw-EV threshold: no positive fitted ROI region found")
	else:
		print(f"Learned raw-EV threshold: {threshold_value:.4%}")
	print(f"CV mean baseline ROI: {report['cv_nested_evaluation']['baseline_mean']['bankroll_roi']:.4f}")
	print(f"CV mean thresholded ROI: {report['cv_nested_evaluation']['thresholded_mean']['bankroll_roi']:.4f}")
	print(
		"CV delta ROI (thresholded - baseline): "
		f"{report['cv_nested_evaluation']['delta_mean_thresholded_minus_baseline']['bankroll_roi']:+.4f}"
	)
	print(
		"Best fixed raw-EV threshold on CV grid: "
		f"{report['cv_fixed_threshold_grid']['best_by_mean_roi']['threshold']:.4%} "
		f"(mean ROI={report['cv_fixed_threshold_grid']['best_by_mean_roi']['mean_metrics']['bankroll_roi']:.4f})"
	)
	print(
		"Best fixed raw-EV threshold on held-out test: "
		f"{report['held_out_test']['fixed_threshold_grid']['best_by_test_roi']['threshold']:.4%} "
		f"(test ROI={report['held_out_test']['fixed_threshold_grid']['best_by_test_roi']['held_out_test']['metrics']['bankroll_roi']:.4f})"
	)
	print(f"Test baseline ROI: {report['held_out_test']['baseline']['bankroll_roi']:.4f}")
	print(f"Test isotonic-thresholded ROI: {report['held_out_test']['isotonic_thresholded']['bankroll_roi']:.4f}")
	print(
		"Test delta ROI (isotonic thresholded - baseline): "
		f"{report['held_out_test']['delta_isotonic_thresholded_minus_baseline']['bankroll_roi']:+.4f}"
	)
	print(
		"Test best fixed-threshold ROI: "
		f"{report['held_out_test']['best_fixed_threshold']['metrics']['bankroll_roi']:.4f} "
		f"(delta={report['held_out_test']['best_fixed_threshold']['delta_minus_baseline']['bankroll_roi']:+.4f})"
	)
	if len(kelly_sweep) > 1:
		print("\nKelly fraction sweep")
		for row in kelly_sweep:
			print(
				f"  k={row['kelly_fraction']:.2f} "
				f"baseline_test_roi={row['held_out_test']['baseline']['bankroll_roi']:.4f} "
				f"best_thr={row['best_by_mean_roi']['threshold']:.4%} "
				f"cv_roi={row['best_by_mean_roi']['mean_metrics']['bankroll_roi']:.4f} "
				f"test_roi={row['held_out_test']['thresholded']['bankroll_roi']:.4f} "
				f"test_delta={row['held_out_test']['delta_thresholded_minus_baseline']['bankroll_roi']:+.4f} "
				f"best_test_thr={row['best_by_test_roi']['threshold']:.4%} "
				f"best_test_roi={row['best_by_test_roi']['held_out_test']['metrics']['bankroll_roi']:.4f}"
			)
	print(f"Saved baseline wealth plot to {plot_paths['baseline_by_kelly_fraction']}")
	for threshold in DEFAULT_PLOT_THRESHOLDS:
		key = f"fixed_threshold_{int(round(float(threshold) * 100))}pct_by_kelly_fraction"
		print(f"Saved fixed-threshold wealth plot ({float(threshold):.1%}) to {plot_paths[key]}")
	print(f"Saved wealth path data to {wealth_paths_path}")
	print(f"Wrote report to {report_path}")


if __name__ == "__main__":
	main()
