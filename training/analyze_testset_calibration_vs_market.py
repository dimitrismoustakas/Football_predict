from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.inference import predict_probabilities
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.train_utils import add_targets_and_implied, filter_min_history, load_frame, prepare_data, resolve_test_season
from utils.paths import ARTIFACTS_DIR, DATA_DIR, PROJECT_ROOT

DEFAULT_PARQUET = DATA_DIR / "training" / "understat_df.parquet"
DEFAULT_OUTPUT_DIR = ARTIFACTS_DIR / "tmp" / "testset_calibration_review"
OUTCOME_LABELS = ["home", "draw", "away"]


def load_json(path: Path) -> dict:
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def clip_probs(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	clipped = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
	return clipped / clipped.sum(axis=1, keepdims=True)


def multiclass_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
	probs = clip_probs(probs)
	preds = np.argmax(probs, axis=1)
	one_hot = np.eye(probs.shape[1], dtype=float)[y_true]
	accuracy = float(np.mean(preds == y_true))
	brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
	log_loss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true])))
	return {
		"accuracy": accuracy,
		"brier": brier,
		"log_loss": log_loss,
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
		"class_index": class_index,
		"class_label": OUTCOME_LABELS[class_index],
		"sample_count": int(len(confidences)),
		"bin_count": int(len(bins)),
		"adaptive_ece": float(weighted_gap),
		"max_gap": float(max((row["gap"] for row in bins), default=0.0)),
		"bins": bins,
	}


def summarize_calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> dict:
	class_rows = [classwise_adaptive_ece(y_true, probs, class_index=i, n_bins=n_bins) for i in range(probs.shape[1])]
	return {
		"n_bins_requested": int(n_bins),
		"macro_adaptive_ece": float(np.mean([row["adaptive_ece"] for row in class_rows])),
		"weighted_adaptive_ece": float(np.average([row["adaptive_ece"] for row in class_rows])),
		"classes": class_rows,
	}


def plot_reliability_comparison(model_summary: dict, market_summary: dict, output_path: Path):
	fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
	for class_index, axis in enumerate(axes):
		model_bins = model_summary["classes"][class_index]["bins"]
		market_bins = market_summary["classes"][class_index]["bins"]
		model_x = [row["mean_confidence"] for row in model_bins]
		model_y = [row["empirical_frequency"] for row in model_bins]
		market_x = [row["mean_confidence"] for row in market_bins]
		market_y = [row["empirical_frequency"] for row in market_bins]
		axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
		axis.plot(model_x, model_y, marker="o", linewidth=2, color="#0B6E4F", label="Model")
		axis.plot(market_x, market_y, marker="s", linewidth=2, color="#C84C09", label="Market")
		axis.set_title(f"{OUTCOME_LABELS[class_index].title()}\nModel aECE={model_summary['classes'][class_index]['adaptive_ece']:.4f} | Market aECE={market_summary['classes'][class_index]['adaptive_ece']:.4f}")
		axis.set_xlabel("Mean predicted probability")
		if class_index == 0:
			axis.set_ylabel("Empirical frequency")
		axis.set_xlim(0, 1)
		axis.set_ylim(0, 1)
		axis.grid(alpha=0.25)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
	fig.suptitle("Test-set classwise reliability: model vs market", fontsize=14)
	fig.tight_layout(rect=[0, 0.06, 1, 0.95])
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def plot_gap_comparison(model_summary: dict, market_summary: dict, output_path: Path):
	fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
	for class_index, axis in enumerate(axes):
		model_bins = model_summary["classes"][class_index]["bins"]
		market_bins = market_summary["classes"][class_index]["bins"]
		model_x = np.arange(len(model_bins))
		market_x = np.arange(len(market_bins))
		axis.bar(model_x - 0.18, [row["gap"] for row in model_bins], width=0.36, color="#0B6E4F", label="Model")
		axis.bar(market_x + 0.18, [row["gap"] for row in market_bins], width=0.36, color="#C84C09", label="Market")
		axis.set_title(OUTCOME_LABELS[class_index].title())
		axis.set_xlabel("Adaptive bin")
		if class_index == 0:
			axis.set_ylabel("Absolute calibration gap")
		axis.grid(axis="y", alpha=0.25)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
	fig.suptitle("Test-set absolute calibration gaps by adaptive bin", fontsize=14)
	fig.tight_layout(rect=[0, 0.06, 1, 0.95])
	fig.savefig(output_path, dpi=160)
	plt.close(fig)


def build_summary(test_season: str, sample_count: int, model_probs: np.ndarray, market_probs: np.ndarray, y_true: np.ndarray, n_bins: int) -> dict:
	model_metrics = multiclass_metrics(y_true, model_probs)
	market_metrics = multiclass_metrics(y_true, market_probs)
	model_calibration = summarize_calibration(y_true, model_probs, n_bins=n_bins)
	market_calibration = summarize_calibration(y_true, market_probs, n_bins=n_bins)
	per_class = []
	for class_index, class_label in enumerate(OUTCOME_LABELS):
		model_class = model_calibration["classes"][class_index]
		market_class = market_calibration["classes"][class_index]
		per_class.append({
			"class_index": class_index,
			"class_label": class_label,
			"model_adaptive_ece": model_class["adaptive_ece"],
			"market_adaptive_ece": market_class["adaptive_ece"],
			"ece_delta_model_minus_market": float(model_class["adaptive_ece"] - market_class["adaptive_ece"]),
			"model_max_gap": model_class["max_gap"],
			"market_max_gap": market_class["max_gap"],
		})
	return {
		"test_season": test_season,
		"sample_count": int(sample_count),
		"comparison": {
			"model_metrics": model_metrics,
			"market_metrics": market_metrics,
			"log_loss_delta_model_minus_market": float(model_metrics["log_loss"] - market_metrics["log_loss"]),
			"brier_delta_model_minus_market": float(model_metrics["brier"] - market_metrics["brier"]),
			"accuracy_delta_model_minus_market": float(model_metrics["accuracy"] - market_metrics["accuracy"]),
			"macro_adaptive_ece_delta_model_minus_market": float(model_calibration["macro_adaptive_ece"] - market_calibration["macro_adaptive_ece"]),
		},
		"model": model_calibration,
		"market": market_calibration,
		"per_class_comparison": per_class,
	}


def print_report(summary: dict, output_dir: Path):
	comparison = summary["comparison"]
	print(f"Test season: {summary['test_season']}")
	print(f"Samples: {summary['sample_count']}")
	print("")
	print("Overall metrics")
	print(
		"  Model  "
		f"log_loss={comparison['model_metrics']['log_loss']:.6f} "
		f"brier={comparison['model_metrics']['brier']:.6f} "
		f"acc={comparison['model_metrics']['accuracy']:.4f}"
	)
	print(
		"  Market "
		f"log_loss={comparison['market_metrics']['log_loss']:.6f} "
		f"brier={comparison['market_metrics']['brier']:.6f} "
		f"acc={comparison['market_metrics']['accuracy']:.4f}"
	)
	print(
		"  Delta  "
		f"log_loss={comparison['log_loss_delta_model_minus_market']:+.6f} "
		f"brier={comparison['brier_delta_model_minus_market']:+.6f} "
		f"acc={comparison['accuracy_delta_model_minus_market']:+.4f}"
	)
	print("")
	print("Adaptive ECE")
	print(
		"  Model  "
		f"macro_aECE={summary['model']['macro_adaptive_ece']:.6f}"
	)
	print(
		"  Market "
		f"macro_aECE={summary['market']['macro_adaptive_ece']:.6f}"
	)
	print(
		"  Delta  "
		f"macro_aECE={comparison['macro_adaptive_ece_delta_model_minus_market']:+.6f}"
	)
	print("")
	print("Classwise adaptive ECE")
	for row in summary["per_class_comparison"]:
		print(
			f"  {row['class_label']:>4s} "
			f"model={row['model_adaptive_ece']:.6f} "
			f"market={row['market_adaptive_ece']:.6f} "
			f"delta={row['ece_delta_model_minus_market']:+.6f}"
		)
	print("")
	print(f"Wrote outputs to {output_dir}")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--n-bins", type=int, default=15)
	args = parser.parse_args()

	device = torch.device("cpu")
	bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=device)
	evaluation_path = PROJECT_ROOT / "training" / "configs" / "main_models" / "evaluation.json"
	evaluation_config = load_json(evaluation_path)
	configured_test_season = bundle.metadata.get("evaluation_protocol", {}).get("test_season") or evaluation_config.get("test_season")

	df = load_frame(args.parquet_path)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	test_season = resolve_test_season(df, str(configured_test_season) if configured_test_season else None)
	data = prepare_data(df, bundle.feature_cols, [test_season], scaler=None, fit_scaler=False)

	model_probs = predict_probabilities(
		model=bundle.model,
		scaler=bundle.scaler,
		X_raw=data["X"],
		device=device,
		cat_features=data["cat_features"],
		implied_probs=data["implied"],
		raw_margin=data["raw_margin"],
	)
	model_probs = clip_probs(model_probs)
	market_probs = clip_probs(data["implied"])
	summary = build_summary(
		test_season=test_season,
		sample_count=len(data["y"]),
		model_probs=model_probs,
		market_probs=market_probs,
		y_true=data["y"],
		n_bins=args.n_bins,
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = args.output_dir / "testset_calibration_summary.json"
	reliability_path = args.output_dir / "testset_reliability_model_vs_market.png"
	gap_path = args.output_dir / "testset_reliability_gap_model_vs_market.png"

	with open(summary_path, "w", encoding="utf-8") as file:
		json.dump(summary, file, indent=2)

	plot_reliability_comparison(summary["model"], summary["market"], reliability_path)
	plot_gap_comparison(summary["model"], summary["market"], gap_path)
	print_report(summary, args.output_dir)


if __name__ == "__main__":
	main()
