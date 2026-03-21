"""
Analyze a trained gated-residual match-result model on a held-out season.

This script focuses on the parts that are structurally interpretable:
- market calibration parameters
- calibrated anchor probabilities vs raw implied probabilities
- residual logits before the gate
- applied residual logits after the gate
- per-league and overall held-out metrics

It assumes the production bundle in artifacts/models/ was produced by the
current gated residual trainer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.feature_engineering import LEAGUE_IDS
from training.evaluation.metrics import ranked_probability_score
from training.inference import model_requires_cat_features
from training.model_bundle import RESULT_MODEL_BUNDLE_PATHS, load_model_bundle
from training.train_utils import add_targets_and_implied, filter_min_history, load_frame, prepare_data, resolve_test_season
from utils.paths import MODELS_DIR, PROJECT_ROOT

DEFAULT_PARQUET = PROJECT_ROOT / "data" / "training" / "understat_df.parquet"
DEFAULT_OUTPUT = MODELS_DIR / "result_model_analysis.json"

OUTCOME_LABELS = ["home", "draw", "away"]
LEAGUE_NAME_BY_IDX = {idx: name for name, idx in LEAGUE_IDS.items()}


def load_json(path: Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
	logits = logits - logits.max(axis=1, keepdims=True)
	exp = np.exp(logits)
	return exp / exp.sum(axis=1, keepdims=True)


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
	preds = np.argmax(probs, axis=1)
	y_onehot = np.eye(3)[y_true]
	return {
		"accuracy": float(accuracy_score(y_true, preds)),
		"log_loss": float(log_loss(y_true, probs, labels=[0, 1, 2])),
		"brier": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
		"rps": float(ranked_probability_score(y_true, probs)),
	}


def summarize_vector(values: np.ndarray) -> dict[str, float]:
	array = np.asarray(values, dtype=float).reshape(-1)
	return {
		"mean": float(array.mean()),
		"std": float(array.std()),
		"min": float(array.min()),
		"p10": float(np.percentile(array, 10)),
		"p50": float(np.percentile(array, 50)),
		"p90": float(np.percentile(array, 90)),
		"max": float(array.max()),
	}


def summarize_probability_shift(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
	diff = after - before
	tv = 0.5 * np.abs(diff).sum(axis=1)
	return {
		"mean_abs_delta_per_class": float(np.abs(diff).mean()),
		"mean_total_variation": float(tv.mean()),
		"p90_total_variation": float(np.percentile(tv, 90)),
		"max_total_variation": float(tv.max()),
	}


def centered_bias_row(row: np.ndarray) -> list[float]:
	centered = row - row.mean()
	return [float(value) for value in centered]


def reshape_mixer(row: np.ndarray) -> list[list[float]]:
	matrix = row.reshape(3, 3).copy()
	for idx in range(3):
		matrix[idx, idx] = 0.0
	return [[float(value) for value in matrix_row] for matrix_row in matrix]


def league_row_is_enabled(mask: torch.Tensor | None, league_idx: int) -> bool:
	if mask is None:
		return True
	return bool(mask[league_idx, 0].item() > 0.5)


def extract_calibration_parameters(model) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"market_logit_scale": float(model.market_logit_scale),
		"gate_bias": float(model.gate_bias.detach().cpu().view(-1)[0].item()),
		"shared_gate": bool(model.shared_gate),
		"linear_gate": bool(model.linear_gate),
		"gate_market_feature_order": [
			"implied_home",
			"implied_draw",
			"implied_away",
			"market_entropy",
			"market_max_prob",
			"raw_margin",
			"market_min_prob",
		],
	}
	if getattr(model, "market_class_scale", None) is not None:
		payload["global_market_class_scale"] = [
			float(value) for value in torch.exp(model.market_class_scale.detach().cpu()).tolist()
		]
	if getattr(model, "learn_market_bias", False):
		payload["global_market_bias_centered"] = centered_bias_row(model.market_bias.detach().cpu().numpy())

	league_rows = []
	for league_idx in range(int(getattr(model, "num_leagues", 0))):
		row: dict[str, Any] = {
			"league_idx": league_idx,
			"league_name": LEAGUE_NAME_BY_IDX.get(league_idx, f"league_{league_idx}"),
		}
		if getattr(model, "league_market_scale", None) is not None:
			enabled = league_row_is_enabled(getattr(model, "league_market_scale_enabled_mask", None), league_idx)
			raw_value = model.league_market_scale.weight[league_idx, 0].detach().cpu().item()
			row["league_market_scale_enabled"] = enabled
			row["league_market_scale_raw"] = float(raw_value)
			row["league_market_scale_multiplier"] = float(np.exp(raw_value) if enabled else 1.0)
		if getattr(model, "league_market_class_scale", None) is not None:
			enabled = league_row_is_enabled(getattr(model, "league_market_class_scale_enabled_mask", None), league_idx)
			raw_vector = model.league_market_class_scale.weight[league_idx].detach().cpu().numpy()
			row["league_market_class_scale_enabled"] = enabled
			row["league_market_class_scale_raw"] = [float(value) for value in raw_vector.tolist()]
			row["league_market_class_scale_multiplier"] = [
				float(np.exp(value) if enabled else 1.0)
				for value in raw_vector.tolist()
			]
		if getattr(model, "league_market_bias", None) is not None:
			row["league_market_bias_centered"] = centered_bias_row(
				model.league_market_bias.weight[league_idx].detach().cpu().numpy()
			)
		if getattr(model, "league_market_logit_mixer", None) is not None:
			row["league_market_logit_mixer"] = reshape_mixer(
				model.league_market_logit_mixer.weight[league_idx].detach().cpu().numpy()
			)
		league_rows.append(row)
	payload["per_league"] = league_rows

	if getattr(model, "linear_gate", False):
		gate_weight = model.gate_head.weight.detach().cpu().numpy().reshape(-1)
		gate_bias = float(model.gate_head.bias.detach().cpu().reshape(-1)[0].item())
		market_weight = gate_weight[-7:]
		payload["linear_gate_market_weights"] = {
			name: float(weight)
			for name, weight in zip(payload["gate_market_feature_order"], market_weight)
		}
		payload["linear_gate_head_bias"] = gate_bias

	return payload


def compute_forward_components(model, X: np.ndarray, cat_features: np.ndarray, implied: np.ndarray, raw_margin: np.ndarray) -> dict[str, np.ndarray]:
	device = next(model.parameters()).device
	X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
	cat_tensor = torch.tensor(cat_features, dtype=torch.long, device=device)
	implied_tensor = torch.tensor(implied, dtype=torch.float32, device=device)
	raw_margin_tensor = torch.tensor(raw_margin, dtype=torch.float32, device=device)

	needs_cat = model_requires_cat_features(model, getattr(model, "cat_config", None))
	cat_in = cat_tensor if needs_cat else None

	with torch.no_grad():
		hidden = model.backbone.get_hidden(X_tensor, cat_in)
		residual_logits = model._compute_residual_logits(hidden, cat_in)
		anchor_logits = model._compute_implied_logits(implied_tensor, cat_in)
		gate_logits = model._compute_gate_logits(hidden, implied_tensor, raw_margin_tensor, cat_in)
		gate = torch.sigmoid(gate_logits + model.gate_bias)
		if model.shared_gate:
			gate = gate.expand(-1, model.n_classes)
		applied_residual_logits = gate * residual_logits
		open_logits = anchor_logits + residual_logits
		final_logits = anchor_logits + applied_residual_logits

	return {
		"anchor_logits": anchor_logits.cpu().numpy(),
		"residual_logits": residual_logits.cpu().numpy(),
		"gate": gate.cpu().numpy(),
		"applied_residual_logits": applied_residual_logits.cpu().numpy(),
		"open_logits": open_logits.cpu().numpy(),
		"final_logits": final_logits.cpu().numpy(),
	}


def summarize_stage_comparison(y_true: np.ndarray, raw_probs: np.ndarray, anchor_probs: np.ndarray, open_probs: np.ndarray, final_probs: np.ndarray) -> dict[str, Any]:
	return {
		"raw_market": evaluate_probs(y_true, raw_probs),
		"calibrated_anchor": evaluate_probs(y_true, anchor_probs),
		"anchor_plus_full_residual": evaluate_probs(y_true, open_probs),
		"gated_final": evaluate_probs(y_true, final_probs),
		"log_loss_deltas": {
			"anchor_minus_raw": float(evaluate_probs(y_true, anchor_probs)["log_loss"] - evaluate_probs(y_true, raw_probs)["log_loss"]),
			"open_minus_anchor": float(evaluate_probs(y_true, open_probs)["log_loss"] - evaluate_probs(y_true, anchor_probs)["log_loss"]),
			"final_minus_anchor": float(evaluate_probs(y_true, final_probs)["log_loss"] - evaluate_probs(y_true, anchor_probs)["log_loss"]),
			"final_minus_raw": float(evaluate_probs(y_true, final_probs)["log_loss"] - evaluate_probs(y_true, raw_probs)["log_loss"]),
		},
	}


def summarize_residual_effects(y_true: np.ndarray, anchor_probs: np.ndarray, open_probs: np.ndarray, final_probs: np.ndarray, residual_logits: np.ndarray, applied_residual_logits: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
	residual_norm = np.linalg.norm(residual_logits, axis=1)
	applied_norm = np.linalg.norm(applied_residual_logits, axis=1)
	gate_scalar = gate[:, 0]
	row_index = np.arange(len(y_true))
	anchor_true = anchor_probs[row_index, y_true]
	open_true = open_probs[row_index, y_true]
	final_true = final_probs[row_index, y_true]
	return {
		"gate": summarize_vector(gate_scalar),
		"residual_logit_norm": summarize_vector(residual_norm),
		"applied_residual_logit_norm": summarize_vector(applied_norm),
		"anchor_to_open_shift": summarize_probability_shift(anchor_probs, open_probs),
		"anchor_to_final_shift": summarize_probability_shift(anchor_probs, final_probs),
		"true_class_probability_delta_open_vs_anchor": {
			"mean": float((open_true - anchor_true).mean()),
			"positive_share": float(np.mean(open_true > anchor_true)),
		},
		"true_class_probability_delta_final_vs_anchor": {
			"mean": float((final_true - anchor_true).mean()),
			"positive_share": float(np.mean(final_true > anchor_true)),
		},
	}


def summarize_by_league(league_idx: np.ndarray, y_true: np.ndarray, raw_probs: np.ndarray, anchor_probs: np.ndarray, open_probs: np.ndarray, final_probs: np.ndarray, gate: np.ndarray, residual_logits: np.ndarray, applied_residual_logits: np.ndarray) -> list[dict[str, Any]]:
	rows = []
	for idx in sorted(np.unique(league_idx).tolist()):
		mask = league_idx == idx
		rows.append({
			"league_idx": int(idx),
			"league_name": LEAGUE_NAME_BY_IDX.get(int(idx), f"league_{idx}"),
			"sample_count": int(mask.sum()),
			"metrics": {
				"raw_market": evaluate_probs(y_true[mask], raw_probs[mask]),
				"calibrated_anchor": evaluate_probs(y_true[mask], anchor_probs[mask]),
				"anchor_plus_full_residual": evaluate_probs(y_true[mask], open_probs[mask]),
				"gated_final": evaluate_probs(y_true[mask], final_probs[mask]),
			},
			"gate_mean": float(gate[mask, 0].mean()),
			"residual_norm_mean": float(np.linalg.norm(residual_logits[mask], axis=1).mean()),
			"applied_residual_norm_mean": float(np.linalg.norm(applied_residual_logits[mask], axis=1).mean()),
			"anchor_to_final_mean_total_variation": float(
				(0.5 * np.abs(final_probs[mask] - anchor_probs[mask]).sum(axis=1)).mean()
			),
		})
	return rows


def print_metric_line(name: str, metrics: dict[str, float]):
	print(
		f"{name:24s} "
		f"log_loss={metrics['log_loss']:.6f} "
		f"rps={metrics['rps']:.6f} "
		f"brier={metrics['brier']:.6f} "
		f"acc={metrics['accuracy']:.4f}"
	)


def print_report(summary: dict[str, Any]):
	print("\n=== Bundle ===")
	print(f"Model path: {summary['bundle']['model_path']}")
	print(f"Config path: {summary['bundle']['config_path']}")
	print(f"Test season: {summary['bundle']['test_season']}")
	print(f"Rows: {summary['bundle']['sample_count']}")

	print("\n=== Stage Metrics ===")
	stages = summary["overall"]["stage_metrics"]
	print_metric_line("Raw market", stages["raw_market"])
	print_metric_line("Calibrated anchor", stages["calibrated_anchor"])
	print_metric_line("Anchor + full residual", stages["anchor_plus_full_residual"])
	print_metric_line("Gated final", stages["gated_final"])

	print("\n=== Gate / Residual ===")
	gate_summary = summary["overall"]["residual_effects"]["gate"]
	print(
		"Gate scalar "
		f"mean={gate_summary['mean']:.4f} "
		f"p10={gate_summary['p10']:.4f} "
		f"p50={gate_summary['p50']:.4f} "
		f"p90={gate_summary['p90']:.4f}"
	)
	print(
		"Residual norm "
		f"mean={summary['overall']['residual_effects']['residual_logit_norm']['mean']:.4f} | "
		f"Applied mean={summary['overall']['residual_effects']['applied_residual_logit_norm']['mean']:.4f}"
	)
	print(
		"Anchor->Final mean TV "
		f"{summary['overall']['residual_effects']['anchor_to_final_shift']['mean_total_variation']:.4f} | "
		"Open->Anchor true-class delta "
		f"{summary['overall']['residual_effects']['true_class_probability_delta_open_vs_anchor']['mean']:.4f} | "
		"Final->Anchor true-class delta "
		f"{summary['overall']['residual_effects']['true_class_probability_delta_final_vs_anchor']['mean']:.4f}"
	)

	print("\n=== Calibration Parameters ===")
	cal = summary["calibration"]
	print(f"Global market_logit_scale: {cal['market_logit_scale']:.6f}")
	print(f"Learned gate_bias: {cal['gate_bias']:.6f}")
	if "linear_gate_market_weights" in cal:
		print("Linear gate market weights:")
		for key, value in cal["linear_gate_market_weights"].items():
			print(f"  {key:18s} {value:+.6f}")
	print("Per-league calibration:")
	for row in cal["per_league"]:
		scale = row.get("league_market_scale_multiplier", 1.0)
		class_scale = row.get("league_market_class_scale_multiplier", [1.0, 1.0, 1.0])
		bias = row.get("league_market_bias_centered", [0.0, 0.0, 0.0])
		print(
			f"  {row['league_name']:<20s} "
			f"scale={scale:.4f} "
			f"class_scale={[round(v, 4) for v in class_scale]} "
			f"bias={[round(v, 4) for v in bias]}"
		)

	print("\n=== By League ===")
	for row in summary["by_league"]:
		print(
			f"{row['league_name']:<20s} "
			f"n={row['sample_count']:4d} "
			f"raw_ll={row['metrics']['raw_market']['log_loss']:.6f} "
			f"anchor_ll={row['metrics']['calibrated_anchor']['log_loss']:.6f} "
			f"final_ll={row['metrics']['gated_final']['log_loss']:.6f} "
			f"gate_mean={row['gate_mean']:.4f} "
			f"anchor->final_tv={row['anchor_to_final_mean_total_variation']:.4f}"
		)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
	parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	device = torch.device("cpu")
	bundle = load_model_bundle(RESULT_MODEL_BUNDLE_PATHS, device=device)
	model = bundle.model
	if bundle.metadata.get("model_name") != "gated_residual":
		raise ValueError(
			f"Loaded bundle is not a gated residual model: model_name={bundle.metadata.get('model_name')!r}"
		)

	test_season = bundle.metadata.get("evaluation_protocol", {}).get("test_season")
	if not test_season:
		evaluation_config = load_json(PROJECT_ROOT / "training" / "configs" / "main_models" / "evaluation.json")
		test_season = evaluation_config.get("test_season")

	df = load_frame(args.parquet_path)
	df = filter_min_history(df)
	df = add_targets_and_implied(df)
	df = df.drop_nulls(subset=["odds_home", "odds_draw", "odds_away"])
	test_season = resolve_test_season(df, str(test_season) if test_season else None)
	data = prepare_data(df, bundle.feature_cols, [test_season], scaler=bundle.scaler)

	components = compute_forward_components(
		model=model,
		X=data["X"],
		cat_features=data["cat_features"],
		implied=data["implied"],
		raw_margin=data["raw_margin"],
	)

	raw_probs = data["implied"]
	anchor_probs = probs_from_logits(components["anchor_logits"])
	open_probs = probs_from_logits(components["open_logits"])
	final_probs = probs_from_logits(components["final_logits"])

	stage_metrics = summarize_stage_comparison(
		y_true=data["y"],
		raw_probs=raw_probs,
		anchor_probs=anchor_probs,
		open_probs=open_probs,
		final_probs=final_probs,
	)
	residual_effects = summarize_residual_effects(
		y_true=data["y"],
		anchor_probs=anchor_probs,
		open_probs=open_probs,
		final_probs=final_probs,
		residual_logits=components["residual_logits"],
		applied_residual_logits=components["applied_residual_logits"],
		gate=components["gate"],
	)
	by_league = summarize_by_league(
		league_idx=data["cat_features"][:, 0],
		y_true=data["y"],
		raw_probs=raw_probs,
		anchor_probs=anchor_probs,
		open_probs=open_probs,
		final_probs=final_probs,
		gate=components["gate"],
		residual_logits=components["residual_logits"],
		applied_residual_logits=components["applied_residual_logits"],
	)

	summary = {
		"bundle": {
			"model_path": str(RESULT_MODEL_BUNDLE_PATHS.model_path),
			"config_path": str(RESULT_MODEL_BUNDLE_PATHS.config_path),
			"feature_count": len(bundle.feature_cols),
			"test_season": test_season,
			"sample_count": int(len(data["y"])),
		},
		"calibration": extract_calibration_parameters(model),
		"overall": {
			"stage_metrics": stage_metrics,
			"residual_effects": residual_effects,
		},
		"by_league": by_league,
	}

	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	with open(args.output_json, "w", encoding="utf-8") as file:
		json.dump(summary, file, indent=2)

	print_report(summary)
	print(f"\nWrote JSON summary to {args.output_json}")


if __name__ == "__main__":
	main()
