"""
Evaluation metrics for match-result prediction.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, log_loss

from training.result_modeling import predict_result_proba_from_data


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
	"""Compute Ranked Probability Score for Home/Draw/Away outcomes."""

	n_classes = 3
	y_onehot = np.eye(n_classes)[y_true]
	cdf_pred = np.cumsum(probs, axis=1)
	cdf_actual = np.cumsum(y_onehot, axis=1)
	rps_per_sample = np.sum((cdf_pred[:, :-1] - cdf_actual[:, :-1]) ** 2, axis=1) / (n_classes - 1)
	return float(np.mean(rps_per_sample))


def evaluate_profit(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_home: np.ndarray,
	odds_draw: np.ndarray,
	odds_away: np.ndarray,
) -> Dict:
	"""Evaluate one best-EV result bet per match when EV is positive."""

	odds_matrix = np.stack([odds_home, odds_draw, odds_away], axis=1)
	ev = probs * odds_matrix - 1
	best_outcome = np.argmax(ev, axis=1)
	best_ev = ev[np.arange(len(ev)), best_outcome]
	bet_mask = best_ev > 0
	bet_indices = np.flatnonzero(bet_mask)
	n_samples = len(y_true)
	n_bets = int(bet_mask.sum())

	if n_bets == 0:
		return {
			"total_profit": 0.0,
			"avg_profit": 0.0,
			"n_bets": 0,
			"percent_bets": 0.0,
			"n_home_bets": 0,
			"n_draw_bets": 0,
			"n_away_bets": 0,
		}

	bet_outcomes = best_outcome[bet_indices]
	actual_outcomes = y_true[bet_indices]
	bet_odds = odds_matrix[bet_indices, bet_outcomes]
	wins = bet_outcomes == actual_outcomes
	profits = np.where(wins, bet_odds - 1, -1)
	total_profit = float(np.sum(profits))

	return {
		"total_profit": total_profit,
		"avg_profit": total_profit / n_bets,
		"n_bets": n_bets,
		"percent_bets": float((n_bets / n_samples) * 100) if n_samples > 0 else 0.0,
		"n_home_bets": int(np.sum(bet_outcomes == 0)),
		"n_draw_bets": int(np.sum(bet_outcomes == 1)),
		"n_away_bets": int(np.sum(bet_outcomes == 2)),
	}


def evaluate_model(
	model,
	data: Dict[str, np.ndarray],
	device: torch.device = None,
	verbose: bool = True,
	metadata: dict | None = None,
	scaler=None,
) -> Dict:
	"""Full evaluation for the match-result model."""

	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if metadata is None:
		metadata = {"model_family": "gated_residual"}

	probs = predict_result_proba_from_data(
		model=model,
		data=data,
		metadata=metadata,
		scaler=scaler,
		device=device,
	)

	y_true = data["y"]
	implied_np = data["implied"]
	preds = np.argmax(probs, axis=1)
	acc = accuracy_score(y_true, preds)
	y_onehot = np.eye(3)[y_true]
	brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
	ll = log_loss(y_true, probs, labels=[0, 1, 2])
	corr_per_class = []
	for class_idx in range(3):
		corr = np.corrcoef(probs[:, class_idx], implied_np[:, class_idx])[0, 1]
		corr_per_class.append(0.0 if np.isnan(corr) else corr)
	avg_corr = float(np.mean(corr_per_class))
	rps = ranked_probability_score(y_true, probs)
	profit_metrics = evaluate_profit(
		probs,
		y_true,
		data["odds_home"],
		data["odds_draw"],
		data["odds_away"],
	)

	if verbose:
		print(
			f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, RPS: {rps:.4f}, LogLoss: {ll:.4f}, AvgCorr: {avg_corr:.4f}"
		)
		print(
			f"Profit: {profit_metrics['n_bets']} bets ({profit_metrics['percent_bets']:.1f}%), Total: {profit_metrics['total_profit']:.2f}"
		)
		print(
			f"  Home bets: {profit_metrics['n_home_bets']}, Draw bets: {profit_metrics['n_draw_bets']}, Away bets: {profit_metrics['n_away_bets']}"
		)

	return {
		"accuracy": float(acc),
		"brier": brier,
		"rps": rps,
		"log_loss": float(ll),
		"corr_with_implied": avg_corr,
		**profit_metrics,
	}


def plot_losses(history: Dict, title: str, filepath: Path):
	"""Plot and save training and validation loss curves."""

	plt.figure(figsize=(10, 6))
	plt.plot(history["train_loss"], label="Train Loss")
	if history["val_loss"]:
		plt.plot(history["val_loss"], label="Val Loss")
	plt.title(f"Loss Curve - {title}")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)
	plt.savefig(filepath)
	plt.close()
