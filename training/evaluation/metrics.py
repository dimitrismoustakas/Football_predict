"""
Evaluation metrics for football prediction models.
"""


from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from utils.portfolio import evaluate_portfolio


def _logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert probability to logits with numerical stability."""
	p = torch.clamp(p, eps, 1 - eps)
	return torch.log(p) - torch.log(1 - p)


def evaluate_profit(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
) -> Dict:
	"""Evaluates profit based on value betting: Bet if Model_Prob * Odds - 1 > 0"""
	ev_over = probs * odds_over - 1
	ev_under = (1 - probs) * odds_under - 1

	bets_over = ev_over > 0
	bets_under = ev_under > 0

	profit_over_outcomes = np.where(y_true == 1, odds_over - 1, -1)
	profit_under_outcomes = np.where(y_true == 0, odds_under - 1, -1)

	actual_profit_over = profit_over_outcomes[bets_over]
	actual_profit_under = profit_under_outcomes[bets_under]

	total_profit = np.sum(actual_profit_over) + np.sum(actual_profit_under)
	n_bets = len(actual_profit_over) + len(actual_profit_under)

	return {
		"total_profit": float(total_profit),
		"avg_profit": float(total_profit / n_bets) if n_bets > 0 else 0.0,
		"n_bets": int(n_bets),
		"percent_bets": float((n_bets / len(y_true)) * 100),
	}


def evaluate_model(
	model: nn.Module, 
	data: Dict[str, np.ndarray], 
	device: torch.device = None,
	verbose: bool = True,
) -> Dict:
	"""Full evaluation of model on a dataset."""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
	model.eval()
	X = torch.tensor(data["X"], dtype=torch.float32).to(device)
	implied = torch.tensor(data["implied"], dtype=torch.float32).unsqueeze(1).to(device)

	with torch.no_grad():
		residual_logits = model(X)
		logits = _logits(implied) + residual_logits
		prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()

	y_true = data["y"]
	implied_np = data["implied"]

	preds = (prob >= 0.5).astype(int)
	acc = accuracy_score(y_true, preds)
	brier = brier_score_loss(y_true, prob)
	ll = log_loss(y_true, np.c_[1 - prob, prob], labels=[0, 1])
	corr = float(np.corrcoef(prob, implied_np)[0, 1])

	profit_metrics = evaluate_profit(prob, y_true, data["odds_over"], data["odds_under"])
	portfolio_metrics = evaluate_portfolio(
		prob, y_true, data["odds_over"], data["odds_under"], data["dates"]
	)

	if verbose:
		print(
			f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, LogLoss: {ll:.4f}, Corr: {corr:.4f}"
		)
		print(
			f"Profit: {profit_metrics['n_bets']} bets ({profit_metrics['percent_bets']:.1f}%), "
			f"Total: {profit_metrics['total_profit']:.2f}"
		)
		print(
			f"Portfolio Sharpe: {portfolio_metrics['sharpe_ratio']:.4f}, "
			f"Total: {portfolio_metrics['sharpe_total_profit']:.2f}"
		)

	return {
		"accuracy": float(acc),
		"brier": float(brier),
		"log_loss": float(ll),
		"corr_with_implied": float(corr),
		**profit_metrics,
		**portfolio_metrics,
	}


def plot_losses(history: Dict, title: str, filepath: Path):
	"""Plot and save training/validation loss curves."""
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
