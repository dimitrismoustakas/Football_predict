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


def evaluate_portfolio(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
) -> Dict:
	"""Portfolio-style evaluation with Sharpe-weighted betting."""
	df = pd.DataFrame(
		{
			"date": dates,
			"prob_over": probs,
			"y_true": y_true,
			"odds_over": odds_over,
			"odds_under": odds_under,
		}
	)

	p = df["prob_over"]
	o_over = df["odds_over"]
	o_under = df["odds_under"]

	# Expected value calculations
	mu_over = p * o_over - 1
	mu_under = (1 - p) * o_under - 1

	# Variance calculations
	e_x2_over = p * (o_over - 1) ** 2 + (1 - p) * 1
	var_over = e_x2_over - mu_over**2
	e_x2_under = (1 - p) * (o_under - 1) ** 2 + p * 1
	var_under = e_x2_under - mu_under**2

	better_is_over = mu_over >= mu_under
	mu_best = np.where(better_is_over, mu_over, mu_under)
	var_best = np.where(better_is_over, var_over, var_under)
	odds_best = np.where(better_is_over, o_over, o_under)

	df["bet_over"] = better_is_over
	df["mu"] = mu_best
	df["var"] = var_best
	df["odds"] = odds_best
	df["won"] = np.where(df["bet_over"], df["y_true"] == 1, df["y_true"] == 0)
	df["eligible"] = df["mu"] > 0

	daily_results_sharpe = []

	for _, group in df.groupby("date"):
		group = group[group["eligible"]]
		if len(group) <= 1:
			daily_results_sharpe.append(0.0)
			continue

		mus = group["mu"].values
		vars_ = group["var"].values + 1e-6
		raw_weights = np.maximum(0, mus / vars_)
		sum_weights = raw_weights.sum()

		if sum_weights > 0:
			norm_weights = raw_weights / sum_weights
			bets_sharpe = budget_per_day * norm_weights
			profits = np.where(
				group["won"].values,
				bets_sharpe * (group["odds"].values - 1),
				-bets_sharpe,
			)
			daily_results_sharpe.append(profits.sum())
		else:
			daily_results_sharpe.append(0.0)

	daily = np.array(daily_results_sharpe)

	def sharpe_ratio(x):
		if len(x) == 0:
			return 0.0
		std = x.std()
		return float(x.mean() / std) if std > 0 else 0.0

	return {
		"sharpe_total_profit": float(daily.sum()),
		"sharpe_avg_daily_profit": float(daily.mean()) if len(daily) > 0 else 0.0,
		"sharpe_ratio": sharpe_ratio(daily),
		"n_days": int(len(daily)),
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
