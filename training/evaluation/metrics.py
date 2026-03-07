"""
Evaluation metrics for football prediction models.

Supports both binary (over/under) and multiclass (home/draw/away) tasks.
"""


from pathlib import Path
from typing import Dict, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from utils.portfolio import evaluate_portfolio


def ranked_probability_score(y_true: np.ndarray, probs: np.ndarray) -> float:
	"""
	Compute Ranked Probability Score (RPS) for ordinal outcomes.
	
	RPS measures how well predicted probabilities match actual ordinal outcomes.
	For football results, outcomes are treated as ordered: Home(0) < Draw(1) < Away(2).
	This ordering reflects that a draw is conceptually "between" home and away wins.
	
	RPS = (1/n) * sum_i (1/(K-1)) * sum_k (CDF_pred[k] - CDF_actual[k])^2
	
	where CDF is cumulative distribution function.
	
	IMPORTANT: Input ordering must be consistent:
	- probs columns: [home_prob, draw_prob, away_prob] (indices 0, 1, 2)
	- y_true labels: 0=Home win, 1=Draw, 2=Away win
	
	Args:
		y_true: Shape (n,) with class labels 0=Home, 1=Draw, 2=Away
		probs: Shape (n, 3) with [home, draw, away] probabilities
	
	Returns:
		Mean RPS across all samples (lower is better, range [0, 1])
	"""
	n_classes = 3
	
	# Create one-hot encoding of true outcomes
	y_onehot = np.eye(n_classes)[y_true]  # (n, 3)
	
	# Compute cumulative distributions
	cdf_pred = np.cumsum(probs, axis=1)  # (n, 3)
	cdf_actual = np.cumsum(y_onehot, axis=1)  # (n, 3)
	
	# RPS for each sample: (1/(K-1)) * sum((CDF_pred - CDF_actual)^2)
	# We sum over K-1 positions (exclude the last which is always 1)
	rps_per_sample = np.sum((cdf_pred[:, :-1] - cdf_actual[:, :-1]) ** 2, axis=1) / (n_classes - 1)
	
	return float(np.mean(rps_per_sample))


TaskType = Literal["binary", "multiclass"]


def _logits(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert probability to logits with numerical stability (binary)."""
	p = torch.clamp(p, eps, 1 - eps)
	return torch.log(p) - torch.log(1 - p)


def _log_softmax_from_implied(implied_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
	"""Convert implied probabilities to log-softmax values (multiclass)."""
	implied_probs = torch.clamp(implied_probs, eps, 1.0 - eps)
	implied_probs = implied_probs / implied_probs.sum(dim=-1, keepdim=True)
	return torch.log(implied_probs)


def evaluate_profit(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
) -> Dict:
	"""Evaluates profit based on value betting for over/under: Bet if Model_Prob * Odds - 1 > 0"""
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


def evaluate_profit_result(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_home: np.ndarray,
	odds_draw: np.ndarray,
	odds_away: np.ndarray,
) -> Dict:
	"""
	Evaluates profit for result prediction (home/draw/away).
	
	Value betting: For each game, bet on the outcome with highest EV,
	but only if that EV > 0. At most one bet per game.
	
	Args:
		probs: Shape (n, 3) with [home, draw, away] probabilities
		y_true: Shape (n,) with class labels 0=Home, 1=Draw, 2=Away
		odds_*: Shape (n,) with decimal odds for each outcome
	"""
	odds_matrix = np.stack([odds_home, odds_draw, odds_away], axis=1)  # (n, 3)
	
	# Expected value for each outcome
	ev = probs * odds_matrix - 1  # (n, 3)
	
	# For each game, find the outcome with highest EV
	best_outcome = np.argmax(ev, axis=1)  # (n,)
	best_ev = ev[np.arange(len(ev)), best_outcome]  # (n,)
	
	# Only bet if best EV > 0
	bet_mask = best_ev > 0  # (n,)
	
	n_samples = len(y_true)
	n_bets = int(np.sum(bet_mask))
	
	# Calculate profit for bets placed
	if n_bets > 0:
		bet_outcomes = best_outcome[bet_mask]  # which outcome we bet on
		actual_outcomes = y_true[bet_mask]  # what actually happened
		bet_odds = odds_matrix[bet_mask, bet_outcomes]  # odds for our bets
		
		# Win if bet_outcome == actual_outcome
		wins = bet_outcomes == actual_outcomes
		profits = np.where(wins, bet_odds - 1, -1)
		total_profit = float(np.sum(profits))
		
		# Count bets by outcome type
		n_home_bets = int(np.sum(bet_outcomes == 0))
		n_draw_bets = int(np.sum(bet_outcomes == 1))
		n_away_bets = int(np.sum(bet_outcomes == 2))
	else:
		total_profit = 0.0
		n_home_bets = n_draw_bets = n_away_bets = 0
	
	return {
		"total_profit": total_profit,
		"avg_profit": total_profit / n_bets if n_bets > 0 else 0.0,
		"n_bets": n_bets,
		"percent_bets": float((n_bets / n_samples) * 100) if n_samples > 0 else 0.0,
		"n_home_bets": n_home_bets,
		"n_draw_bets": n_draw_bets,
		"n_away_bets": n_away_bets,
	}


def evaluate_model(
	model: nn.Module, 
	data: Dict[str, np.ndarray], 
	device: torch.device = None,
	verbose: bool = True,
	task_type: TaskType = "binary",
) -> Dict:
	"""
	Full evaluation of model on a dataset.
	
	Args:
		model: Trained neural network
		data: Dict with 'X', 'y', 'implied', 'cat_features' (optional), odds columns, 'dates'
		device: Target device
		verbose: Whether to print metrics
		task_type: "binary" for over/under, "multiclass" for result
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	model = model.to(device)
	model.eval()
	X = torch.tensor(data["X"], dtype=torch.float32).to(device)
	
	# Categorical features are optional; all in-repo models accept cat_features as an optional arg.
	cat_features = None
	if "cat_features" in data:
		cat_features = torch.tensor(data["cat_features"], dtype=torch.long).to(device)
	
	if task_type == "binary":
		return _evaluate_model_binary(model, X, cat_features, data, device, verbose)
	else:
		return _evaluate_model_multiclass(model, X, cat_features, data, device, verbose)


def _model_forward(
	model: nn.Module,
	X: torch.Tensor,
	cat_features: torch.Tensor,
	implied: torch.Tensor,
	raw_margin: torch.Tensor,
) -> torch.Tensor:
	"""Forward pass that respects gated models when market features are available."""
	use_market = implied is not None and raw_margin is not None
	if use_market and hasattr(model, "gate_head"):
		return model(X, cat_features, implied, raw_margin)
	return model(X, cat_features)


def _evaluate_model_binary(
	model: nn.Module,
	X: torch.Tensor,
	cat_features: torch.Tensor,
	data: Dict[str, np.ndarray],
	device: torch.device,
	verbose: bool,
) -> Dict:
	"""Evaluate binary classification model (over/under)."""
	implied = torch.tensor(data["implied"], dtype=torch.float32).to(device)
	raw_margin = torch.tensor(data["raw_margin"], dtype=torch.float32).to(device)

	with torch.no_grad():
		pred_logits = _model_forward(model, X, cat_features, implied, raw_margin)
		prob = torch.sigmoid(pred_logits).view(-1).cpu().numpy()

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


def _evaluate_model_multiclass(
	model: nn.Module,
	X: torch.Tensor,
	cat_features: torch.Tensor,
	data: Dict[str, np.ndarray],
	device: torch.device,
	verbose: bool,
) -> Dict:
	"""Evaluate multiclass model (home/draw/away result)."""
	implied = torch.tensor(data["implied"], dtype=torch.float32).to(device)  # (n, 3)
	raw_margin = torch.tensor(data["raw_margin"], dtype=torch.float32).to(device)

	with torch.no_grad():
		pred_logits = _model_forward(model, X, cat_features, implied, raw_margin)
		probs = F.softmax(pred_logits, dim=-1).cpu().numpy()  # (n, 3)

	y_true = data["y"]
	implied_np = data["implied"]

	preds = np.argmax(probs, axis=1)
	acc = accuracy_score(y_true, preds)
	
	# Multiclass Brier score: mean squared error of one-hot vs probs
	n_classes = 3
	y_onehot = np.eye(n_classes)[y_true]
	brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
	
	ll = log_loss(y_true, probs, labels=[0, 1, 2])
	
	# Correlation: compute mean correlation across classes
	corr_per_class = []
	for c in range(n_classes):
		corr_c = np.corrcoef(probs[:, c], implied_np[:, c])[0, 1]
		corr_per_class.append(corr_c)
	corr = float(np.mean(corr_per_class))
	
	# Ranked Probability Score
	rps = ranked_probability_score(y_true, probs)

	profit_metrics = evaluate_profit_result(
		probs, y_true, data["odds_home"], data["odds_draw"], data["odds_away"]
	)

	if verbose:
		print(
			f"Accuracy: {acc:.4f}, Brier: {brier:.4f}, RPS: {rps:.4f}, LogLoss: {ll:.4f}, AvgCorr: {corr:.4f}"
		)
		print(
			f"Profit: {profit_metrics['n_bets']} bets ({profit_metrics['percent_bets']:.1f}%), "
			f"Total: {profit_metrics['total_profit']:.2f}"
		)
		print(
			f"  Home bets: {profit_metrics['n_home_bets']}, "
			f"Draw bets: {profit_metrics['n_draw_bets']}, "
			f"Away bets: {profit_metrics['n_away_bets']}"
		)

	return {
		"accuracy": float(acc),
		"brier": brier,
		"rps": rps,
		"log_loss": float(ll),
		"corr_with_implied": corr,
		**profit_metrics,
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
