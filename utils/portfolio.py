"""Portfolio allocation helpers for positive-EV result bets."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_BANKROLL = 100.0
DEFAULT_JOINT_QUADRATURE_ORDER = 64
DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS = 80
DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP = 0.10
DEFAULT_JOINT_OPTIMIZER_MIN_STEP = 1e-6


def normalized_implied_probs(odds_matrix: np.ndarray) -> np.ndarray:
	"""Convert decimal odds to overround-normalized implied probabilities."""

	inv_odds = 1.0 / odds_matrix
	return inv_odds / inv_odds.sum(axis=1, keepdims=True)


def select_best_result_value(
	probs: np.ndarray,
	odds_matrix: np.ndarray,
	implied_probs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
	"""Select the highest-EV outcome for each match and derive allocation signals."""

	if implied_probs is None:
		implied_probs = normalized_implied_probs(odds_matrix)

	best_index = np.argmax(probs * odds_matrix - 1.0, axis=1)
	row_index = np.arange(len(probs))
	selected_probs = probs[row_index, best_index]
	selected_implied = implied_probs[row_index, best_index]
	selected_odds = odds_matrix[row_index, best_index]
	best_ev = selected_probs * selected_odds - 1.0
	positive_mask = best_ev > 0.0
	edge = selected_probs - selected_implied

	with np.errstate(divide="ignore", invalid="ignore"):
		full_kelly = np.where(
			selected_odds > 1.0,
			(selected_probs * selected_odds - 1.0) / (selected_odds - 1.0),
			0.0,
		)
	full_kelly = np.clip(full_kelly, 0.0, None)
	full_kelly = np.where(positive_mask, full_kelly, 0.0)

	return {
		"best_index": best_index,
		"selected_probs": selected_probs,
		"selected_implied": selected_implied,
		"selected_odds": selected_odds,
		"best_ev": best_ev,
		"positive_mask": positive_mask,
		"edge": edge,
		"full_kelly": full_kelly,
	}


@lru_cache(maxsize=4)
def get_joint_quadrature_rule(
	order: int = DEFAULT_JOINT_QUADRATURE_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return Gauss-Laguerre nodes and weights for the joint Kelly objective."""

	nodes, weights = np.polynomial.laguerre.laggauss(int(order))
	return nodes.astype(float), weights.astype(float)


def _project_nonnegative_l1_ball(values: np.ndarray, radius: float) -> np.ndarray:
	"""Project onto the non-negative L1 ball {x >= 0, sum(x) <= radius}."""

	clipped = np.clip(np.asarray(values, dtype=float), 0.0, None)
	if float(clipped.sum()) <= float(radius):
		return clipped

	sorted_values = np.sort(clipped)[::-1]
	cumulative = np.cumsum(sorted_values)
	indices = np.arange(1, len(sorted_values) + 1, dtype=float)
	threshold_candidates = sorted_values - (cumulative - float(radius)) / indices
	rho = int(np.flatnonzero(threshold_candidates > 0.0)[-1])
	theta = (cumulative[rho] - float(radius)) / float(rho + 1)
	return np.clip(clipped - theta, 0.0, None)


def _stable_joint_terms(
	weights: np.ndarray,
	selected_probs: np.ndarray,
	selected_odds: np.ndarray,
	nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Compute stable node-wise log-products and gradient ratios."""

	weights = np.asarray(weights, dtype=float)
	selected_probs = np.asarray(selected_probs, dtype=float)
	win_returns = np.asarray(selected_odds, dtype=float) - 1.0

	scaled_loss = nodes[:, None] * weights[None, :]
	scaled_win = -nodes[:, None] * win_returns[None, :] * weights[None, :]
	shift = np.maximum(scaled_loss, scaled_win)

	loss_term = (1.0 - selected_probs.reshape(1, -1)) * np.exp(scaled_loss - shift)
	win_term = selected_probs.reshape(1, -1) * np.exp(scaled_win - shift)
	denominator = loss_term + win_term
	log_product = np.sum(shift + np.log(denominator), axis=1)
	ratio = (loss_term - win_returns.reshape(1, -1) * win_term) / denominator
	return log_product, ratio


def _joint_expected_log_growth_and_grad(
	weights: np.ndarray,
	selected_probs: np.ndarray,
	selected_odds: np.ndarray,
	quadrature_nodes: np.ndarray,
	quadrature_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
	"""Evaluate expected log growth and gradient deterministically for any slate size."""

	if weights.size == 0:
		return 0.0, np.zeros(0, dtype=float)

	log_product, ratio = _stable_joint_terms(
		weights=weights,
		selected_probs=selected_probs,
		selected_odds=selected_odds,
		nodes=quadrature_nodes,
	)

	log_weighted_product = np.log(quadrature_weights) + log_product
	log_weighted_product_over_node = log_weighted_product - np.log(quadrature_nodes)
	constant_term = float(np.sum(quadrature_weights / quadrature_nodes))

	second_shift = float(np.max(log_weighted_product_over_node))
	second_sum = float(
		np.exp(second_shift) * np.sum(np.exp(log_weighted_product_over_node - second_shift))
	)
	value = constant_term - second_sum

	gradient_shift = float(np.max(log_weighted_product))
	weighted_products = np.exp(log_weighted_product - gradient_shift)
	gradient = -np.exp(gradient_shift) * np.sum(weighted_products[:, None] * ratio, axis=0)
	return value, gradient


def _optimize_joint_weights(
	full_kelly: np.ndarray,
	selected_probs: np.ndarray,
	selected_odds: np.ndarray,
	quadrature_order: int = DEFAULT_JOINT_QUADRATURE_ORDER,
	max_iterations: int = DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS,
	initial_step: float = DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP,
	min_step: float = DEFAULT_JOINT_OPTIMIZER_MIN_STEP,
) -> np.ndarray:
	"""Optimize one-side-per-game Kelly weights jointly over the active slate."""

	if len(full_kelly) == 0:
		return np.zeros(0, dtype=float)

	quadrature_nodes, quadrature_weights = get_joint_quadrature_rule(quadrature_order)
	weights = _project_nonnegative_l1_ball(np.asarray(full_kelly, dtype=float), radius=1.0 - 1e-12)
	step = max(float(initial_step), float(min_step))
	current_value, current_grad = _joint_expected_log_growth_and_grad(
		weights=weights,
		selected_probs=selected_probs,
		selected_odds=selected_odds,
		quadrature_nodes=quadrature_nodes,
		quadrature_weights=quadrature_weights,
	)

	for _ in range(int(max_iterations)):
		candidate = _project_nonnegative_l1_ball(weights + step * current_grad, radius=1.0 - 1e-12)
		next_value, next_grad = _joint_expected_log_growth_and_grad(
			weights=candidate,
			selected_probs=selected_probs,
			selected_odds=selected_odds,
			quadrature_nodes=quadrature_nodes,
			quadrature_weights=quadrature_weights,
		)
		if np.isfinite(next_value) and next_value >= current_value:
			weights = candidate
			current_value = next_value
			current_grad = next_grad
			step *= 1.05
		else:
			step *= 0.5
		if step < float(min_step):
			break

	return weights


def allocate_bankroll_kelly(
	selection: dict[str, np.ndarray],
	total_bankroll: float,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	quadrature_order: int = DEFAULT_JOINT_QUADRATURE_ORDER,
	max_iterations: int = DEFAULT_JOINT_OPTIMIZER_MAX_ITERATIONS,
	initial_step: float = DEFAULT_JOINT_OPTIMIZER_INITIAL_STEP,
	min_step: float = DEFAULT_JOINT_OPTIMIZER_MIN_STEP,
) -> dict[str, Any]:
	"""Allocate bankroll with a deterministic one-side-per-game joint Kelly optimizer."""

	positive_mask = np.asarray(selection["positive_mask"], dtype=bool)
	raw_weights = np.zeros_like(selection["best_ev"], dtype=float)
	if not np.any(positive_mask):
		return {
			"strategy": "joint_bankroll_kelly",
			"kelly_fraction": float(kelly_fraction),
			"raw_weights": raw_weights,
			"stake_shares": raw_weights.copy(),
			"stake_amounts": raw_weights.copy(),
			"allocated_budget": 0.0,
		}

	active_idx = np.flatnonzero(positive_mask)
	selected_probs = np.asarray(selection["selected_probs"], dtype=float)[active_idx]
	selected_odds = np.asarray(selection["selected_odds"], dtype=float)[active_idx]
	full_kelly = np.asarray(selection["full_kelly"], dtype=float)[active_idx]

	if len(active_idx) == 1:
		solution = np.asarray([full_kelly[0]], dtype=float)
	else:
		solution = _optimize_joint_weights(
			full_kelly=full_kelly,
			selected_probs=selected_probs,
			selected_odds=selected_odds,
			quadrature_order=quadrature_order,
			max_iterations=max_iterations,
			initial_step=initial_step,
			min_step=min_step,
		)

	scaled_solution = solution * max(0.0, float(kelly_fraction))
	scaled_solution = _project_nonnegative_l1_ball(scaled_solution, radius=1.0 - 1e-12)
	raw_weights[active_idx] = scaled_solution
	stake_shares = raw_weights.copy()
	stake_amounts = stake_shares * max(0.0, float(total_bankroll))
	return {
		"strategy": "joint_bankroll_kelly",
		"kelly_fraction": float(kelly_fraction),
		"raw_weights": raw_weights,
		"stake_shares": stake_shares,
		"stake_amounts": stake_amounts,
		"allocated_budget": float(stake_amounts.sum()),
	}


def _normalize_groups(groups: np.ndarray | None, size: int) -> np.ndarray:
	"""Convert arbitrary group labels to stable strings."""

	if groups is None:
		return np.zeros(size, dtype=object)

	array = np.asarray(groups)
	if array.shape[0] != size:
		raise ValueError(f"Group label count {array.shape[0]} does not match sample count {size}.")

	normalized = []
	for value in array:
		if hasattr(value, "strftime"):
			normalized.append(value.strftime("%Y-%m-%d"))
		else:
			text = str(value)
			normalized.append(text[:10] if "T" in text else text)
	return np.asarray(normalized, dtype=object)


def evaluate_bankroll_strategy(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_home: np.ndarray,
	odds_draw: np.ndarray,
	odds_away: np.ndarray,
	groups: np.ndarray | None = None,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	initial_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, float]:
	"""Evaluate a bankroll-compounding joint Kelly strategy over grouped fixtures."""

	odds_matrix = np.stack([odds_home, odds_draw, odds_away], axis=1)
	implied_probs = normalized_implied_probs(odds_matrix)
	selection = select_best_result_value(probs, odds_matrix, implied_probs=implied_probs)
	group_labels = _normalize_groups(groups, size=len(y_true))
	stake_amounts = np.zeros(len(y_true), dtype=float)
	starting_bankroll = max(0.0, float(initial_bankroll))
	current_bankroll = starting_bankroll
	peak_bankroll = starting_bankroll
	max_drawdown = 0.0

	for group in np.unique(group_labels):
		if current_bankroll <= 0.0:
			break
		mask = group_labels == group
		allocation = allocate_bankroll_kelly(
			selection={
				key: value[mask] if isinstance(value, np.ndarray) else value
				for key, value in selection.items()
			},
			total_bankroll=current_bankroll,
			kelly_fraction=kelly_fraction,
		)
		group_stakes = allocation["stake_amounts"]
		if float(group_stakes.sum()) <= 0.0:
			continue
		stake_amounts[mask] = group_stakes

		group_outcomes = selection["best_index"][mask]
		group_odds = selection["selected_odds"][mask]
		group_truth = y_true[mask]
		group_profit = np.where(group_outcomes == group_truth, group_stakes * (group_odds - 1.0), -group_stakes)
		current_bankroll += float(group_profit.sum())
		peak_bankroll = max(peak_bankroll, current_bankroll)
		if peak_bankroll > 0.0:
			max_drawdown = max(max_drawdown, (peak_bankroll - current_bankroll) / peak_bankroll)

	n_staked_bets = int(np.count_nonzero(stake_amounts > 0.0))
	roi = (current_bankroll - starting_bankroll) / starting_bankroll if starting_bankroll > 0.0 else 0.0
	return {
		"bankroll_roi": float(roi),
		"bankroll_bet_count": n_staked_bets,
		"max_drawdown": float(max_drawdown),
	}
