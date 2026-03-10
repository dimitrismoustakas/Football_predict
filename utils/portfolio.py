"""Portfolio allocation helpers for positive-EV result bets."""

from __future__ import annotations

from typing import Any

import numpy as np

RESULT_BUDGET_STRATEGIES = {"flat", "edge", "kelly"}
DEFAULT_BUDGET_STRATEGY = "kelly"
DEFAULT_KELLY_FRACTION = 0.5


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


def _resolve_budget_weights(selection: dict[str, np.ndarray], strategy: str, kelly_fraction: float) -> np.ndarray:
	"""Build raw positive weights before normalizing to a fixed budget."""

	resolved = strategy.strip().lower()
	if resolved not in RESULT_BUDGET_STRATEGIES:
		raise ValueError(f"Unsupported budget strategy '{strategy}'. Expected one of {sorted(RESULT_BUDGET_STRATEGIES)}.")

	positive_mask = selection["positive_mask"]
	if resolved == "flat":
		weights = positive_mask.astype(float)
	elif resolved == "edge":
		weights = np.where(positive_mask, np.clip(selection["edge"], 0.0, None), 0.0)
	else:
		weights = np.where(positive_mask, selection["full_kelly"] * max(0.0, float(kelly_fraction)), 0.0)

	if positive_mask.any() and not np.any(weights > 0.0):
		weights = positive_mask.astype(float)
	return weights


def allocate_fixed_budget(
	selection: dict[str, np.ndarray],
	total_budget: float,
	strategy: str = DEFAULT_BUDGET_STRATEGY,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Any]:
	"""Split a fixed budget across positive-EV bets using the chosen strategy."""

	weights = _resolve_budget_weights(selection, strategy=strategy, kelly_fraction=kelly_fraction)
	weight_sum = float(weights.sum())
	stake_shares = np.zeros_like(weights, dtype=float)
	if weight_sum > 0.0:
		stake_shares = weights / weight_sum
	stake_amounts = stake_shares * float(total_budget)
	return {
		"strategy": strategy.strip().lower(),
		"kelly_fraction": float(kelly_fraction),
		"raw_weights": weights,
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


def evaluate_budget_strategy(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_home: np.ndarray,
	odds_draw: np.ndarray,
	odds_away: np.ndarray,
	groups: np.ndarray | None = None,
	strategy: str = DEFAULT_BUDGET_STRATEGY,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	group_budget: float = 1.0,
) -> dict[str, float]:
	"""Evaluate fixed-budget allocation over all positive-EV selections."""

	odds_matrix = np.stack([odds_home, odds_draw, odds_away], axis=1)
	implied_probs = normalized_implied_probs(odds_matrix)
	selection = select_best_result_value(probs, odds_matrix, implied_probs=implied_probs)
	group_labels = _normalize_groups(groups, size=len(y_true))
	stake_amounts = np.zeros(len(y_true), dtype=float)
	total_profit = 0.0
	total_staked = 0.0
	active_groups = 0

	for group in np.unique(group_labels):
		mask = group_labels == group
		allocation = allocate_fixed_budget(
			selection={
				key: value[mask] if isinstance(value, np.ndarray) else value
				for key, value in selection.items()
			},
			total_budget=group_budget,
			strategy=strategy,
			kelly_fraction=kelly_fraction,
		)
		group_stakes = allocation["stake_amounts"]
		if float(group_stakes.sum()) <= 0.0:
			continue
		active_groups += 1
		stake_amounts[mask] = group_stakes

		group_outcomes = selection["best_index"][mask]
		group_odds = selection["selected_odds"][mask]
		group_truth = y_true[mask]
		group_profit = np.where(group_outcomes == group_truth, group_stakes * (group_odds - 1.0), -group_stakes)
		total_profit += float(group_profit.sum())
		total_staked += float(group_stakes.sum())

	n_staked_bets = int(np.count_nonzero(stake_amounts > 0.0))
	avg_stake = float(stake_amounts[stake_amounts > 0.0].mean()) if n_staked_bets else 0.0
	max_stake = float(stake_amounts.max()) if n_staked_bets else 0.0
	roi = total_profit / total_staked if total_staked > 0.0 else 0.0
	return {
		"budget_profit": float(total_profit),
		"budget_roi": float(roi),
		"budget_total_staked": float(total_staked),
		"budget_bet_count": n_staked_bets,
		"budget_active_groups": int(active_groups),
		"budget_avg_stake": avg_stake,
		"budget_max_stake": max_stake,
	}
