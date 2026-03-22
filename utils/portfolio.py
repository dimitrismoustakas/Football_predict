"""Portfolio allocation helpers for positive-EV result bets."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_BANKROLL = 100.0


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


def allocate_bankroll_kelly(
	selection: dict[str, np.ndarray],
	total_bankroll: float,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Any]:
	"""Allocate proper partial Kelly stakes from a bankroll without forcing full deployment."""

	raw_fractions = np.where(
		selection["positive_mask"],
		selection["full_kelly"] * max(0.0, float(kelly_fraction)),
		0.0,
	)
	raw_fractions = np.clip(raw_fractions, 0.0, None)
	fraction_sum = float(raw_fractions.sum())
	scale = 1.0 / fraction_sum if fraction_sum > 1.0 and fraction_sum > 0.0 else 1.0
	stake_shares = raw_fractions * scale
	stake_amounts = stake_shares * max(0.0, float(total_bankroll))
	return {
		"strategy": "bankroll_kelly",
		"kelly_fraction": float(kelly_fraction),
		"raw_weights": raw_fractions,
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
	"""Evaluate a bankroll-compounding partial Kelly strategy over grouped fixtures."""

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
