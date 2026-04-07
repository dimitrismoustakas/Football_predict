"""Exact evaluation helpers for named system bets over selected match legs."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any

import numpy as np

from utils.portfolio import DEFAULT_BANKROLL, DEFAULT_KELLY_FRACTION

SYSTEM_SPECS = {
	"2/3": {
		"selection_count": 3,
		"line_sizes": (2, 3),
	},
	"2/3/4": {
		"selection_count": 4,
		"line_sizes": (2, 3, 4),
	},
}
_SYSTEM_BISECTION_TOL = 1e-10
_SYSTEM_BISECTION_MAX_ITERATIONS = 96
_SYSTEM_MAX_STAKE = 1.0 - 1e-12


def _resolve_system_spec(system_name: str) -> dict[str, Any]:
	if system_name not in SYSTEM_SPECS:
		raise ValueError(f"Unsupported system bet {system_name!r}. Expected one of {sorted(SYSTEM_SPECS)}.")
	return SYSTEM_SPECS[system_name]


@lru_cache(maxsize=None)
def _line_index_sets(selection_count: int, line_sizes: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
	return tuple(
		combo
		for line_size in line_sizes
		for combo in combinations(range(int(selection_count)), int(line_size))
	)


@lru_cache(maxsize=None)
def _local_win_states(selection_count: int) -> np.ndarray:
	state_ids = np.arange(1 << int(selection_count), dtype=np.uint32)[:, None]
	bit_offsets = np.arange(int(selection_count), dtype=np.uint32)
	states = ((state_ids >> bit_offsets) & 1).astype(bool)
	states.flags.writeable = False
	return states


def _normalize_groups(groups: np.ndarray | None, size: int) -> np.ndarray:
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


def _state_probabilities(selected_probs: np.ndarray) -> np.ndarray:
	states = _local_win_states(len(selected_probs))
	probs = np.asarray(selected_probs, dtype=float).reshape(1, -1)
	return np.prod(np.where(states, probs, 1.0 - probs), axis=1)


def _system_gross_returns(selected_odds: np.ndarray, system_name: str) -> np.ndarray:
	spec = _resolve_system_spec(system_name)
	selection_count = int(spec["selection_count"])
	line_sets = _line_index_sets(selection_count, tuple(spec["line_sizes"]))
	states = _local_win_states(selection_count)
	gross_returns = np.zeros(states.shape[0], dtype=float)
	local_odds = np.asarray(selected_odds, dtype=float)
	for line in line_sets:
		gross_returns += float(np.prod(local_odds[list(line)])) * states[:, line].all(axis=1)
	return gross_returns / float(len(line_sets))


def _expected_log_growth(state_probs: np.ndarray, net_returns: np.ndarray, stake_share: float) -> float:
	wealth = 1.0 + float(stake_share) * np.asarray(net_returns, dtype=float)
	if np.any(wealth <= 0.0):
		return float("-inf")
	return float(np.dot(np.asarray(state_probs, dtype=float), np.log(wealth)))


def _expected_log_growth_derivative(state_probs: np.ndarray, net_returns: np.ndarray, stake_share: float) -> float:
	wealth = 1.0 + float(stake_share) * np.asarray(net_returns, dtype=float)
	if np.any(wealth <= 0.0):
		return float("-inf")
	return float(np.dot(np.asarray(state_probs, dtype=float), np.asarray(net_returns, dtype=float) / wealth))


def _solve_full_kelly_share(state_probs: np.ndarray, net_returns: np.ndarray) -> float:
	derivative_at_zero = _expected_log_growth_derivative(state_probs, net_returns, stake_share=0.0)
	if derivative_at_zero <= 0.0:
		return 0.0

	upper = _SYSTEM_MAX_STAKE
	derivative_at_upper = _expected_log_growth_derivative(state_probs, net_returns, stake_share=upper)
	if derivative_at_upper >= 0.0:
		return float(upper)

	lower = 0.0
	for _ in range(_SYSTEM_BISECTION_MAX_ITERATIONS):
		midpoint = 0.5 * (lower + upper)
		derivative = _expected_log_growth_derivative(state_probs, net_returns, stake_share=midpoint)
		if derivative > 0.0:
			lower = midpoint
		else:
			upper = midpoint
		if upper - lower <= _SYSTEM_BISECTION_TOL:
			break
	return float(0.5 * (lower + upper))


def analyze_system_ticket(
	selected_probs: np.ndarray,
	selected_odds: np.ndarray,
	system_name: str,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Any]:
	"""Return exact Kelly sizing diagnostics for one named system ticket."""

	spec = _resolve_system_spec(system_name)
	selection_count = int(spec["selection_count"])
	local_probs = np.asarray(selected_probs, dtype=float)
	local_odds = np.asarray(selected_odds, dtype=float)
	if local_probs.shape[0] != selection_count or local_odds.shape[0] != selection_count:
		raise ValueError(
			f"{system_name} requires {selection_count} selections, got probs={local_probs.shape[0]} odds={local_odds.shape[0]}."
		)

	state_probs = _state_probabilities(local_probs)
	gross_returns = _system_gross_returns(local_odds, system_name)
	net_returns = gross_returns - 1.0
	full_kelly = _solve_full_kelly_share(state_probs, net_returns)
	scaled_share = min(_SYSTEM_MAX_STAKE, max(0.0, float(kelly_fraction)) * full_kelly)
	line_sets = _line_index_sets(selection_count, tuple(spec["line_sizes"]))
	return {
		"system_name": str(system_name),
		"selection_count": selection_count,
		"line_sizes": tuple(int(value) for value in spec["line_sizes"]),
		"line_count": int(len(line_sets)),
		"full_kelly": float(full_kelly),
		"stake_share": float(scaled_share),
		"full_log_growth": _expected_log_growth(state_probs, net_returns, stake_share=full_kelly),
		"scaled_log_growth": _expected_log_growth(state_probs, net_returns, stake_share=scaled_share),
	}


def system_gross_return(selected_odds: np.ndarray, wins: np.ndarray, system_name: str) -> float:
	"""Return the gross payout multiple for a settled named system ticket."""

	local_odds = np.asarray(selected_odds, dtype=float)
	local_wins = np.asarray(wins, dtype=bool)
	spec = _resolve_system_spec(system_name)
	selection_count = int(spec["selection_count"])
	if local_odds.shape[0] != selection_count or local_wins.shape[0] != selection_count:
		raise ValueError(
			f"{system_name} requires {selection_count} selections, got odds={local_odds.shape[0]} wins={local_wins.shape[0]}."
		)

	line_sets = _line_index_sets(selection_count, tuple(spec["line_sizes"]))
	gross_return = 0.0
	for line in line_sets:
		if local_wins[list(line)].all():
			gross_return += float(np.prod(local_odds[list(line)]))
	return float(gross_return / len(line_sets))


def select_best_system_ticket(
	selection: dict[str, np.ndarray],
	system_name: str,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
) -> dict[str, Any] | None:
	"""Search all eligible subsets and return the best exact system ticket for one slate."""

	spec = _resolve_system_spec(system_name)
	selection_count = int(spec["selection_count"])
	positive_idx = np.flatnonzero(np.asarray(selection["positive_mask"], dtype=bool))
	if positive_idx.size < selection_count:
		return None

	selected_probs = np.asarray(selection["selected_probs"], dtype=float)
	selected_odds = np.asarray(selection["selected_odds"], dtype=float)
	selected_outcomes = np.asarray(selection["best_index"], dtype=int)
	best_ticket = None
	best_key = None
	for combo in combinations(positive_idx.tolist(), selection_count):
		combo_index = np.asarray(combo, dtype=int)
		ticket = analyze_system_ticket(
			selected_probs=selected_probs[combo_index],
			selected_odds=selected_odds[combo_index],
			system_name=system_name,
			kelly_fraction=kelly_fraction,
		)
		key = (
			float(ticket["scaled_log_growth"]),
			float(ticket["stake_share"]),
			float(ticket["full_kelly"]),
		)
		if best_key is not None and key <= best_key:
			continue
		best_key = key
		best_ticket = ticket | {
			"match_indices": combo_index,
			"selected_probs": selected_probs[combo_index],
			"selected_odds": selected_odds[combo_index],
			"selected_outcomes": selected_outcomes[combo_index],
		}

	if best_ticket is None or float(best_ticket["stake_share"]) <= 0.0:
		return None
	return best_ticket


def system_bankroll_path(
	selection: dict[str, np.ndarray],
	y_true: np.ndarray,
	system_name: str,
	groups: np.ndarray | None = None,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	initial_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
	"""Return bankroll progression for a best-ticket-per-group system strategy."""

	group_labels = _normalize_groups(groups, size=len(y_true))
	starting_bankroll = max(0.0, float(initial_bankroll))
	current_bankroll = starting_bankroll
	peak_bankroll = starting_bankroll
	max_drawdown = 0.0
	ordered_groups = np.unique(group_labels)
	group_rows: list[dict[str, Any]] = []
	bankroll_after_group: list[float] = []
	ticket_count = 0
	line_count = 0

	for group in ordered_groups:
		mask = group_labels == group
		starting_group_bankroll = float(current_bankroll)
		stake_amount = 0.0
		profit = 0.0
		gross_return = 0.0
		chosen_ticket = None
		if current_bankroll > 0.0:
			group_selection = {
				key: value[mask] if isinstance(value, np.ndarray) else value
				for key, value in selection.items()
			}
			chosen_ticket = select_best_system_ticket(
				selection=group_selection,
				system_name=system_name,
				kelly_fraction=kelly_fraction,
			)
			if chosen_ticket is not None:
				stake_amount = current_bankroll * float(chosen_ticket["stake_share"])
				group_truth = np.asarray(y_true, dtype=int)[mask]
				wins = chosen_ticket["selected_outcomes"] == group_truth[chosen_ticket["match_indices"]]
				gross_return = system_gross_return(
					selected_odds=chosen_ticket["selected_odds"],
					wins=wins,
					system_name=system_name,
				)
				profit = stake_amount * (gross_return - 1.0)
				current_bankroll += profit
				ticket_count += 1
				line_count += int(chosen_ticket["line_count"])

		peak_bankroll = max(peak_bankroll, current_bankroll)
		if peak_bankroll > 0.0:
			max_drawdown = max(max_drawdown, (peak_bankroll - current_bankroll) / peak_bankroll)

		group_rows.append({
			"group": str(group),
			"starting_bankroll": starting_group_bankroll,
			"ending_bankroll": float(current_bankroll),
			"staked_amount": float(stake_amount),
			"profit": float(profit),
			"gross_return_multiple": float(gross_return),
			"ticket_placed": bool(chosen_ticket is not None),
			"line_count": int(chosen_ticket["line_count"]) if chosen_ticket is not None else 0,
			"ticket_full_kelly": float(chosen_ticket["full_kelly"]) if chosen_ticket is not None else 0.0,
			"ticket_stake_share": float(chosen_ticket["stake_share"]) if chosen_ticket is not None else 0.0,
			"ticket_scaled_log_growth": float(chosen_ticket["scaled_log_growth"]) if chosen_ticket is not None else 0.0,
			"ticket_match_indices": chosen_ticket["match_indices"].astype(int).tolist() if chosen_ticket is not None else [],
			"ticket_probs": [float(value) for value in chosen_ticket["selected_probs"]] if chosen_ticket is not None else [],
			"ticket_odds": [float(value) for value in chosen_ticket["selected_odds"]] if chosen_ticket is not None else [],
		})
		bankroll_after_group.append(float(current_bankroll))

	roi = (current_bankroll - starting_bankroll) / starting_bankroll if starting_bankroll > 0.0 else 0.0
	return {
		"system_name": str(system_name),
		"bankroll_roi": float(roi),
		"bankroll_bet_count": int(ticket_count),
		"bankroll_line_count": int(line_count),
		"max_drawdown": float(max_drawdown),
		"final_bankroll": float(current_bankroll),
		"starting_bankroll": float(starting_bankroll),
		"groups": [str(group) for group in ordered_groups],
		"bankroll_after_group": bankroll_after_group,
		"group_rows": group_rows,
	}


def evaluate_system_bankroll_strategy(
	selection: dict[str, np.ndarray],
	y_true: np.ndarray,
	system_name: str,
	groups: np.ndarray | None = None,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	initial_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, float]:
	"""Evaluate a best-ticket-per-group system strategy."""

	path = system_bankroll_path(
		selection=selection,
		y_true=y_true,
		system_name=system_name,
		groups=groups,
		kelly_fraction=kelly_fraction,
		initial_bankroll=initial_bankroll,
	)
	return {
		"bankroll_roi": float(path["bankroll_roi"]),
		"bankroll_bet_count": int(path["bankroll_bet_count"]),
		"bankroll_line_count": int(path["bankroll_line_count"]),
		"max_drawdown": float(path["max_drawdown"]),
		"final_bankroll": float(path["final_bankroll"]),
	}