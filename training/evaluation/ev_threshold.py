"""Helpers for testing raw-EV thresholded Kelly strategies."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from utils.portfolio import (
	DEFAULT_BANKROLL,
	DEFAULT_KELLY_FRACTION,
	allocate_bankroll_kelly,
)


@dataclass
class FittedEvThreshold:
	"""Monotone fit from predicted raw EV to realized ROI."""

	threshold: float
	model: IsotonicRegression
	support_raw_ev: np.ndarray
	support_realized_roi: np.ndarray


def selection_to_bet_records(selection: dict[str, np.ndarray], y_true: np.ndarray) -> dict[str, np.ndarray]:
	"""Extract positive-EV selected-side records using actual decimal odds."""

	positive_mask = np.asarray(selection["positive_mask"], dtype=bool)
	positive_idx = np.flatnonzero(positive_mask)
	if positive_idx.size == 0:
		return {
			"raw_ev": np.zeros(0, dtype=float),
			"selected_odds": np.zeros(0, dtype=float),
			"selected_probs": np.zeros(0, dtype=float),
			"realized_roi": np.zeros(0, dtype=float),
			"wins": np.zeros(0, dtype=bool),
		}

	selected_odds = np.asarray(selection["selected_odds"], dtype=float)[positive_idx]
	selected_probs = np.asarray(selection["selected_probs"], dtype=float)[positive_idx]
	raw_ev = np.asarray(selection["best_ev"], dtype=float)[positive_idx]
	selected_outcomes = np.asarray(selection["best_index"], dtype=int)[positive_idx]
	truth = np.asarray(y_true, dtype=int)[positive_idx]
	wins = selected_outcomes == truth
	realized_roi = np.where(wins, selected_odds - 1.0, -1.0)
	return {
		"raw_ev": raw_ev,
		"selected_odds": selected_odds,
		"selected_probs": selected_probs,
		"realized_roi": realized_roi.astype(float),
		"wins": wins,
	}


def fit_monotone_ev_threshold(raw_ev: np.ndarray, realized_roi: np.ndarray) -> FittedEvThreshold:
	"""Fit a monotone ROI curve and return the first raw-EV level with positive fitted ROI."""

	raw_ev = np.asarray(raw_ev, dtype=float)
	realized_roi = np.asarray(realized_roi, dtype=float)
	if raw_ev.ndim != 1 or realized_roi.ndim != 1:
		raise ValueError("raw_ev and realized_roi must be 1D arrays.")
	if raw_ev.shape[0] != realized_roi.shape[0]:
		raise ValueError("raw_ev and realized_roi must have matching lengths.")
	if raw_ev.size == 0:
		raise ValueError("Need at least one positive-EV record to fit an EV threshold.")

	model = IsotonicRegression(increasing=True, out_of_bounds="clip")
	model.fit(raw_ev, realized_roi)
	support_raw_ev = np.unique(np.sort(raw_ev))
	support_realized_roi = np.asarray(model.predict(support_raw_ev), dtype=float)
	positive_idx = np.flatnonzero(support_realized_roi > 0.0)
	threshold = float(support_raw_ev[positive_idx[0]]) if positive_idx.size else float("inf")
	return FittedEvThreshold(
		threshold=threshold,
		model=model,
		support_raw_ev=support_raw_ev,
		support_realized_roi=support_realized_roi,
	)


def apply_ev_threshold(selection: dict[str, np.ndarray], ev_threshold: float) -> dict[str, np.ndarray]:
	"""Keep only selected bets whose raw EV clears the supplied threshold."""

	updated = {
		key: value.copy() if isinstance(value, np.ndarray) else value
		for key, value in selection.items()
	}
	if not np.isfinite(ev_threshold):
		keep_mask = np.zeros_like(np.asarray(selection["positive_mask"], dtype=bool))
	else:
		keep_mask = np.asarray(selection["positive_mask"], dtype=bool) & (
			np.asarray(selection["best_ev"], dtype=float) >= float(ev_threshold)
		)
	updated["positive_mask"] = keep_mask
	updated["full_kelly"] = np.where(
		keep_mask,
		np.asarray(selection["full_kelly"], dtype=float),
		0.0,
	)
	return updated


def summarize_raw_ev_bins(
	raw_ev: np.ndarray,
	realized_roi: np.ndarray,
	bin_edges: np.ndarray,
) -> list[dict[str, float | int | str]]:
	"""Summarize realized ROI by raw-EV bucket."""

	raw_ev = np.asarray(raw_ev, dtype=float)
	realized_roi = np.asarray(realized_roi, dtype=float)
	bin_edges = np.asarray(bin_edges, dtype=float)
	rows: list[dict[str, float | int | str]] = []
	for idx in range(len(bin_edges) - 1):
		left = float(bin_edges[idx])
		right = float(bin_edges[idx + 1])
		if idx == len(bin_edges) - 2:
			mask = (raw_ev >= left) & (raw_ev <= right)
		else:
			mask = (raw_ev >= left) & (raw_ev < right)
		count = int(mask.sum())
		rows.append({
			"label": f"[{left:.3f}, {right:.3f}{']' if idx == len(bin_edges) - 2 else ')'}",
			"left": left,
			"right": right,
			"count": count,
			"mean_raw_ev": float(raw_ev[mask].mean()) if count else 0.0,
			"mean_realized_roi": float(realized_roi[mask].mean()) if count else 0.0,
		})
	return rows


def _coerce_group_date(value: Any) -> date:
	"""Convert a date-like value from the evaluation frames into a Python date."""

	if isinstance(value, date) and not isinstance(value, datetime):
		return value
	if isinstance(value, datetime):
		return value.date()
	if isinstance(value, np.datetime64):
		return date.fromisoformat(np.datetime_as_string(value, unit="D"))
	if hasattr(value, "date"):
		candidate = value.date()
		if isinstance(candidate, date):
			return candidate
	text = str(value).strip()
	if "T" in text:
		text = text.split("T", 1)[0]
	if " " in text:
		text = text.split(" ", 1)[0]
	return date.fromisoformat(text[:10])


def build_group_labels(groups: np.ndarray | None, mode: str = "day") -> np.ndarray:
	"""Return stable bankroll group labels for a chosen grouping mode."""

	if mode == "day":
		size = 0 if groups is None else len(np.asarray(groups))
		return _normalize_groups(groups, size=size)

	if groups is None:
		return np.zeros(0, dtype=object)

	array = np.asarray(groups)
	labels: list[str] = []
	for value in array:
		current_date = _coerce_group_date(value)
		weekday = current_date.weekday()
		if weekday in {1, 2, 3}:
			anchor = current_date - timedelta(days=weekday - 1)
			window = "tue-thu"
		else:
			offset_to_friday = {4: 0, 5: 1, 6: 2, 0: 3}[weekday]
			anchor = current_date - timedelta(days=offset_to_friday)
			window = "fri-mon"
		labels.append(f"{anchor.isoformat()}_{window}")
	return np.asarray(labels, dtype=object)


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


def evaluate_selection_bankroll_strategy(
	selection: dict[str, np.ndarray],
	y_true: np.ndarray,
	groups: np.ndarray | None = None,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	initial_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, float]:
	"""Evaluate a preselected-side Kelly strategy over grouped fixtures."""

	path = selection_bankroll_path(
		selection=selection,
		y_true=y_true,
		groups=groups,
		kelly_fraction=kelly_fraction,
		initial_bankroll=initial_bankroll,
	)
	return {
		"bankroll_roi": float(path["bankroll_roi"]),
		"bankroll_bet_count": int(path["bankroll_bet_count"]),
		"max_drawdown": float(path["max_drawdown"]),
		"final_bankroll": float(path["final_bankroll"]),
	}


def selection_bankroll_path(
	selection: dict[str, np.ndarray],
	y_true: np.ndarray,
	groups: np.ndarray | None = None,
	kelly_fraction: float = DEFAULT_KELLY_FRACTION,
	initial_bankroll: float = DEFAULT_BANKROLL,
) -> dict[str, Any]:
	"""Return bankroll progression for a preselected-side Kelly strategy."""

	group_labels = _normalize_groups(groups, size=len(y_true))
	stake_amounts = np.zeros(len(y_true), dtype=float)
	starting_bankroll = max(0.0, float(initial_bankroll))
	current_bankroll = starting_bankroll
	peak_bankroll = starting_bankroll
	max_drawdown = 0.0
	ordered_groups = np.unique(group_labels)
	group_rows: list[dict[str, Any]] = []
	bankroll_after_group: list[float] = []

	for group in ordered_groups:
		mask = group_labels == group
		starting_group_bankroll = float(current_bankroll)
		if current_bankroll > 0.0:
			allocation = allocate_bankroll_kelly(
				selection={
					key: value[mask] if isinstance(value, np.ndarray) else value
					for key, value in selection.items()
				},
				total_bankroll=current_bankroll,
				kelly_fraction=kelly_fraction,
			)
			group_stakes = np.asarray(allocation["stake_amounts"], dtype=float)
			stake_amounts[mask] = group_stakes
			group_outcomes = np.asarray(selection["best_index"], dtype=int)[mask]
			group_odds = np.asarray(selection["selected_odds"], dtype=float)[mask]
			group_truth = np.asarray(y_true, dtype=int)[mask]
			group_profit = np.where(group_outcomes == group_truth, group_stakes * (group_odds - 1.0), -group_stakes)
			current_bankroll += float(group_profit.sum())
		else:
			group_stakes = np.zeros(int(mask.sum()), dtype=float)
			group_profit = np.zeros(int(mask.sum()), dtype=float)
		peak_bankroll = max(peak_bankroll, current_bankroll)
		if peak_bankroll > 0.0:
			max_drawdown = max(max_drawdown, (peak_bankroll - current_bankroll) / peak_bankroll)
		group_rows.append({
			"group": str(group),
			"starting_bankroll": starting_group_bankroll,
			"ending_bankroll": float(current_bankroll),
			"staked_amount": float(group_stakes.sum()),
			"profit": float(group_profit.sum()),
			"bet_count": int(np.count_nonzero(group_stakes > 0.0)),
		})
		bankroll_after_group.append(float(current_bankroll))

	n_staked_bets = int(np.count_nonzero(stake_amounts > 0.0))
	roi = (current_bankroll - starting_bankroll) / starting_bankroll if starting_bankroll > 0.0 else 0.0
	return {
		"bankroll_roi": float(roi),
		"bankroll_bet_count": n_staked_bets,
		"max_drawdown": float(max_drawdown),
		"final_bankroll": float(current_bankroll),
		"starting_bankroll": float(starting_bankroll),
		"groups": [str(group) for group in ordered_groups],
		"bankroll_after_group": bankroll_after_group,
		"group_rows": group_rows,
	}


def subtract_metric_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
	"""Return left-minus-right for numeric metric dictionaries."""

	keys = sorted(set(left) & set(right))
	return {
		key: float(left[key]) - float(right[key])
		for key in keys
		if isinstance(left[key], (int, float)) and isinstance(right[key], (int, float))
	}
