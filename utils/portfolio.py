"""
Portfolio and betting allocation utilities.
"""

from typing import Dict

import numpy as np
import pandas as pd


def _build_binary_bet_frame(
	probs: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray | list,
) -> pd.DataFrame:
	df = pd.DataFrame({
		"date": pd.to_datetime(dates, utc=True, errors="coerce"),
		"prob_over": probs,
		"odds_over": odds_over,
		"odds_under": odds_under,
	})

	mu_over = df["prob_over"] * df["odds_over"] - 1
	mu_under = (1 - df["prob_over"]) * df["odds_under"] - 1
	better_is_over = mu_over >= mu_under

	df["bet_side"] = np.where(better_is_over, "Over", "Under")
	df["mu"] = np.where(better_is_over, mu_over, mu_under)
	df["selected_odds"] = np.where(better_is_over, df["odds_over"], df["odds_under"])
	df["bet_date"] = df["date"].dt.strftime("%Y-%m-%d")

	return df


def calculate_betting_allocations(
	probs: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	home_teams: list,
	away_teams: list,
	dates: list,
	budget: float = 100.0,
	min_edge: float = 0.0,
) -> pd.DataFrame:
	"""
	Allocate each day's fixed budget equally across all positive-EV bets for that day.

	`allocation_pct` is the share of that day's budget, not a share of the full multi-day window.
	"""
	df = _build_binary_bet_frame(probs, odds_over, odds_under, dates)
	df["home_team"] = home_teams
	df["away_team"] = away_teams
	df["eligible"] = df["mu"] > min_edge
	df["allocation_pct"] = 0.0
	df["daily_budget"] = float(budget)

	for bet_date, group in df.groupby("bet_date", dropna=False):
		eligible_idx = group.index[group["eligible"]].tolist()
		if not eligible_idx:
			continue
		allocation_pct = round(100.0 / len(eligible_idx), 2)
		df.loc[eligible_idx, "allocation_pct"] = allocation_pct

	return df


def evaluate_daily_betting_results(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
	min_edge: float = 0.0,
) -> Dict:
	"""
	Evaluate betting performance with a fixed budget per calendar day.

	For each day, the budget is split equally across all positive-EV bets.
	No minimum number of games per day is required.
	"""
	df = _build_binary_bet_frame(probs, odds_over, odds_under, dates)
	df["y_true"] = y_true
	df["eligible"] = df["mu"] > min_edge
	df["won"] = np.where(df["bet_side"] == "Over", df["y_true"] == 1, df["y_true"] == 0)

	calendar_days = [day for day in df["bet_date"].dropna().unique().tolist()]
	calendar_days.sort()
	daily_profits = []
	betting_day_profits = []
	total_bets = 0

	for bet_date in calendar_days:
		group = df[df["bet_date"] == bet_date]
		eligible = group[group["eligible"]]
		if eligible.empty:
			daily_profits.append(0.0)
			continue

		stake = budget_per_day / len(eligible)
		profits = np.where(
			eligible["won"].to_numpy(),
			stake * (eligible["selected_odds"].to_numpy() - 1),
			-stake,
		)
		daily_profit = float(np.sum(profits))
		daily_profits.append(daily_profit)
		betting_day_profits.append(daily_profit)
		total_bets += int(len(eligible))

	daily_profits_np = np.array(daily_profits, dtype=float)
	betting_day_profits_np = np.array(betting_day_profits, dtype=float)
	invested_capital = budget_per_day * len(betting_day_profits)

	return {
		"daily_total_profit": float(daily_profits_np.sum()),
		"avg_profit_per_calendar_day": float(daily_profits_np.mean()) if len(daily_profits_np) > 0 else 0.0,
		"avg_profit_per_betting_day": float(betting_day_profits_np.mean()) if len(betting_day_profits_np) > 0 else 0.0,
		"median_profit_per_betting_day": float(np.median(betting_day_profits_np)) if len(betting_day_profits_np) > 0 else 0.0,
		"daily_profit_std": float(daily_profits_np.std()) if len(daily_profits_np) > 0 else 0.0,
		"daily_roi": float(daily_profits_np.sum() / invested_capital) if invested_capital > 0 else 0.0,
		"n_calendar_days": int(len(calendar_days)),
		"n_betting_days": int(len(betting_day_profits)),
		"n_bets": int(total_bets),
	}


def evaluate_portfolio(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
	min_edge: float = 0.0,
) -> Dict:
	"""Backward-compatible alias for daily fixed-budget betting evaluation."""
	return evaluate_daily_betting_results(
		probs=probs,
		y_true=y_true,
		odds_over=odds_over,
		odds_under=odds_under,
		dates=dates,
		budget_per_day=budget_per_day,
		min_edge=min_edge,
	)
