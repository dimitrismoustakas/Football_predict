"""
Portfolio and betting allocation utilities.
"""

from typing import Dict
import numpy as np
import pandas as pd


def calculate_betting_allocations(
	probs: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	home_teams: list,
	away_teams: list,
	dates: list,
	budget: float = 100.0,
) -> pd.DataFrame:
	"""
	Calculate betting allocations using Sharpe-weighted portfolio strategy.
	
	Args:
		probs: Predicted probabilities for over 2.5 goals
		odds_over: Bookmaker odds for over 2.5 goals
		odds_under: Bookmaker odds for under 2.5 goals
		home_teams: List of home team names
		away_teams: List of away team names
		dates: List of match dates
		budget: Total budget for allocation (percentage)
	
	Returns:
		DataFrame with betting allocations and metrics
	"""
	df = pd.DataFrame({
		"home_team": home_teams,
		"away_team": away_teams,
		"date": dates,
		"prob_over": probs,
		"odds_over": odds_over,
		"odds_under": odds_under,
	})
	
	p = df["prob_over"]
	o_over = df["odds_over"]
	o_under = df["odds_under"]
	
	# Expected value calculations
	# For Over bet: E[profit] = p * (odds - 1) + (1-p) * (-1) = p * odds - 1
	mu_over = p * o_over - 1
	# For Under bet: E[profit] = (1-p) * (odds - 1) + p * (-1) = (1-p) * odds - 1
	mu_under = (1 - p) * o_under - 1
	
	# Variance calculations
	# Var(profit) = E[profit^2] - E[profit]^2
	# For Over: outcomes are (odds-1) with prob p, or -1 with prob (1-p)
	e_x2_over = p * (o_over - 1) ** 2 + (1 - p) * 1
	var_over = e_x2_over - mu_over ** 2
	
	e_x2_under = (1 - p) * (o_under - 1) ** 2 + p * 1
	var_under = e_x2_under - mu_under ** 2
	
	# Determine better side for each game
	better_is_over = mu_over >= mu_under
	mu_best = np.where(better_is_over, mu_over, mu_under)
	var_best = np.where(better_is_over, var_over, var_under)
	odds_best = np.where(better_is_over, o_over, o_under)
	
	df["bet_side"] = np.where(better_is_over, "Over", "Under")
	df["mu"] = mu_best
	df["var"] = var_best
	df["odds_selected"] = odds_best
	df["eligible"] = df["mu"] > 0  # Only bet on positive EV
	
	# Calculate Sharpe-weighted allocations
	eligible_df = df[df["eligible"]].copy()
	
	if len(eligible_df) > 0:
		# Sharpe weight = mu / var (capped variance at small value for stability)
		vars_ = eligible_df["var"].values + 1e-6
		mus = eligible_df["mu"].values
		raw_weights = np.maximum(0, mus / vars_)
		sum_weights = raw_weights.sum()
		
		if sum_weights > 0:
			norm_weights = raw_weights / sum_weights
			eligible_df["allocation_pct"] = (norm_weights * 100).round(2)
		else:
			eligible_df["allocation_pct"] = 0.0
	
	# Merge back
	df = df.merge(
		eligible_df[["home_team", "away_team", "allocation_pct"]],
		on=["home_team", "away_team"],
		how="left"
	)
	df["allocation_pct"] = df["allocation_pct"].fillna(0.0)
	
	return df


def evaluate_portfolio(
	probs: np.ndarray,
	y_true: np.ndarray,
	odds_over: np.ndarray,
	odds_under: np.ndarray,
	dates: np.ndarray,
	budget_per_day: float = 10.0,
) -> Dict:
	"""
	Portfolio-style evaluation with Sharpe-weighted betting.
	
	Args:
		probs: Predicted probabilities for over 2.5 goals
		y_true: Actual outcomes (1 for over, 0 for under)
		odds_over: Bookmaker odds for over 2.5 goals
		odds_under: Bookmaker odds for under 2.5 goals
		dates: Match dates for grouping
		budget_per_day: Daily budget allocation
	
	Returns:
		Dictionary with portfolio performance metrics
	"""
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
