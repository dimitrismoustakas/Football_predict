"""Shared utilities for Football Prediction Pipeline."""

from utils.email_utils import send_email
from utils.portfolio import allocate_bankroll_kelly, evaluate_bankroll_strategy, select_best_result_value

__all__ = ["allocate_bankroll_kelly", "evaluate_bankroll_strategy", "select_best_result_value", "send_email"]
