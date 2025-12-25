"""
Shared utilities for Football Prediction Pipeline.
"""

from utils.portfolio import calculate_betting_allocations, evaluate_portfolio
from utils.email_utils import send_email

__all__ = [
	"calculate_betting_allocations",
	"evaluate_portfolio",
	"send_email",
]
