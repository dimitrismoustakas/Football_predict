"""
Architecture Search for Result Prediction (Home/Draw/Away - Multiclass Classification)

Usage:
	uv run training/result_architecture_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.architecture_search_core import run_pipeline

if __name__ == "__main__":
	run_pipeline(task_type="multiclass")
