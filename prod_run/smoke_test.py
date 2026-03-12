"""Offline production smoke test using synthetic odds and a deterministic fake model."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import torch

if __package__ is None or __package__ == "":
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prod_run.generate_html_report import generate_html_report
from prod_run.pipeline import build_prediction_outputs, score_result_predictions
from training.train_utils import load_feature_manifest

PREDICTIONS_DIR = Path("data/predictions")
SMOKE_CSV_PATH = PREDICTIONS_DIR / "smoke_upcoming_predictions.csv"
SMOKE_HTML_PATH = PREDICTIONS_DIR / "smoke_upcoming_predictions.html"
PROD_FEATURES_PATH = Path("data/prod/features_season.parquet")
PREDICTION_WINDOW_DAYS = 5
FIXED_BUDGET = 100.0


class IdentityScaler:
	def transform(self, X):
		return X


class FakeModel:
	def __call__(self, X, cat_features, implied, raw_margin):
		bias = torch.tensor([0.22, -0.08, -0.14], dtype=torch.float32, device=implied.device)
		return torch.log(implied.clamp_min(1e-6)) + bias


def _synthetic_market_probs(elo_diff: np.ndarray) -> np.ndarray:
	shift = np.tanh(np.nan_to_num(elo_diff, nan=0.0) / 350.0)
	home = 0.43 + 0.16 * shift
	draw = 0.27 - 0.05 * np.abs(shift)
	away = 0.30 - 0.16 * shift
	probs = np.column_stack([home, draw, away])
	return probs / probs.sum(axis=1, keepdims=True)


def _build_synthetic_odds(features_df: pd.DataFrame) -> pd.DataFrame:
	probs = _synthetic_market_probs(features_df["elo_diff"].to_numpy(dtype=float))
	margin = 1.05
	odds = 1.0 / (probs * margin)
	return pd.DataFrame({
		"league": features_df["league"],
		"home_team": features_df["home_team"],
		"away_team": features_df["away_team"],
		"commence_time": pd.to_datetime(features_df["date"], utc=True),
		"odds_home": np.round(odds[:, 0], 2),
		"odds_draw": np.round(odds[:, 1], 2),
		"odds_away": np.round(odds[:, 2], 2),
	})


def _load_upcoming_features() -> pd.DataFrame:
	features_df = pl.read_parquet(PROD_FEATURES_PATH).to_pandas()
	features_df["date"] = pd.to_datetime(features_df["date"], utc=True, errors="coerce")
	now_utc = datetime.now(timezone.utc)
	today_utc = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
	window_end_utc = today_utc + pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
	upcoming = features_df[(features_df["date"] >= today_utc) & (features_df["date"] < window_end_utc)].copy()
	if upcoming.empty:
		raise RuntimeError("No upcoming fixtures found in data/prod/features_season.parquet for the smoke-test window.")
	return upcoming


def main() -> None:
	feature_cols = load_feature_manifest()
	upcoming_features = _load_upcoming_features()
	synthetic_odds = _build_synthetic_odds(upcoming_features)
	merged = synthetic_odds.merge(
		upcoming_features,
		on=["league", "home_team", "away_team"],
		how="inner",
		suffixes=("_odds", "_feat"),
	)
	merged = merged.reset_index(drop=True)
	merged["_row_id"] = merged.index

	bundle = SimpleNamespace(
		model=FakeModel(),
		scaler=IdentityScaler(),
		feature_cols=feature_cols,
		cat_config=None,
	)
	result_predictions = score_result_predictions(
		merged,
		bundle,
		fixed_budget=FIXED_BUDGET,
		kelly_fraction=0.5,
	)
	output_df, value_df = build_prediction_outputs(merged, result_predictions)

	PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
	output_df.to_csv(SMOKE_CSV_PATH, index=False)
	generate_html_report(
		output_df,
		SMOKE_HTML_PATH,
		fixed_budget=FIXED_BUDGET,
		kelly_fraction=0.5,
	)

	print(f"Wrote smoke-test CSV: {SMOKE_CSV_PATH}")
	print(f"Wrote smoke-test HTML: {SMOKE_HTML_PATH}")
	print(f"Fixtures scored: {len(output_df)}")
	print(f"Positive EV picks: {len(value_df)}")
	print(f"Allocated bankroll: {value_df['Result_Budget_Amount'].sum():.2f}")


if __name__ == "__main__":
	main()
