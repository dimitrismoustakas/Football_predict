import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl

from preprocessing.odds_integration import join_odds
from prod_run.build_prod_features import fetch_current_data, fill_missing_future_elo


class ProductionFeatureInputTests(unittest.TestCase):
	def test_join_odds_preserves_understat_stats_when_match_history_missing(self):
		understat = pl.DataFrame({
			"season": ["2526"],
			"home_team": ["Team A"],
			"away_team": ["Team B"],
			"home_shots": [11.0],
			"away_shots": [8.0],
			"home_sot": [5.0],
			"away_sot": [3.0],
		}).lazy()
		match_history = pl.DataFrame({
			"season": ["2526"],
			"home_team": ["Other Home"],
			"away_team": ["Other Away"],
			"home_shots": [20.0],
			"away_shots": [7.0],
			"home_sot": [9.0],
			"away_sot": [2.0],
			"odds_h": [2.1],
			"odds_d": [3.4],
			"odds_a": [3.8],
		})

		joined = join_odds(understat, match_history).collect()

		self.assertEqual(joined["home_shots"].to_list(), [11.0])
		self.assertEqual(joined["away_shots"].to_list(), [8.0])
		self.assertEqual(joined["home_sot"].to_list(), [5.0])
		self.assertEqual(joined["away_sot"].to_list(), [3.0])
		self.assertTrue(joined["odds_h"].is_null().all())

	def test_join_odds_prefers_match_history_values_when_available(self):
		understat = pl.DataFrame({
			"season": ["2526"],
			"home_team": ["Team A"],
			"away_team": ["Team B"],
			"home_shots": [11.0],
			"away_shots": [8.0],
			"home_sot": [5.0],
			"away_sot": [3.0],
		}).lazy()
		match_history = pl.DataFrame({
			"season": ["2526"],
			"home_team": ["Team A"],
			"away_team": ["Team B"],
			"home_shots": [14.0],
			"away_shots": [10.0],
			"home_sot": [6.0],
			"away_sot": [4.0],
			"odds_h": [2.2],
			"odds_d": [3.5],
			"odds_a": [3.6],
		})

		joined = join_odds(understat, match_history).collect()

		self.assertEqual(joined["home_shots"].to_list(), [14.0])
		self.assertEqual(joined["away_shots"].to_list(), [10.0])
		self.assertEqual(joined["home_sot"].to_list(), [6.0])
		self.assertEqual(joined["away_sot"].to_list(), [4.0])
		self.assertEqual(joined["odds_h"].to_list(), [2.2])
		self.assertEqual(joined["odds_d"].to_list(), [3.5])
		self.assertEqual(joined["odds_a"].to_list(), [3.6])

	def test_fill_missing_future_elo_only_updates_future_rows(self):
		matches = pl.DataFrame({
			"date": [datetime(2026, 3, 10), datetime(2026, 3, 20)],
			"home_team": ["Team A", "Team A"],
			"away_team": ["Team B", "Team B"],
			"home_elo": [None, None],
			"away_elo": [None, None],
			"elo_diff": [None, None],
			"elo_sum": [None, None],
			"elo_mean": [None, None],
		})
		elo_mapped = pl.DataFrame({
			"team_canonical": ["Team A", "Team B"],
			"elo": [1510.0, 1430.0],
		})

		filled = fill_missing_future_elo(
			matches,
			elo_mapped,
			reference_time=datetime(2026, 3, 15),
		)

		self.assertIsNone(filled["home_elo"][0])
		self.assertIsNone(filled["away_elo"][0])
		self.assertEqual(filled["home_elo"][1], 1510.0)
		self.assertEqual(filled["away_elo"][1], 1430.0)
		self.assertEqual(filled["elo_diff"][1], 80.0)
		self.assertEqual(filled["elo_sum"][1], 2940.0)
		self.assertEqual(filled["elo_mean"][1], 1470.0)

	def test_fetch_current_data_raises_on_league_fetch_error(self):
		def fake_understat(leagues, seasons):
			reader = Mock()
			if leagues == "ESP-La Liga":
				reader.read_team_match_stats.side_effect = RuntimeError("upstream failure")
			else:
				reader.read_team_match_stats.return_value = pd.DataFrame({
					"league": [leagues],
					"league_id": [leagues],
					"season": ["2526"],
					"date": [pd.Timestamp("2026-03-10")],
					"game": [f"{leagues}-match"],
					"home_team": ["Team A"],
					"away_team": ["Team B"],
					"home_team_id": [1],
					"away_team_id": [2],
					"home_shot": [11.0],
					"away_shot": [8.0],
					"home_shotontarget": [5.0],
					"away_shotontarget": [3.0],
				})
			return reader

		with patch("prod_run.build_prod_features.sd.Understat", side_effect=fake_understat):
			with self.assertRaisesRegex(RuntimeError, "upstream failure"):
				fetch_current_data(upcoming_fixtures=pd.DataFrame())


if __name__ == "__main__":
	unittest.main()