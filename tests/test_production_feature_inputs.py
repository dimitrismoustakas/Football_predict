import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from preprocessing.match_feature_pipeline import (
	get_player_feature_columns,
	join_player_features_asof,
	join_player_features_by_game_id,
)
from preprocessing.odds_integration import join_odds
from preprocessing.player_feature_engineering import build_player_team_features
from prod_run.build_prod_features import fetch_current_data, fill_missing_future_elo, load_production_player_features


def player_match_history(n_matches: int) -> tuple[pl.DataFrame, pl.DataFrame]:
	matches = []
	players = []
	for game in range(n_matches):
		date = datetime(2025, 8, 1, 17) + timedelta(days=7 * game)
		match = {"league": "ENG-Premier League", "season": "2526", "game_id": str(game), "date": date}
		matches.append({**match, "home_team_id": 1, "away_team_id": 2})
		for team_id in (1, 2):
			for player in range(11 + (2 * game + team_id - 1) % 7):
				players.append({
					**match,
					"match_id": f"{date:%Y-%m-%d} Team A-Team B",
					"team_id": team_id,
					"team": f"Team {team_id}",
					"player_id": player,
					"minutes": 90 - player,
					"goals": int(player == 0),
					"xg": (player + 1) * (game + 1) / 100,
					"xa": (player + 1) / (game + 1),
					"assists": int(player == 1),
					"shots": player % 3,
					"key_passes": player % 2,
				})
	return pl.DataFrame(players), pl.DataFrame(matches)


class ProductionFeatureInputTests(unittest.TestCase):
	def test_production_player_features_include_latest_completed_match(self):
		players, matches = player_match_history(4)
		completed = players.filter(pl.col("game_id") != "3")
		prediction_time = matches["date"][2] + timedelta(hours=3)
		with (
			patch("prod_run.build_prod_features.load_all_player_data", return_value=completed),
			patch("prod_run.build_prod_features.fetch_current_player_data", return_value=completed),
			patch("prod_run.build_prod_features.get_current_season_key", return_value="2526"),
			patch("prod_run.build_prod_features.datetime") as clock,
		):
			clock.utcnow.return_value = prediction_time
			production_features = load_production_player_features()

		actual = join_player_features_asof(matches, production_features).sort("date")
		expected = join_player_features_by_game_id(matches, build_player_team_features(players)).sort("date")
		assert_frame_equal(actual, expected)
		# The completed match keeps its pre-match average; the next match uses all 11, 13, 15 players.
		self.assertEqual(actual["home_unique_players_r15"].to_list(), [None, None, 12.0, 13.0])

	def test_upcoming_player_features_preserve_history_and_rolling_windows(self):
		players, matches = player_match_history(19)
		reference = build_player_team_features(players)
		feature_columns = [
			f"{side}_{column}"
			for side in ("home", "away")
			for column in get_player_feature_columns(reference)
		]
		for n_completed in (1, 2, 17):
			with self.subTest(n_completed=n_completed):
				completed = players.filter(pl.col("game_id").cast(pl.Int64) < n_completed)
				production_features = build_player_team_features(
					completed, prediction_time=matches["date"][n_completed - 1] + timedelta(hours=3),
				)
				assert_frame_equal(
					production_features.filter(pl.col("game_id").is_not_null()),
					build_player_team_features(completed),
				)
				upcoming = matches.slice(n_completed)
				actual = join_player_features_asof(upcoming, production_features).select(feature_columns)
				expected = join_player_features_by_game_id(upcoming.head(1), reference).select(feature_columns)
				# All upcoming fixtures reuse the same completed history, including the capped windows.
				assert_frame_equal(actual, pl.concat([expected] * upcoming.height))

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
