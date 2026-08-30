import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd
import polars as pl

from data_collection.collect_elo import collect_elo, fetch_current_elo, fetch_team_history
from preprocessing.elo_integration import merge_elo_features
from prod_run.build_prod_features import fill_missing_future_elo, map_current_elo


def club_page(rows, links=()):
	chart = {"datasets": {"other": [{"x": 1}], "ratings": rows}}
	# The chart is JSON, but the site's table is a single-quoted JavaScript array.
	table = [[f'<a href="/{slug}">{name}<span class="min481"></span></a>', "1600"] for slug, name in links]
	return f"<script>var vegaJson = {json.dumps(chart)}; const eloData = {table!r};</script>"


class EloCollectionTests(unittest.TestCase):
	def setUp(self):
		directory = TemporaryDirectory()
		self.addCleanup(directory.cleanup)
		self.path = Path(directory.name) / "elo.parquet"
		self.history = pd.DataFrame({
			"team": ["Arsenal", "Chelsea"],
			"country": ["ENG", "ENG"],
			"from": pd.to_datetime(["2026-05-01", "2026-05-01"]),
			"to": pd.to_datetime(["2026-05-05", "2026-05-05"]),
			"elo": [1510.125, 1430.75],
		})
		self.history.to_parquet(self.path, index=False)
		clock = patch("data_collection.collect_elo.datetime")
		clock.start().now.return_value = datetime(2026, 8, 31, 12, tzinfo=timezone.utc)
		self.addCleanup(clock.stop)

	def matches(self, *dates):
		return pl.DataFrame({
			"game_id": list(range(len(dates))),
			"league": ["ENG-Premier League"] * len(dates),
			"date": [datetime.fromisoformat(date) for date in dates],
			"home_team": ["Arsenal"] * len(dates),
			"away_team": ["Chelsea"] * len(dates),
		})

	def test_missing_dates_preserve_history_and_live_ratings_stay_out_of_cache(self):
		matches = self.matches(
			"2026-05-03T15:00", "2026-08-30T15:00", "2026-08-31T20:00",
			"2026-09-01T15:00", "2026-09-03T15:00",
		)
		pages = {
			"Arsenal": club_page([
				{"Date": "2026-05-01T00:00:00", "Elo": 9999.0},
				{"Date": "2026-08-28T00:00:00", "Elo": 1590.125},
				{"Date": "2026-08-30T00:00:00", "Elo": 1600.875},
				{"Date": "2026-08-31T00:00:00", "Elo": 1700.0},
			]),
			"Chelsea": club_page([
				{"Date": "2026-08-30T00:00:00", "Elo": 1490.125},
				{"Date": "2026-08-28T00:00:00", "Elo": 1480.25},
			]),
			"ENG": club_page([
				{"Name": "Arsenal", "Elo": 1710.123456789, "Level": 1},
				{"Name": "Chelsea", "Elo": 1500.987654321, "Level": 1},
			], [("Arsenal", "Arsenal"), ("Chelsea", "Chelsea")]),
		}
		with patch("data_collection.collect_elo._read_page", side_effect=pages.__getitem__) as read:
			current = collect_elo(matches, self.path)
			self.assertEqual([call.args[0] for call in read.call_args_list], ["Arsenal", "Chelsea", "ENG"])

		saved = pd.read_parquet(self.path)
		original = saved[saved["from"] == pd.Timestamp("2026-05-01")].reset_index(drop=True)
		pd.testing.assert_frame_equal(original, self.history)
		added = saved[saved["from"] > pd.Timestamp("2026-05-01")]
		self.assertEqual(len(added), 4)
		self.assertEqual(set(added["from"]), {pd.Timestamp("2026-08-29"), pd.Timestamp("2026-08-30")})
		self.assertTrue((added["from"] == added["to"]).all())

		joined = merge_elo_features(matches, self.path)
		filled = fill_missing_future_elo(joined, map_current_elo(current), datetime(2026, 8, 31, 12)).sort("game_id")
		self.assertEqual(filled["home_elo"].to_list(), [1510.125, 1590.125, 1600.875, 1710.123456789, 1710.123456789])
		self.assertEqual(filled["away_elo"].to_list(), [1430.75, 1480.25, 1490.125, 1500.987654321, 1500.987654321])

		before = self.path.read_bytes()
		with patch("data_collection.collect_elo._read_page", side_effect=AssertionError("unexpected request")):
			current = collect_elo(matches.head(3), self.path)
		self.assertTrue(current.is_empty())
		self.assertEqual(self.path.read_bytes(), before)

	def test_current_ratings_use_link_identity_and_exact_chart_values(self):
		pages = {
			"GER": club_page([
				{"Name": "Köln", "Elo": 1555.123456789, "Level": 1},
				{"Name": "Bayern München", "Elo": 2000.25, "Level": 1},
			], [("Koeln", "K&ouml;ln"), ("Bayern", "Bayern München")]),
			"ITA": club_page([
				{"Name": "Venezia", "Elo": 1646.007492458501, "Level": 1},
			], [("venezia-fc", "Venezia")]),
			"ENG": club_page([
				{"Name": "Brentford", "Elo": 1701.7654321, "Level": 2},
			], [("Brentford", "Brentford")]),
		}
		teams = pd.DataFrame({"team": ["Koeln", "Venezia", "Brentford"], "country": ["GER", "ITA", "ENG"]})
		with patch("data_collection.collect_elo._read_page", side_effect=pages.__getitem__):
			current = fetch_current_elo(teams)
		self.assertEqual(dict(current.iter_rows()), {
			"Koeln": 1555.123456789, "Venezia": 1646.007492458501, "Brentford": 1701.7654321,
		})

	def test_failed_live_fetch_does_not_save_partial_history(self):
		matches = self.matches("2026-08-30T15:00", "2026-09-01T15:00")
		past_page = club_page([{"Date": "2026-08-28T00:00:00", "Elo": 1600.0}])
		before = self.path.read_bytes()
		with patch("data_collection.collect_elo._read_page", side_effect=[past_page, past_page, OSError("upstream failure")]):
			with self.assertRaisesRegex(OSError, "upstream failure"):
				collect_elo(matches, self.path)
		self.assertEqual(self.path.read_bytes(), before)

	def test_history_window_does_not_fill_backwards(self):
		matches = self.matches("2022-05-01T15:00")
		page = club_page([{"Date": "2022-09-01T00:00:00", "Elo": 1600.0}])
		before = self.path.read_bytes()
		with patch("data_collection.collect_elo._read_page", return_value=page):
			with self.assertRaisesRegex(ValueError, "does not cover Arsenal: 2022-04-30"):
				collect_elo(matches, self.path)
		self.assertEqual(self.path.read_bytes(), before)

	def test_missing_current_rating_is_not_replaced_by_a_historical_value(self):
		page = club_page([{"Name": "Arsenal", "Elo": 1800.0}], [("Arsenal", "Arsenal")])
		with patch("data_collection.collect_elo._read_page", return_value=page):
			with self.assertRaisesRegex(ValueError, "no full-precision current rating for Hull"):
				fetch_current_elo(pd.DataFrame({"team": ["Hull"], "country": ["ENG"]}))

	def test_missing_club_redirect_cannot_be_parsed_as_club_history(self):
		response = Mock()
		response.geturl.return_value = "https://clubelo.com/"
		response.__enter__ = Mock(return_value=response)
		response.__exit__ = Mock(return_value=False)
		with patch("data_collection.collect_elo.urlopen", return_value=response):
			with self.assertRaisesRegex(ValueError, "redirected.*no club data"):
				fetch_team_history("Pisa")
		response.read.assert_not_called()


if __name__ == "__main__":
	unittest.main()
