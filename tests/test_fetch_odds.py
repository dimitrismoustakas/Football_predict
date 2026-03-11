import unittest

from prod_run.fetch_odds import parse_odds_data


def _build_game(bookmakers):
	return {
		"league_id": "ENG-Premier League",
		"commence_time": "2026-03-14T15:00:00Z",
		"home_team": "Manchester United",
		"away_team": "Arsenal",
		"bookmakers": bookmakers,
	}


def _bookmaker(key, home_price, draw_price, away_price):
	return {
		"key": key,
		"markets": [
			{
				"key": "h2h",
				"outcomes": [
					{"name": "Manchester United", "price": home_price},
					{"name": "Draw", "price": draw_price},
					{"name": "Arsenal", "price": away_price},
				],
			}
		],
	}


class FetchOddsTests(unittest.TestCase):
	def test_parse_odds_data_prefers_betsson(self):
		games = [
			_build_game([
				_bookmaker("williamhill", 2.8, 3.3, 2.5),
				_bookmaker("betsson", 2.6, 3.4, 2.7),
			])
		]

		parsed = parse_odds_data(games)

		self.assertEqual(len(parsed), 1)
		self.assertEqual(parsed[0]["odds_home"], 2.6)
		self.assertEqual(parsed[0]["odds_draw"], 3.4)
		self.assertEqual(parsed[0]["odds_away"], 2.7)

	def test_parse_odds_data_falls_back_to_williamhill(self):
		games = [
			_build_game([
				_bookmaker("williamhill", 2.8, 3.3, 2.5),
			])
		]

		parsed = parse_odds_data(games)

		self.assertEqual(len(parsed), 1)
		self.assertEqual(parsed[0]["odds_home"], 2.8)
		self.assertEqual(parsed[0]["odds_draw"], 3.3)
		self.assertEqual(parsed[0]["odds_away"], 2.5)

	def test_parse_odds_data_skips_games_without_supported_bookmaker(self):
		games = [
			_build_game([
				_bookmaker("onexbet", 2.7, 3.2, 2.6),
			])
		]

		parsed = parse_odds_data(games)

		self.assertEqual(parsed, [])


if __name__ == "__main__":
	unittest.main()