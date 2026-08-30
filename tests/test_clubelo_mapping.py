import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

from preprocessing import generate_clubelo_mapping
from preprocessing.elo_integration import merge_elo_features


class ClubEloMappingTests(unittest.TestCase):
	def test_generator_uses_exact_names_and_explicit_aliases(self):
		expected = {
			"Barcelona": "Barcelona",
			"Bastia": "SC Bastia",
			"Milan": "AC Milan",
			"Real Madrid": "Real Madrid",
			"Sevilla": "Sevilla",
			"Spal": "SPAL 2013",
			"Villarreal": "Villarreal",
		}
		other_clubs = ["Barcelona B", "CA Bastia", "Real Madrid B", "Sevilla B", "Villarreal B"]
		with TemporaryDirectory() as directory:
			root = Path(directory)
			universe = root / "universe.parquet"
			canonical = root / "understat.json"
			output = root / "clubelo.json"
			pl.DataFrame({"team": list(expected) + other_clubs}).write_parquet(universe)
			canonical.write_text(json.dumps({name: name for name in expected.values()}))
			with patch.multiple(
				generate_clubelo_mapping,
				ELO_UNIVERSE_PATH=universe,
				UNDERSTAT_MAPPING_PATH=canonical,
				OUTPUT_MAPPING_PATH=output,
				MAPPINGS_DIR=root,
			):
				generate_clubelo_mapping.main()
			self.assertEqual(json.loads(output.read_text()), expected)

	def test_elo_join_ignores_distinct_clubs_and_later_updates(self):
		clubs = [
			("Barcelona", "Barcelona B", "Barcelona"),
			("Bastia", "CA Bastia", "SC Bastia"),
			("Real Madrid", "Real Madrid B", "Real Madrid"),
			("Sevilla", "Sevilla B", "Sevilla"),
			("Villarreal", "Villarreal B", "Villarreal"),
		]
		rows = [{"team": "Arsenal", "from": datetime(2015, 4, 16), "elo": 1850.0}]
		for index, (first_team, other_team, _) in enumerate(clubs):
			rows.extend([
				{"team": first_team, "from": datetime(2015, 4, 16), "elo": 2000.0 + index},
				{"team": other_team, "from": datetime(2015, 4, 17), "elo": 1400.0 + index},
			])
		history = pl.DataFrame(rows).with_columns(pl.lit(datetime(2015, 4, 30)).alias("to"))
		matches = pl.DataFrame({
			"game_id": list(range(len(clubs))),
			"date": [datetime(2015, 4, 18)] * len(clubs),
			"home_team": [canonical for _, _, canonical in clubs],
			"away_team": ["Arsenal"] * len(clubs),
		})
		with TemporaryDirectory() as directory:
			path = Path(directory) / "elo.parquet"
			history.write_parquet(path)
			before = merge_elo_features(matches, path).sort("game_id")
			self.assertEqual(before["home_elo"].to_list(), [2000.0 + i for i in range(len(clubs))])
			self.assertEqual(before["away_elo"].to_list(), [1850.0] * len(clubs))

			later = history.filter(pl.col("team").is_in([first for first, _, _ in clubs])).with_columns(
				pl.lit(datetime(2015, 4, 18)).alias("from"),
				(pl.col("elo") + 100.0).alias("elo"),
			)
			pl.concat([history, later]).reverse().write_parquet(path)
			after = merge_elo_features(matches, path).sort("game_id")
			assert_frame_equal(before, after, check_exact=True)


if __name__ == "__main__":
	unittest.main()
