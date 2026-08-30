from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl
from preprocessing.feature_engineering import rename_and_cast
from preprocessing.match_feature_pipeline import (
	add_match_categorical_features,
	apply_team_name_mapping,
	build_match_features_from_lf,
	join_player_features_by_game_id,
	load_player_features,
)
from preprocessing.odds_integration import load_match_history_and_map, join_odds
from preprocessing.elo_integration import merge_elo_features
from utils.paths import DATA_DIR, MAPPINGS_DIR

# ---------- Config ----------
INPUT_GLOB = str(DATA_DIR / "understat" / "*" / "*" / "matches.parquet")
OUTPUT_DIR = DATA_DIR / "training"
OUTPUT_PARQUET = OUTPUT_DIR / "understat_df.parquet"
UNDERSTAT_MAPPING_PATH = MAPPINGS_DIR / "understat_to_canonical.json"
EUROPEAN_SCHEDULE_PATH = DATA_DIR / "full_schedule" / "european_all.csv"

def main():
    pl.enable_string_cache()

    lf = rename_and_cast(pl.scan_parquet(INPUT_GLOB))
    lf = apply_team_name_mapping(lf, UNDERSTAT_MAPPING_PATH, "Understat data")

    mh = load_match_history_and_map()
    if mh is not None:
        print("Joining Match History data...")
        lf = join_odds(lf, mh)

    print("Merging Elo ratings...")
    lf = merge_elo_features(lf.collect()).lazy()

    final_df = build_match_features_from_lf(lf, EUROPEAN_SCHEDULE_PATH).collect()
    final_df = join_player_features_by_game_id(final_df, load_player_features())
    final_df = add_match_categorical_features(final_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
