import soccerdata as sd
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import sys
import os
import json

# Add project root to path
sys.path.append(os.getcwd())

from preprocessing.feature_engineering import rename_and_cast
from preprocessing.match_feature_pipeline import (
	add_match_categorical_features,
	apply_team_name_mapping,
	build_match_features_from_lf,
	join_player_features_asof,
	load_player_features,
)
from preprocessing.odds_integration import load_match_history_and_map, join_odds
from preprocessing.elo_integration import merge_elo_features
from prod_run.elo_scrap import build_prod_elo
from utils.paths import MAPPINGS_DIR

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
OUTPUT_DIR = Path("data/prod")
OUTPUT_PARQUET = OUTPUT_DIR / "features_season.parquet"
EUROPEAN_SCHEDULE_PATH = Path("data/full_schedule/european_all.csv")


def get_current_season_key():
    now = datetime.now()
    if now.month > 6:
        start_year = now.year
    else:
        start_year = now.year - 1
    return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"


def _canonicalize_team_names(frame: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    if frame.empty or not mapping_path.exists():
        return frame
    with open(mapping_path, "r", encoding="utf-8") as file:
        mapping = json.load(file)
    updated = frame.copy()
    updated["home_team"] = updated["home_team"].replace(mapping)
    updated["away_team"] = updated["away_team"].replace(mapping)
    return updated


def _build_team_lookup(stats_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if stats_df.empty:
        return pd.DataFrame(columns=["league", "team", "team_id"]), {}

    understat_mapping_path = MAPPINGS_DIR / "understat_to_canonical.json"
    canonical_stats = _canonicalize_team_names(stats_df, understat_mapping_path)

    home_lookup = canonical_stats[["league", "league_id", "home_team", "home_team_id"]].rename(
        columns={"home_team": "team", "home_team_id": "team_id"}
    )
    away_lookup = canonical_stats[["league", "league_id", "away_team", "away_team_id"]].rename(
        columns={"away_team": "team", "away_team_id": "team_id"}
    )
    team_lookup = pd.concat([home_lookup, away_lookup], ignore_index=True)
    team_lookup = team_lookup.dropna(subset=["league", "team", "team_id"]).drop_duplicates(
        subset=["league", "team"], keep="last"
    )

    league_lookup = (
        team_lookup[["league", "league_id"]]
        .dropna(subset=["league", "league_id"])
        .drop_duplicates(subset=["league"], keep="last")
    )
    league_id_by_league = dict(zip(league_lookup["league"], league_lookup["league_id"]))
    return team_lookup[["league", "team", "team_id"]], league_id_by_league


def _build_upcoming_fixture_rows(upcoming_fixtures: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    if upcoming_fixtures.empty:
        return pd.DataFrame()

    team_lookup, league_id_by_league = _build_team_lookup(stats_df)
    fixtures = upcoming_fixtures.copy()
    fixtures["league"] = fixtures["league_id"]
    fixtures["date"] = pd.to_datetime(fixtures["commence_time"], utc=True, errors="coerce").dt.tz_localize(None)
    fixtures["season"] = get_current_season_key()
    fixtures["match_id"] = fixtures.apply(
        lambda row: f"odds::{row['league']}::{row['date'].isoformat()}::{row['home_team']}::{row['away_team']}",
        axis=1,
    )
    fixtures["game_id"] = fixtures["match_id"]
    fixtures["league_id"] = fixtures["league"].map(league_id_by_league)

    home_ids = team_lookup.rename(columns={"team": "home_team", "team_id": "home_team_id"})
    away_ids = team_lookup.rename(columns={"team": "away_team", "team_id": "away_team_id"})
    fixtures = fixtures.merge(home_ids, on=["league", "home_team"], how="left")
    fixtures = fixtures.merge(away_ids, on=["league", "away_team"], how="left")

    fixture_cols = {
        "league": fixtures["league"],
        "league_id": fixtures["league_id"],
        "season": fixtures["season"],
        "date": fixtures["date"],
        "match_id": fixtures["match_id"],
        "game_id": fixtures["game_id"],
        "home_team": fixtures["home_team"],
        "away_team": fixtures["away_team"],
        "home_team_id": fixtures["home_team_id"],
        "away_team_id": fixtures["away_team_id"],
    }
    return pd.DataFrame(fixture_cols)

def get_current_season_str():
    now = datetime.now()
    if now.month > 6:
        return f"{now.year}/{now.year + 1}"
    else:
        return f"{now.year - 1}/{now.year}"

def fetch_current_data(upcoming_fixtures: pd.DataFrame | None = None):
    season_str = get_current_season_str()
    print(f"Fetching data for season {season_str}...")
    
    all_matches = []
    
    for league in LEAGUES:
        print(f"  Processing {league}...")
        try:
            reader = sd.Understat(leagues=league, seasons=season_str)
            
            # 1. Completed matches stats
            stats = reader.read_team_match_stats()
            if not stats.empty:
                stats = stats.reset_index()
                # Rename columns
                stats.columns = [c.lower() for c in stats.columns]
                rename_map = {
                    "home_shot": "home_shots",
                    "away_shot": "away_shots",
                    "home_shotontarget": "home_sot",
                    "away_shotontarget": "away_sot",
                    "game": "match_id"
                }
                stats = stats.rename(columns=rename_map)
                
                # Ensure date is datetime
                if 'date' in stats.columns:
                    stats['date'] = pd.to_datetime(stats['date'])
                
                all_matches.append(stats)
            
            if upcoming_fixtures is None:
                # 2. Schedule (upcoming fallback when no odds-driven slate is supplied)
                schedule = reader.read_schedule(include_matches_without_data=True)
                if not schedule.empty:
                    schedule = schedule.reset_index()
                    schedule.columns = [c.lower() for c in schedule.columns]
                    if 'game' in schedule.columns:
                        schedule = schedule.rename(columns={'game': 'match_id'})
                    
                    # Ensure date is datetime
                    if 'date' in schedule.columns:
                        schedule['date'] = pd.to_datetime(schedule['date'])
                    
                    all_matches.append(schedule)
                
        except Exception as e:
            print(f"    Error fetching {league}: {e}")
            
    if not all_matches:
        return pl.DataFrame()
        
    # Combine all
    full_df = pd.concat(all_matches, ignore_index=True)

    if upcoming_fixtures is not None and not upcoming_fixtures.empty:
        fixture_rows = _build_upcoming_fixture_rows(upcoming_fixtures, full_df)
        full_df = pd.concat([full_df, fixture_rows], ignore_index=True, sort=False)
    
    # Deduplicate: if a match is in both stats and schedule, keep stats (more cols)
    full_df['non_null_count'] = full_df.notnull().sum(axis=1)
    full_df = full_df.sort_values('non_null_count', ascending=False)
    # match_id is unique per league usually, but let's be safe with league+match_id
    full_df = full_df.drop_duplicates(subset=['league', 'match_id'])
    full_df = full_df.drop(columns=['non_null_count'])

    for col in ["match_id", "game_id", "league_id", "season", "home_team", "away_team"]:
        if col in full_df.columns:
            full_df[col] = full_df[col].map(lambda value: None if pd.isna(value) else str(value))
    
    return pl.from_pandas(full_df)

def main(upcoming_fixtures: pd.DataFrame | None = None):
    pl.enable_string_cache()
    
    # 1. Fetch data
    df = fetch_current_data(upcoming_fixtures=upcoming_fixtures)
    if df.is_empty():
        print("No data fetched.")
        return

    lf = df.lazy()
    lf = rename_and_cast(lf)
    UNDERSTAT_MAPPING_PATH = MAPPINGS_DIR / "understat_to_canonical.json"
    lf = apply_team_name_mapping(lf, UNDERSTAT_MAPPING_PATH, "production data")

    # 2. Join Odds (Match History)
    mh = load_match_history_and_map()
    if mh is not None:
        print("Joining Match History data (Odds)...")
        lf = join_odds(lf, mh)

    # --- Elo Integration ---
    print("Merging Elo ratings...")
    df_temp = lf.collect()
    df_temp = merge_elo_features(df_temp)
    
    # Fill missing Elo with current Elo (for upcoming games)
    missing_mask = df_temp["home_elo"].is_null() | df_temp["away_elo"].is_null()
    if missing_mask.any():
        print("Fetching current Elo for missing values...")
        try:
            elo_paths = build_prod_elo(write_histories=False)
            elo_asof_path = elo_paths["elo_asof"]
            elo_current = pl.read_parquet(elo_asof_path)
            
            with open(MAPPINGS_DIR / "clubelo_to_canonical.json", "r") as f:
                mapping = json.load(f)
            
            mapping_df = pl.DataFrame([{"team_clubelo": k, "team_canonical": v} for k, v in mapping.items()])
            
            elo_mapped = elo_current.join(
                mapping_df,
                left_on="team_clubelo",
                right_on="team_clubelo",
                how="inner"
            ).select([pl.col("team_canonical"), pl.col("elo")])
            
            # Join and fill
            df_temp = df_temp.join(
                elo_mapped.rename({"team_canonical": "home_team", "elo": "home_elo_curr"}),
                on="home_team",
                how="left"
            ).join(
                elo_mapped.rename({"team_canonical": "away_team", "elo": "away_elo_curr"}),
                on="away_team",
                how="left"
            ).with_columns([
                pl.col("home_elo").fill_null(pl.col("home_elo_curr")),
                pl.col("away_elo").fill_null(pl.col("away_elo_curr"))
            ]).drop(["home_elo_curr", "away_elo_curr"])
            
            # Recompute features
            df_temp = df_temp.with_columns([
                (pl.col("home_elo") - pl.col("away_elo")).alias("elo_diff"),
                (pl.col("home_elo") + pl.col("away_elo")).alias("elo_sum"),
                ((pl.col("home_elo") + pl.col("away_elo")) / 2).alias("elo_mean")
            ])
        except Exception as e:
            print(f"Error filling current Elo: {e}")

    lf = df_temp.lazy()

    final_df = build_match_features_from_lf(lf, EUROPEAN_SCHEDULE_PATH).collect()
    final_df = join_player_features_asof(final_df, load_player_features())
    final_df = add_match_categorical_features(final_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
