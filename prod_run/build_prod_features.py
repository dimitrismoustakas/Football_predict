import soccerdata as sd
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
import sys
import json

# Add project root to path when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.collect_full_schedule import (
    ensure_league_config,
    european_leagues_for_season,
    fetch_league_schedule,
)
from data_collection.collect_elo import ELO_HISTORY_PATH, collect_elo

from preprocessing.feature_engineering import rename_and_cast
from preprocessing.match_feature_pipeline import (
	add_match_categorical_features,
	apply_team_name_mapping,
	build_match_features_from_lf,
	join_player_features_asof,
)
from preprocessing.odds_integration import load_match_history_and_map, join_odds
from preprocessing.elo_integration import merge_elo_features
from preprocessing.player_feature_engineering import build_player_team_features, load_all_player_data
from utils.paths import DATA_DIR, MAPPINGS_DIR

LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
OUTPUT_DIR = DATA_DIR / "prod"
OUTPUT_PARQUET = OUTPUT_DIR / "features_season.parquet"
PROD_EUROPEAN_SCHEDULE_PATH = OUTPUT_DIR / "european_schedule_current.csv"
PROD_MATCH_HISTORY_PATH = OUTPUT_DIR / "match_history_current.parquet"


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


def get_current_fbref_season_str():
    now = datetime.now()
    if now.month > 6:
        start_year = now.year
    else:
        start_year = now.year - 1
    return f"{start_year}-{start_year + 1}"


def get_current_match_history_season_str():
    now = datetime.now()
    if now.month > 6:
        start_year = now.year
    else:
        start_year = now.year - 1
    return f"{str(start_year)[-2:]}-{str(start_year + 1)[-2:]}"


def refresh_production_european_schedule() -> Path:
    season = get_current_fbref_season_str()
    print(f"Refreshing European schedule for {season}...")
    ensure_league_config()
    european_df = fetch_league_schedule(
        european_leagues_for_season(season),
        season,
        strict=True,
    )
    if european_df.empty:
        raise RuntimeError(f"No European schedule rows fetched for {season}")
    PROD_EUROPEAN_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    european_df.to_csv(PROD_EUROPEAN_SCHEDULE_PATH, index=False)
    print(f"Wrote fresh European schedule to {PROD_EUROPEAN_SCHEDULE_PATH}")
    return PROD_EUROPEAN_SCHEDULE_PATH


def refresh_production_match_history() -> Path:
    season = get_current_match_history_season_str()
    print(f"Refreshing MatchHistory for season {season}...")
    mh = sd.MatchHistory(leagues=LEAGUES, seasons=[season])
    df = mh.read_games()
    if df.empty:
        raise RuntimeError(f"No MatchHistory rows fetched for {season}")
    df = df.reset_index()
    PROD_MATCH_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROD_MATCH_HISTORY_PATH)
    print(f"Wrote fresh MatchHistory to {PROD_MATCH_HISTORY_PATH}")
    return PROD_MATCH_HISTORY_PATH


def _normalize_player_stats_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.reset_index().copy()
    normalized.columns = [c.lower() for c in normalized.columns]
    if "game" in normalized.columns:
        normalized = normalized.rename(columns={"game": "match_id"})
    return normalized


def _validate_current_player_schema(frame: pd.DataFrame, league: str) -> None:
    required_cols = {"league", "season", "match_id", "team", "player"}
    missing_cols = sorted(required_cols - set(frame.columns))
    if missing_cols:
        raise RuntimeError(
            f"Current-season player data for {league} is missing required columns {missing_cols}. "
            f"Available columns: {sorted(frame.columns.tolist())}"
        )


def fetch_current_player_data() -> pl.DataFrame:
    season_str = get_current_season_str()
    print(f"Fetching current player data for season {season_str}...")
    player_frames = []
    failed_leagues = []

    for league in LEAGUES:
        print(f"  Processing player data for {league}...")
        try:
            reader = sd.Understat(leagues=league, seasons=season_str)
            player_stats = reader.read_player_match_stats()
            if player_stats.empty:
                failed_leagues.append(f"{league} (empty player data)")
                continue
            normalized = _normalize_player_stats_frame(player_stats)
            _validate_current_player_schema(normalized, league)
            player_frames.append(normalized)
        except Exception as e:
            failed_leagues.append(f"{league} ({e})")

    if failed_leagues:
        raise RuntimeError(f"Failed to fetch current-season player data for: {', '.join(failed_leagues)}")

    combined = pd.concat(player_frames, ignore_index=True)
    return pl.from_pandas(combined)


def load_production_player_features() -> pl.DataFrame:
    current_season_key = get_current_season_key()
    try:
        local_player_data = load_all_player_data()
    except FileNotFoundError as e:
        raise RuntimeError("Local historical player data is required for production features") from e

    try:
        fresh_player_data = fetch_current_player_data()
    except Exception as e:
        raise RuntimeError("Failed to refresh current-season player data") from e

    print("Merging fresh current-season player data into production player features...")
    fresh_player_data = fresh_player_data.with_columns(pl.col("season").cast(pl.Utf8))
    local_player_data = local_player_data.with_columns(pl.col("season").cast(pl.Utf8))
    local_player_data = local_player_data.filter(
        ~(
            pl.col("league").is_in(LEAGUES)
            & (pl.col("season") == current_season_key)
        )
    )
    combined_player_data = pl.concat([local_player_data, fresh_player_data], how="diagonal_relaxed")

    return build_player_team_features(combined_player_data)


def map_current_elo(elo_current: pl.DataFrame) -> pl.DataFrame:
    with open(MAPPINGS_DIR / "clubelo_to_canonical.json", "r") as f:
        mapping = json.load(f)

    mapping_df = pl.DataFrame([{"team_clubelo": k, "team_canonical": v} for k, v in mapping.items()])
    return elo_current.join(
        mapping_df,
        left_on="team_clubelo",
        right_on="team_clubelo",
        how="inner"
    ).select([pl.col("team_canonical"), pl.col("elo")])


def fill_missing_future_elo(
    matches_df: pl.DataFrame,
    elo_mapped: pl.DataFrame,
    reference_time: datetime | None = None,
) -> pl.DataFrame:
    if elo_mapped.is_empty():
        return matches_df

    fill_reference = pd.Timestamp(reference_time or datetime.utcnow()).tz_localize(None).to_pydatetime()
    future_mask = pl.col("date") >= pl.lit(fill_reference)
    filled = matches_df.join(
        elo_mapped.rename({"team_canonical": "home_team", "elo": "home_elo_curr"}),
        on="home_team",
        how="left"
    ).join(
        elo_mapped.rename({"team_canonical": "away_team", "elo": "away_elo_curr"}),
        on="away_team",
        how="left"
    ).with_columns([
        pl.when(future_mask)
        .then(pl.coalesce([pl.col("home_elo"), pl.col("home_elo_curr")]))
        .otherwise(pl.col("home_elo"))
        .alias("home_elo"),
        pl.when(future_mask)
        .then(pl.coalesce([pl.col("away_elo"), pl.col("away_elo_curr")]))
        .otherwise(pl.col("away_elo"))
        .alias("away_elo"),
    ]).drop(["home_elo_curr", "away_elo_curr"])

    return filled.with_columns([
        (pl.col("home_elo") - pl.col("away_elo")).alias("elo_diff"),
        (pl.col("home_elo") + pl.col("away_elo")).alias("elo_sum"),
        ((pl.col("home_elo") + pl.col("away_elo")) / 2).alias("elo_mean")
    ])

def fetch_current_data(upcoming_fixtures: pd.DataFrame | None = None):
    season_str = get_current_season_str()
    print(f"Fetching data for season {season_str}...")
    
    all_matches = []
    
    for league in LEAGUES:
        print(f"  Processing {league}...")
        reader = sd.Understat(leagues=league, seasons=season_str)

        # 1. Completed matches stats
        stats = reader.read_team_match_stats()
        if stats.empty:
            raise RuntimeError(f"No Understat team-match stats returned for {league} in {season_str}")
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
            
    if not all_matches:
        raise RuntimeError(f"No current-season production data fetched for {season_str}")
        
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
        raise RuntimeError("No data fetched for production features")

    lf = df.lazy()
    lf = rename_and_cast(lf)
    UNDERSTAT_MAPPING_PATH = MAPPINGS_DIR / "understat_to_canonical.json"
    lf = apply_team_name_mapping(lf, UNDERSTAT_MAPPING_PATH, "production data")

    # 2. Join Odds (Match History)
    match_history_path = refresh_production_match_history()
    mh = load_match_history_and_map(match_history_path=match_history_path)
    if mh is None:
        raise RuntimeError("Fresh MatchHistory data and canonical mappings are required for production features.")
    print("Joining Match History data (Odds)...")
    lf = join_odds(lf, mh)

    # --- Elo Integration ---
    print("Merging Elo ratings...")
    df_temp = lf.collect()
    elo_current = collect_elo(df_temp)
    df_temp = merge_elo_features(df_temp, elo_history_path=ELO_HISTORY_PATH)
    
    # Fill missing Elo for future fixtures only, using current ratings.
    missing_mask = df_temp["home_elo"].is_null() | df_temp["away_elo"].is_null()
    if missing_mask.any():
        print("Filling missing future Elo values with current ratings...")
        df_temp = fill_missing_future_elo(df_temp, map_current_elo(elo_current))

    lf = df_temp.lazy()

    european_schedule_path = refresh_production_european_schedule()
    final_df = build_match_features_from_lf(lf, european_schedule_path).collect()
    final_df = join_player_features_asof(final_df, load_production_player_features())
    final_df = add_match_categorical_features(final_df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.write_parquet(OUTPUT_PARQUET, compression="zstd")
    print(f"Wrote: {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
