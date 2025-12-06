import polars as pl
import json
from pathlib import Path

ELO_HISTORY_PATH = Path("data/eloscores/elo_history.parquet")
MAPPING_PATH = Path("data/mappings/clubelo_to_canonical.json")

def load_elo_data():
    if not ELO_HISTORY_PATH.exists():
        raise FileNotFoundError(f"Elo history not found at {ELO_HISTORY_PATH}")
    
    lf = pl.scan_parquet(str(ELO_HISTORY_PATH))
    
    # Load mapping
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping not found at {MAPPING_PATH}")
    
    with open(MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    
    # Create mapping dataframe
    # Filter out keys that are not in the mapping (if any)
    mapping_list = [{"team_clubelo": k, "team_canonical": v} for k, v in mapping.items()]
    mapping_df = pl.DataFrame(mapping_list)
    
    # Join mapping to elo data
    # elo_history has 'team' column which is ClubElo name
    lf = lf.join(
        mapping_df.lazy(),
        left_on="team",
        right_on="team_clubelo",
        how="inner"
    )
    
    # Select relevant columns
    # We need: team_canonical, from, to, elo
    # Cast datetimes to 'us' to match matches_df (usually 'us')
    lf = lf.select([
        pl.col("team_canonical").alias("team"),
        pl.col("from").cast(pl.Datetime("us")),
        pl.col("to").cast(pl.Datetime("us")),
        pl.col("elo")
    ])
    
    return lf

def merge_elo_features(matches_df: pl.DataFrame) -> pl.DataFrame:
    """
    Enriches matches_df with Elo features.
    matches_df must have 'home_team', 'away_team', 'date'.
    """
    try:
        elo_lf = load_elo_data()
        elo_df = elo_lf.collect()
    except FileNotFoundError as e:
        print(f"Skipping Elo integration: {e}")
        return matches_df
    
    # Ensure join keys are Strings (matches might be Categorical)
    matches_df = matches_df.with_columns([
        pl.col("home_team").cast(pl.Utf8),
        pl.col("away_team").cast(pl.Utf8)
    ])
    
    # Add lookup date
    # Use midnight of the previous day to match Elo 'from'/'to' timestamps which are at midnight
    matches_df = matches_df.with_columns(
        (pl.col("date").dt.truncate("1d") - pl.duration(days=1)).alias("lookup_date")
    )

    # Prepare Elo for ASOF join
    # Sort by group + date for asof join with 'by'
    elo_sorted = elo_df.sort(["team", "from"])
    matches_sorted = matches_df.sort(["home_team", "lookup_date"])

    # Join for Home Team
    matches_with_home = matches_sorted.join_asof(
        elo_sorted,
        left_on="lookup_date",
        right_on="from",
        by_left="home_team",
        by_right="team",
        strategy="backward"
    ).rename({"elo": "home_elo"})
    
    # Check if lookup_date is within [from, to]
    matches_with_home = matches_with_home.with_columns(
        pl.when(pl.col("lookup_date") > pl.col("to"))
        .then(None)
        .otherwise(pl.col("home_elo"))
        .alias("home_elo")
    ).drop(["from", "to"]) # Drop columns from join (team is merged into home_team)
    
    # Join for Away Team
    # Sort by away_team + lookup_date for second join
    matches_with_home_sorted = matches_with_home.sort(["away_team", "lookup_date"])
    
    matches_with_both = matches_with_home_sorted.join_asof(
        elo_sorted,
        left_on="lookup_date",
        right_on="from",
        by_left="away_team",
        by_right="team",
        strategy="backward"
    ).rename({"elo": "away_elo"})
    
    matches_with_both = matches_with_both.with_columns(
        pl.when(pl.col("lookup_date") > pl.col("to"))
        .then(None)
        .otherwise(pl.col("away_elo"))
        .alias("away_elo")
    ).drop(["from", "to", "lookup_date"]) # Drop columns from join (team is merged into away_team)
    
    # Compute features
    matches_final = matches_with_both.with_columns([
        (pl.col("home_elo") - pl.col("away_elo")).alias("elo_diff"),
        (pl.col("home_elo") + pl.col("away_elo")).alias("elo_sum"),
        ((pl.col("home_elo") + pl.col("away_elo")) / 2).alias("elo_mean")
    ])
    
    # Report missing
    missing_home = matches_final.filter(pl.col("home_elo").is_null())
    missing_away = matches_final.filter(pl.col("away_elo").is_null())
    
    if len(missing_home) > 0:
        print(f"Warning: {len(missing_home)} matches missing Home Elo.")
        # print(missing_home.select(["date", "home_team", "season"]).head())
        
    if len(missing_away) > 0:
        print(f"Warning: {len(missing_away)} matches missing Away Elo.")
        # print(missing_away.select(["date", "away_team", "season"]).head())
        
    return matches_final
