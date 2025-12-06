import polars as pl
import json
from pathlib import Path

MATCH_HISTORY_PATH = Path("data/match_history/matches.parquet")
FOOTBALLDATA_MAPPING_PATH = Path("data/mappings/footballdata_to_canonical.json")

def load_match_history_and_map():
    if not MATCH_HISTORY_PATH.exists() or not FOOTBALLDATA_MAPPING_PATH.exists():
        print("Warning: Match history or mapping not found. Skipping join.")
        return None

    mh = pl.read_parquet(MATCH_HISTORY_PATH)
    with open(FOOTBALLDATA_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    
    mh = mh.with_columns([
        pl.col("home_team").replace(mapping).alias("home_team_mapped"),
        pl.col("away_team").replace(mapping).alias("away_team_mapped"),
        # Parse season: '14-15' -> "1415"
        pl.col("season").str.replace("-", "").alias("season_year")
    ])
    
    cols_to_select = [
        "season_year", "home_team_mapped", "away_team_mapped",
        "HS", "AS", "HST", "AST",
        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
        "B365>2.5", "B365<2.5", "B365C>2.5", "B365C<2.5",
        "BbAv>2.5", "BbAv<2.5", "Avg>2.5", "Avg<2.5"
    ]
    
    existing = mh.columns
    cols_to_select = [c for c in cols_to_select if c in existing or c in ["season_year", "home_team_mapped", "away_team_mapped"]]
    
    mh = mh.select(cols_to_select).rename({
        "season_year": "season",
        "home_team_mapped": "home_team",
        "away_team_mapped": "away_team",
        "HS": "home_shots",
        "AS": "away_shots",
        "HST": "home_sot",
        "AST": "away_sot"
    })

    def coalesce_odds_list(cols, alias):
        # Filter cols that exist
        valid_cols = [c for c in cols if c in mh.columns]
        if not valid_cols:
            return pl.lit(None).alias(alias)
        return pl.coalesce([pl.col(c) for c in valid_cols]).alias(alias)

    mh = mh.with_columns([
        coalesce_odds_list(["B365CH", "B365H"], "odds_h"),
        coalesce_odds_list(["B365CD", "B365D"], "odds_d"),
        coalesce_odds_list(["B365CA", "B365A"], "odds_a"),
        coalesce_odds_list(["B365C>2.5", "B365>2.5", "Avg>2.5", "BbAv>2.5"], "odds_over"),
        coalesce_odds_list(["B365C<2.5", "B365<2.5", "Avg<2.5", "BbAv<2.5"], "odds_under"),
    ])
    
    return mh

def join_odds(lf: pl.LazyFrame, mh: pl.DataFrame) -> pl.LazyFrame:
    # Drop existing shots columns from Understat to prefer MatchHistory
    cols_to_drop = ["home_shots", "away_shots", "home_sot", "away_sot"]
    lf_cols = lf.collect_schema().names()
    lf = lf.drop([c for c in cols_to_drop if c in lf_cols])
    
    # Cast join keys to Utf8
    lf = lf.with_columns([
        pl.col("season").cast(pl.Utf8),
        pl.col("home_team").cast(pl.Utf8),
        pl.col("away_team").cast(pl.Utf8)
    ])
    
    mh_lazy = mh.lazy()
    lf = lf.join(mh_lazy, on=["season", "home_team", "away_team"], how="left")
    return lf
