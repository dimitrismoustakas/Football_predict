import polars as pl
import json
from pathlib import Path
from utils.paths import MAPPINGS_DIR

MATCH_HISTORY_PATH = Path("data/match_history/matches.parquet")
FOOTBALLDATA_MAPPING_PATH = MAPPINGS_DIR / "footballdata_to_canonical.json"


def _first_valid_odds_expr(mh: pl.DataFrame, cols: list[str], alias: str) -> pl.Expr:
    valid_cols = [c for c in cols if c in mh.columns]
    if not valid_cols:
        return pl.lit(None).cast(pl.Float64).alias(alias)

    return pl.coalesce([
        pl.when(pl.col(c) > 1.0)
        .then(pl.col(c).cast(pl.Float64))
        .otherwise(None)
        for c in valid_cols
    ]).alias(alias)

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
        # B365 odds
        "B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
        # Pinnacle odds (fallback)
        "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
        # Average odds (fallback)
        "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA",
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

    mh = mh.with_columns([
        # Home/Draw/Away odds - prefer B365, then Pinnacle, then average.
        # Ignore invalid values (<= 1.0) so broken closing prices don't mask valid fallbacks.
        _first_valid_odds_expr(mh, ["B365H", "B365CH", "PSH", "PSCH", "AvgH", "AvgCH"], "odds_h"),
        _first_valid_odds_expr(mh, ["B365D", "B365CD", "PSD", "PSCD", "AvgD", "AvgCD"], "odds_d"),
        _first_valid_odds_expr(mh, ["B365A", "B365CA", "PSA", "PSCA", "AvgA", "AvgCA"], "odds_a"),
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