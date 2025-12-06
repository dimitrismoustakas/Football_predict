# features_lib.py
from __future__ import annotations
from typing import Iterable, List
import polars as pl

ROLL_WINDOWS = [3, 5, 10]

BASE_STATS_FOR = ["xg_for", "shots_for", "sot_for", "deep_for", "ppda_for", "gf"]
BASE_STATS_AGAINST = [
    "xg_against", "shots_against", "sot_against", "deep_against", "ppda_against", "ga"
]
DERIVED_STATS = ["xgd", "gd", "points", "win", "draw", "loss"]


def rename_and_cast(lf: pl.LazyFrame) -> pl.LazyFrame:
    cols = set(lf.collect_schema().names())
    rename_map = {
        "team_h": "home_team",
        "team_a": "away_team",
        "h_goals": "home_goals",
        "a_goals": "away_goals",
        "h_xg": "home_xg",
        "a_xg": "away_xg",
        "h_shot": "home_shots",
        "a_shot": "away_shots",
        "h_shotOnTarget": "home_sot",
        "a_shotOnTarget": "away_sot",
        "h_deep": "home_deep",
        "a_deep": "away_deep",
        "h_ppda": "home_ppda",
        "a_ppda": "away_ppda",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in cols}
    if rename_map:
        lf = lf.rename(rename_map)
        cols = set(lf.collect_schema().names())

    schema = lf.collect_schema()
    date_dtype = schema.get("date", None)
    if date_dtype is not None:
        if date_dtype == pl.Utf8:
            lf = lf.with_columns(
                pl.coalesce(
                    [
                        pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False),
                        pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d", strict=False),
                        pl.col("date").str.strptime(pl.Datetime, strict=False),
                    ]
                ).dt.replace_time_zone("UTC").alias("date")
            )
        elif date_dtype == pl.Date:
            lf = lf.with_columns(pl.col("date").cast(pl.Datetime).dt.replace_time_zone("UTC"))

    wanted_casts = {
        "id": pl.Utf8, "fid": pl.Utf8, "match_id": pl.Utf8, "league_id": pl.Utf8,
        "league": pl.Utf8, "season": pl.Utf8, "home_team": pl.Utf8, "away_team": pl.Utf8,
        "home_goals": pl.Int32, "away_goals": pl.Int32,
        "home_xg": pl.Float64, "away_xg": pl.Float64,
        "home_shots": pl.Float64, "away_shots": pl.Float64,
        "home_sot": pl.Float64, "away_sot": pl.Float64,
        "home_deep": pl.Float64, "away_deep": pl.Float64,
        "home_ppda": pl.Float64, "away_ppda": pl.Float64,
    }
    for c, dt in wanted_casts.items():
        if c in cols:
            lf = lf.with_columns(pl.col(c).cast(dt))

    cat_cols = [c for c in ("home_team","away_team","league","league_id","season") if c in cols]
    if cat_cols:
        lf = lf.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols])

    sort_keys = [k for k in ("league_id","season","date","match_id") if k in cols]
    if sort_keys:
        lf = lf.sort(sort_keys)
    return lf


def build_long(base: pl.LazyFrame) -> pl.LazyFrame:
    base_cols = [c for c in ("match_id","league_id","league","season","date")
                 if c in base.collect_schema().names()]

    home_rows = base.select(
        *base_cols,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_goals").alias("gf"),
        pl.col("away_goals").alias("ga"),
        pl.col("home_xg").alias("xg_for"),
        pl.col("away_xg").alias("xg_against"),
        pl.col("home_shots").alias("shots_for"),
        pl.col("away_shots").alias("shots_against"),
        pl.col("home_sot").alias("sot_for"),
        pl.col("away_sot").alias("sot_against"),
        pl.col("home_deep").alias("deep_for"),
        pl.col("away_deep").alias("deep_against"),
        pl.col("home_ppda").alias("ppda_for"),
        pl.col("away_ppda").alias("ppda_against"),
        pl.lit("h").alias("side"),
    )

    away_rows = base.select(
        *base_cols,
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.col("away_goals").alias("gf"),
        pl.col("home_goals").alias("ga"),
        pl.col("away_xg").alias("xg_for"),
        pl.col("home_xg").alias("xg_against"),
        pl.col("away_shots").alias("shots_for"),
        pl.col("home_shots").alias("shots_against"),
        pl.col("away_sot").alias("sot_for"),
        pl.col("home_sot").alias("sot_against"),
        pl.col("away_deep").alias("deep_for"),
        pl.col("home_deep").alias("deep_against"),
        pl.col("away_ppda").alias("ppda_for"),
        pl.col("home_ppda").alias("ppda_against"),
        pl.lit("a").alias("side"),
    )

    long_df = (
        pl.concat([home_rows, away_rows])
        .with_columns([
            (pl.col("xg_for") - pl.col("xg_against")).alias("xgd"),
            (pl.col("gf") - pl.col("ga")).alias("gd"),
            (pl.col("gf") > pl.col("ga")).cast(pl.Int8).alias("win"),
            (pl.col("gf") == pl.col("ga")).cast(pl.Int8).alias("draw"),
            (pl.col("gf") < pl.col("ga")).cast(pl.Int8).alias("loss"),
        ])
        .with_columns((3 * pl.col("win") + pl.col("draw")).alias("points"))
    )

    cols = set(long_df.collect_schema().names())
    sort_keys = [k for k in ("league_id", "season") if k in cols]
    long_df = long_df.sort(sort_keys + ["team", "date", "match_id"])

    return long_df


def rolling_feature_exprs(scope: str, window: int) -> List[pl.Expr]:
    gkeys = ["league_id", "season", "team"]
    exprs: List[pl.Expr] = []

    def scoped(colname: str) -> pl.Expr:
        base = pl.col(colname)
        if scope == "home":
            base = pl.when(pl.col("is_home")).then(base).otherwise(None)
        elif scope == "away":
            base = pl.when(~pl.col("is_home")).then(base).otherwise(None)
        return base

    stats = BASE_STATS_FOR + BASE_STATS_AGAINST + DERIVED_STATS
    for s in stats:
        series = scoped(s).shift(1)
        exprs += [
            series.rolling_mean(window_size=window, min_samples=2).over(gkeys).alias(f"{scope}__{s}__r{window}"),
            series.rolling_sum(window_size=window, min_samples=2).over(gkeys).alias(f"{scope}__{s}__sum__r{window}"),
        ]

    ones = (pl.col("gf") * 0 + 1).cast(pl.Int32)
    if scope == "ovr":
        mask = ones
    elif scope == "home":
        mask = pl.when(pl.col("is_home")).then(ones).otherwise(None)
    else:
        mask = pl.when(~pl.col("is_home")).then(ones).otherwise(None)

    exprs.append(
        mask.shift(1).rolling_sum(window_size=window, min_samples=1).over(gkeys).alias(f"{scope}__games__r{window}")
    )
    return exprs


def compute_historical_rolling_features(long_df: pl.LazyFrame) -> pl.LazyFrame:
    exprs: List[pl.Expr] = []
    for w in ROLL_WINDOWS:
        for scope in ("ovr","home","away"):
            exprs += rolling_feature_exprs(scope, w)
    return long_df.with_columns(exprs)


def build_match_level_with_rolling_features(base_matches: pl.LazyFrame, long_feats: pl.LazyFrame) -> pl.LazyFrame:
    """
    Join team-level rolling features (long_feats) to upcoming fixtures (base_matches),
    producing one row per match with separate home/away feature blocks.
    This version uses an asof join to get the features as of each match date.
    """
    join_keys = ["league_id", "season"]
    
    # --- Normalize join keys to Utf8 ---
    base_matches = base_matches.with_columns([pl.col(c).cast(pl.Utf8) for c in join_keys + ["home_team", "away_team"]])
    long_feats = long_feats.with_columns([pl.col(c).cast(pl.Utf8) for c in join_keys + ["team"]])

    # --- Discover feature columns from schema (exclude identifiers) ---
    feature_cols = [c for c in long_feats.collect_schema().names() if c.startswith(("ovr__", "home__", "away__"))]

    # --- Prepare HOME feature block ---
    home_feats = long_feats.select(["date"] + join_keys + ["team"] + feature_cols)
    home_feats = home_feats.rename({c: f"{c}__h" for c in feature_cols})
    home_feats = home_feats.rename({"team": "home_team"})

    # --- Prepare AWAY feature block ---
    away_feats = long_feats.select(["date"] + join_keys + ["team"] + feature_cols)
    away_feats = away_feats.rename({c: f"{c}__a" for c in feature_cols})
    away_feats = away_feats.rename({"team": "away_team"})

    # --- Join to fixtures ---
    out = base_matches.join_asof(
        home_feats, on="date", by=join_keys + ["home_team"], strategy="backward"
    ).join_asof(
        away_feats, on="date", by=join_keys + ["away_team"], strategy="backward"
    )

    # --- Stable sort (use whatever ordering cols exist) ---
    order_cols = [c for c in ["league_id", "season", "date", "match_id", "home_team", "away_team"]
                  if c in out.collect_schema().names()]
    if order_cols:
        out = out.sort(order_cols)

    return out


def build_match_level_with_latest_features(base_matches: pl.LazyFrame, long_feats: pl.LazyFrame) -> pl.LazyFrame:
    """
    Join team-level rolling features (long_feats) to upcoming fixtures (base_matches),
    producing one row per match with separate home/away feature blocks.
    This version takes the LATEST features for each team from long_feats.
    """

    # --- Normalize join keys on the fixtures side (strings) ---
    base_matches = base_matches.with_columns([
        pl.col(c).cast(pl.Utf8) for c in ["home_team", "away_team", "league_id", "season"]
        if c in base_matches.collect_schema().names()
    ])

    gkeys = ["league_id", "season", "team"]
    gkeys = [k for k in gkeys if k in long_feats.collect_schema().names()]
    if "team" not in gkeys:
        raise ValueError("long_feats must have a 'team' column")

    # Cast group keys to Utf8 before grouping
    long_feats_str_keys = long_feats.with_columns([
        pl.col(c).cast(pl.Utf8) for c in gkeys
    ])

    latest_feats = long_feats_str_keys.group_by(gkeys).last()

    # Discover feature columns from schema (exclude identifiers)
    lf_schema_cols = latest_feats.collect_schema().names()
    feature_cols = [c for c in lf_schema_cols if c not in gkeys]

    # --- Prepare HOME feature block ---
    home_feats = latest_feats.rename({c: f"{c}__h" for c in feature_cols})
    home_feats_renamed = home_feats.rename({"team": "home_team"})

    # --- Prepare AWAY feature block ---
    away_feats = latest_feats.rename({c: f"{c}__a" for c in feature_cols})
    away_feats_renamed = away_feats.rename({"team": "away_team"})

    # --- Join to fixtures ---
    join_keys = [k for k in gkeys if k != "team"]
    home_join_keys = join_keys + ["home_team"]
    away_join_keys = join_keys + ["away_team"]

    out = (
        base_matches
        .join(home_feats_renamed, on=home_join_keys, how="left")
        .join(away_feats_renamed, on=away_join_keys, how="left")
    )

    # --- Stable sort (use whatever ordering cols exist) ---
    order_cols = [c for c in ["league_id", "season", "date", "match_id", "home_team", "away_team"]
                  if c in out.collect_schema().names()]
    if order_cols:
        out = out.sort(order_cols)

    return out