import polars as pl

# ---------- Constants ----------
ROLL_WINDOWS = [3, 5, 10]

# Updated stats based on available soccerdata columns
# Missing: shots_for, sot_for (commented out)
# Added: npxg_for
BASE_STATS_FOR = [
    "xg_for", 
    "npxg_for",
    "shots_for", 
    "sot_for", 
    "deep_for", 
    "ppda_for", 
    "gf"
]
BASE_STATS_AGAINST = [
    "xg_against",
    "npxg_against",
    "shots_against",
    "sot_against",
    "deep_against",
    "ppda_against",
    "ga",
]
DERIVED_STATS = ["xgd", "gd", "points", "win", "draw", "loss"]


# ---------- Helpers ----------
def rename_and_cast(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Standardize column names and types from Understat matches.parquet files.
    Robust datetime parsing for strings like '2017-08-18 19:30:00'.
    """
    # Resolve schema once
    cols = set(lf.collect_schema().names())

    # Map soccerdata columns to internal names
    # soccerdata: home_deep_completions, home_np_xg, etc.
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
        # New mappings for soccerdata
        "home_deep_completions": "home_deep",
        "away_deep_completions": "away_deep",
        "home_np_xg": "home_npxg",
        "away_np_xg": "away_npxg",
        # "game_id": "match_id" # match_id already exists
    }
    rename_map = {k: v for k, v in rename_map.items() if k in cols}
    if rename_map:
        lf = lf.rename(rename_map)
        cols = set(lf.collect_schema().names())

    schema = lf.collect_schema()
    date_dtype = schema.get("date", None)

    if date_dtype is not None:
        if date_dtype == pl.Utf8:
            # Parse common formats; keep as Datetime
            lf = lf.with_columns(
                pl.coalesce(
                    [
                        pl.col("date").str.strptime(
                            pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False
                        ),
                        pl.col("date").str.strptime(
                            pl.Datetime, format="%Y-%m-%d", strict=False
                        ),
                        pl.col("date").str.strptime(pl.Datetime, strict=False),
                    ]
                ).alias("date")
            )
        elif date_dtype == pl.Date:
            # If it's a Date already and you want Datetime for sorting/joins:
            lf = lf.with_columns(pl.col("date").cast(pl.Datetime))
        # If it's already Datetime, do nothing

    # Target casts (apply only if column exists)
    wanted_casts = {
        "id": pl.Utf8,
        "fid": pl.Utf8,
        "match_id": pl.Utf8,
        "league_id": pl.Utf8,
        "league": pl.Utf8,
        "season": pl.Utf8,
        "home_team": pl.Utf8,
        "away_team": pl.Utf8,
        "home_goals": pl.Int32,
        "away_goals": pl.Int32,
        "home_xg": pl.Float64,
        "away_xg": pl.Float64,
        "home_npxg": pl.Float64,
        "away_npxg": pl.Float64,
        "home_shots": pl.Float64,
        "away_shots": pl.Float64,
        "home_sot": pl.Float64,
        "away_sot": pl.Float64,
        "home_deep": pl.Float64,
        "away_deep": pl.Float64,
        "home_ppda": pl.Float64,
        "away_ppda": pl.Float64,
    }
    for c, dt in wanted_casts.items():
        if c in cols:
            lf = lf.with_columns(pl.col(c).cast(dt))

    # Categorical for keys (if present)
    cat_cols = [
        c
        for c in ("home_team", "away_team", "league", "league_id", "season")
        if c in cols
    ]
    if cat_cols:
        lf = lf.with_columns([pl.col(c).cast(pl.Categorical) for c in cat_cols])

    # Deterministic order
    sort_keys = [k for k in ("league_id", "season", "date", "match_id") if k in cols]
    if sort_keys:
        lf = lf.sort(sort_keys)

    return lf


def build_long(base: pl.LazyFrame) -> pl.LazyFrame:
    base_cols = [
        c
        for c in ("match_id", "league_id", "league", "season", "date")
        if c in base.collect_schema().names()
    ]
    
    # Helper to safely select columns if they exist, else null
    # But base should have them if we filtered correctly in main.
    # However, shots/sot might be missing now.
    
    # We need to construct the expressions dynamically based on what's available or fill nulls.
    # Since we commented them out in BASE_STATS, we don't strictly need them in the output 
    # unless we want to keep the column structure.
    # Let's try to keep them but fill with null if missing, so downstream code doesn't break if it expects them.
    
    available = set(base.collect_schema().names())
    
    def safe_col(name, alias):
        if name in available:
            return pl.col(name).alias(alias)
        else:
            return pl.lit(None).cast(pl.Float64).alias(alias)

    home_rows = base.select(
        *base_cols,
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.col("home_goals").alias("gf"),
        pl.col("away_goals").alias("ga"),
        pl.col("home_xg").alias("xg_for"),
        pl.col("away_xg").alias("xg_against"),
        safe_col("home_npxg", "npxg_for"),
        safe_col("away_npxg", "npxg_against"),
        safe_col("home_shots", "shots_for"),
        safe_col("away_shots", "shots_against"),
        safe_col("home_sot", "sot_for"),
        safe_col("away_sot", "sot_against"),
        safe_col("home_deep", "deep_for"),
        safe_col("away_deep", "deep_against"),
        safe_col("home_ppda", "ppda_for"),
        safe_col("away_ppda", "ppda_against"),
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
        safe_col("away_npxg", "npxg_for"),
        safe_col("home_npxg", "npxg_against"),
        safe_col("away_shots", "shots_for"),
        safe_col("home_shots", "shots_against"),
        safe_col("away_sot", "sot_for"),
        safe_col("home_sot", "sot_against"),
        safe_col("away_deep", "deep_for"),
        safe_col("home_deep", "deep_against"),
        safe_col("away_ppda", "ppda_for"),
        safe_col("home_ppda", "ppda_against"),
        pl.lit("a").alias("side"),
    )

    long_df = (
        pl.concat([home_rows, away_rows])
        .with_columns(
            [
                (pl.col("xg_for") - pl.col("xg_against")).alias("xgd"),
                (pl.col("gf") - pl.col("ga")).alias("gd"),
                (pl.col("gf") > pl.col("ga")).cast(pl.Int8).alias("win"),
                (pl.col("gf") == pl.col("ga")).cast(pl.Int8).alias("draw"),
                (pl.col("gf") < pl.col("ga")).cast(pl.Int8).alias("loss"),
            ]
        )
        .with_columns((3 * pl.col("win") + pl.col("draw")).alias("points"))
    )

    # Stable team order per league-season
    sort_keys = [
        k for k in ("league_id", "season") if k in long_df.collect_schema().names()
    ]
    long_df = long_df.sort(sort_keys + ["team", "date", "match_id"])
    return long_df


def rolling_feature_exprs(scope: str, window: int):
    gkeys = ["league_id", "season", "team"]
    exprs = []

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
            series.rolling_mean(window_size=window, min_samples=2)
            .over(gkeys)
            .alias(f"{scope}__{s}__r{window}"),
            series.rolling_sum(window_size=window, min_samples=2)
            .over(gkeys)
            .alias(f"{scope}__{s}__sum__r{window}"),
        ]

    # --- per-row ones (avoid literal) ---
    ones = (pl.col("gf") * 0 + 1).cast(pl.Int32)  # any existing column works

    if scope == "ovr":
        mask = ones
    elif scope == "home":
        mask = pl.when(pl.col("is_home")).then(ones).otherwise(None)
    else:  # "away"
        mask = pl.when(~pl.col("is_home")).then(ones).otherwise(None)

    exprs.append(
        mask.shift(1)
        .rolling_sum(window_size=window, min_samples=1)
        .over(gkeys)
        .alias(f"{scope}__games__r{window}")
    )

    return exprs


def compute_rolling_features(long_df: pl.LazyFrame) -> pl.LazyFrame:
    exprs = []
    for w in ROLL_WINDOWS:
        for scope in ("ovr", "home", "away"):
            exprs += rolling_feature_exprs(scope, w)
    return long_df.with_columns(exprs)


def with_side_suffix(lf: pl.LazyFrame, suffix: str) -> pl.LazyFrame:
    """
    Add a side suffix (e.g., '__h' or '__a') to all engineered feature columns in this frame.
    """
    names = lf.collect_schema().names()
    feat_cols = [c for c in names if c.startswith(("ovr__", "home__", "away__"))]
    rename_map = {c: f"{c}{suffix}" for c in feat_cols}
    return lf.rename(rename_map)


def build_match_level(
    base_matches: pl.LazyFrame, long_with_feats: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Join features back to match level.
    We tag home features with '__h' and away with '__a' BEFORE the join to avoid collisions.
    Adds target 'Over'.
    """
    # Minimal feature frames per side
    feat_cols = [
        c
        for c in long_with_feats.collect_schema().names()
        if c.startswith(("ovr__", "home__", "away__"))
    ]
    keep_cols = ["match_id", "team", "side"] + feat_cols

    feats = long_with_feats.select(keep_cols)

    home_feats = feats.filter(pl.col("side") == "h").drop("side")
    away_feats = feats.filter(pl.col("side") == "a").drop("side")

    home_feats = with_side_suffix(home_feats, "__h").rename({"team": "home_team"})
    away_feats = with_side_suffix(away_feats, "__a").rename({"team": "away_team"})

    # Match-level base with Over target
    match_base = base_matches.with_columns(
        (pl.col("home_goals") + pl.col("away_goals") > 2.5).cast(pl.Int8).alias("Over")
    )

    out = match_base.join(home_feats, on=["match_id", "home_team"], how="left").join(
        away_feats, on=["match_id", "away_team"], how="left"
    )

    order_cols = [
        "match_id",
        "league_id",
        "league",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "Over",
    ]
    existing = out.collect_schema().names()
    order_cols = [c for c in order_cols if c in existing]
    remaining = [c for c in existing if c not in order_cols]
    out = out.select(order_cols + remaining)
    return out
