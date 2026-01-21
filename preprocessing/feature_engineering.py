import polars as pl
import json
from pathlib import Path

# ---------- Constants ----------
ROLL_WINDOWS = [3, 5, 10]
PROJECT_ROOT = Path(__file__).parent.parent
MAPPINGS_DIR = PROJECT_ROOT / "data" / "mappings"
PROMOTED_TEAMS_PATH = PROJECT_ROOT / "data" / "promoted_teams.json"

# League IDs for embedding lookup (consistent ordering)
LEAGUE_IDS = {
	"ENG-Premier League": 0,
	"ESP-La Liga": 1,
	"FRA-Ligue 1": 2,
	"GER-Bundesliga": 3,
	"ITA-Serie A": 4,
}

# Season stage boundaries (as percentages of season)
# Early: first ~21% (rounds 1-8 of 38), Mid: ~21-68% (rounds 9-26), Late: ~68%+ (27+)
SEASON_STAGE_EARLY_PCT = 0.21
SEASON_STAGE_MID_PCT = 0.68

# FBRef schedule path for week/round data
FBREF_SCHEDULE_PATH = PROJECT_ROOT / "data" / "full_schedule" / "domestic_all.csv"

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


# ---------- Categorical Feature Helpers ----------

def load_promoted_teams() -> dict[str, dict[str, list[str]]]:
	"""
	Load promoted teams data from JSON file.
	Returns: {league: {season: [team1, team2, ...]}}
	"""
	if not PROMOTED_TEAMS_PATH.exists():
		print(f"Warning: {PROMOTED_TEAMS_PATH} not found. Run compute_promoted_teams.py first.")
		return {}
	
	with open(PROMOTED_TEAMS_PATH, "r", encoding="utf-8") as f:
		data = json.load(f)
	
	# Filter out metadata keys (start with _)
	return {k: v for k, v in data.items() if not k.startswith("_")}


def build_promoted_teams_set(promoted_data: dict) -> dict[str, set[str]]:
	"""
	Build a lookup dict: {(league, season): set of promoted teams}
	For efficient lookup during feature engineering.
	"""
	result = {}
	for league, seasons in promoted_data.items():
		for season, teams in seasons.items():
			key = f"{league}_{season}"
			result[key] = set(teams)
	return result


def load_fbref_week_data() -> pl.LazyFrame | None:
	"""
	Load FBRef schedule data with week/round numbers.
	Returns a lookup table: (league, season, home_team, away_team) → week.
	"""
	if not FBREF_SCHEDULE_PATH.exists():
		return None
	
	# Load and select relevant columns
	df = pl.scan_csv(FBREF_SCHEDULE_PATH)
	
	# Load FBRef-to-canonical mapping
	fbref_mapping_path = MAPPINGS_DIR / "fbref_to_canonical.json"
	if fbref_mapping_path.exists():
		with open(fbref_mapping_path, "r", encoding="utf-8") as f:
			fbref_mapping = json.load(f)
		# Filter out None values (unmapped teams)
		fbref_mapping = {k: v for k, v in fbref_mapping.items() if v is not None}
	else:
		fbref_mapping = {}
	
	# Select columns and apply team name mapping
	df = df.select([
		pl.col("league").cast(pl.Utf8),
		pl.col("season").cast(pl.Utf8),
		pl.col("home_team").replace(fbref_mapping).cast(pl.Utf8),
		pl.col("away_team").replace(fbref_mapping).cast(pl.Utf8),
		pl.col("week").cast(pl.Int32).alias("fbref_week"),
	])
	
	return df


def compute_round_number(lf: pl.LazyFrame, fbref_data: pl.LazyFrame | None = None) -> pl.LazyFrame:
	"""
	Compute round number for each match within a league-season.
	
	Joins on (league, season, home_team, away_team) which uniquely identifies
	each match since teams play each other exactly once home and once away per season.
	
	Returns LazyFrame with 'round_number' column added.
	"""
	if fbref_data is None:
		raise ValueError("FBRef data is required for round_number computation. Run collect_full_schedule.py first.")
	
	# Cast fbref_data columns and deduplicate
	# FBRef may have duplicate entries for rescheduled matches (same fixture with different dates)
	# We prefer the row with a valid week number over null
	fbref_join = (
		fbref_data.select([
			pl.col("league").cast(pl.Categorical),
			pl.col("season").cast(pl.Utf8),
			pl.col("home_team").cast(pl.Utf8),
			pl.col("away_team").cast(pl.Utf8),
			pl.col("fbref_week"),
		])
		# Sort to put non-null weeks first, then deduplicate
		.sort("fbref_week", nulls_last=True)
		.unique(subset=["league", "season", "home_team", "away_team"], keep="first")
	)
	
	# Join on league, season, home_team, away_team (uniquely identifies each match)
	lf = lf.join(
		fbref_join,
		on=["league", "season", "home_team", "away_team"],
		how="left",
	)
	
	# Rename fbref_week to round_number
	lf = lf.with_columns(
		pl.col("fbref_week").alias("round_number")
	).drop("fbref_week")
	
	return lf


def compute_season_stage(lf: pl.LazyFrame) -> pl.LazyFrame:
	"""
	Compute season stage based on round number relative to max round in league-season.
	Uses percentage thresholds to handle varying season lengths:
	- 'early': first ~21% of season
	- 'mid': ~21-68% of season  
	- 'late': final ~32% of season
	
	Returns LazyFrame with 'season_stage' column added.
	"""
	# Compute max round per league-season for percentage calculation
	lf = lf.with_columns(
		pl.col("round_number").max().over(["league_id", "season"]).alias("_max_round")
	)
	
	# Compute relative position in season
	lf = lf.with_columns(
		(pl.col("round_number") / pl.col("_max_round")).alias("_season_pct")
	)
	
	# Assign season stage based on percentage thresholds
	lf = lf.with_columns(
		pl.when(pl.col("_season_pct") <= SEASON_STAGE_EARLY_PCT)
		.then(pl.lit("early"))
		.when(pl.col("_season_pct") <= SEASON_STAGE_MID_PCT)
		.then(pl.lit("mid"))
		.otherwise(pl.lit("late"))
		.alias("season_stage")
	).drop(["_max_round", "_season_pct"])
	return lf


def add_league_id_numeric(lf: pl.LazyFrame) -> pl.LazyFrame:
	"""
	Add numeric league ID for embedding lookup.
	Maps league names to integers 0-4.
	"""
	lf = lf.with_columns(
		pl.col("league").cast(pl.Utf8).replace(LEAGUE_IDS, default=None).alias("league_idx")
	)
	return lf


def add_promoted_flags(
	lf: pl.LazyFrame,
	promoted_lookup: dict[str, set[str]]
) -> pl.LazyFrame:
	"""
	Add binary flags for whether home/away team was promoted this season.
	
	Args:
		lf: LazyFrame with match data (must have league, season, home_team, away_team)
		promoted_lookup: Dict mapping "league_season" to set of promoted team names
	
	Returns: LazyFrame with 'home_promoted' and 'away_promoted' columns (0/1).
	"""
	# Build a DataFrame of all promoted teams for efficient join
	promoted_rows = []
	for key, teams in promoted_lookup.items():
		parts = key.rsplit("_", 1)  # "ENG-Premier League_1415" -> ["ENG-Premier League", "1415"]
		if len(parts) == 2:
			league, season = parts
			for team in teams:
				promoted_rows.append({
					"league": league,
					"season": season,
					"team": team,
					"is_promoted": 1,  # Marker column
				})
	
	if not promoted_rows:
		# No promoted teams data - add 0 columns (not promoted)
		return lf.with_columns([
			pl.lit(0).cast(pl.Int8).alias("home_promoted"),
			pl.lit(0).cast(pl.Int8).alias("away_promoted"),
		])
	
	promoted_df = pl.DataFrame(promoted_rows).lazy()
	
	# Join for home team - use is_promoted as marker
	home_promoted = (
		lf.select(
			pl.col("match_id"),
			pl.col("league").cast(pl.Utf8),
			pl.col("season").cast(pl.Utf8),
			pl.col("home_team").cast(pl.Utf8).alias("team"),
		)
		.join(
			promoted_df,
			on=["league", "season", "team"],
			how="left",
		)
		.with_columns(
			pl.col("is_promoted").fill_null(0).cast(pl.Int8).alias("home_promoted")
		)
		.select(["match_id", "home_promoted"])
	)
	
	# Join for away team
	away_promoted = (
		lf.select(
			pl.col("match_id"),
			pl.col("league").cast(pl.Utf8),
			pl.col("season").cast(pl.Utf8),
			pl.col("away_team").cast(pl.Utf8).alias("team"),
		)
		.join(
			promoted_df,
			on=["league", "season", "team"],
			how="left",
		)
		.with_columns(
			pl.col("is_promoted").fill_null(0).cast(pl.Int8).alias("away_promoted")
		)
		.select(["match_id", "away_promoted"])
	)
	
	# Join back to original
	lf = lf.join(home_promoted, on="match_id", how="left")
	lf = lf.join(away_promoted, on="match_id", how="left")
	
	return lf


def add_categorical_features(
	lf: pl.LazyFrame,
	promoted_lookup: dict[str, set[str]] | None = None,
) -> pl.LazyFrame:
	"""
	Add all categorical features to match-level data:
	1. league_idx: Numeric ID for league (for embeddings)
	2. round_number: Round/matchday number within season (from FBRef if available)
	3. season_stage: 'early', 'mid', or 'late'
	4. home_promoted: Whether home team was promoted this season
	5. away_promoted: Whether away team was promoted this season
	
	Args:
		lf: LazyFrame with match data
		promoted_lookup: Optional pre-loaded promoted teams lookup.
					   If None, will load from file.
	
	Returns: LazyFrame with categorical features added.
	"""
	# Load promoted teams if not provided
	if promoted_lookup is None:
		promoted_data = load_promoted_teams()
		promoted_lookup = build_promoted_teams_set(promoted_data)
	
	# Add league numeric ID
	lf = add_league_id_numeric(lf)
	
	# Load FBRef week data if available
	fbref_data = load_fbref_week_data()
	
	# Add round number (uses FBRef week where available)
	lf = compute_round_number(lf, fbref_data)
	
	# Add season stage based on round number
	lf = compute_season_stage(lf)
	
	# Add promoted flags
	lf = add_promoted_flags(lf, promoted_lookup)
	
	return lf


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
        safe_col("home_elo", "team_elo"),
        safe_col("away_elo", "opponent_elo"),
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
        safe_col("away_elo", "team_elo"),
        safe_col("home_elo", "opponent_elo"),
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
    # Add Elo rolling features (only r5 window)
    exprs += compute_elo_rolling_exprs()
    return long_df.with_columns(exprs)


def compute_elo_rolling_exprs() -> list:
    """
    Compute rolling Elo features for the last 5 games:
    - opponent_elo_r5: Rolling mean of opponent Elo
    - elo_diff_r5: Rolling mean of (team_elo - opponent_elo)
    - opponent_elo_std_r5: Rolling std of opponent Elo
    
    All features use shift(1) to prevent data leakage.
    Grouped by league_id, season, team.
    """
    gkeys = ["league_id", "season", "team"]
    window = 5
    exprs = []
    
    # Shifted values to prevent data leakage
    opponent_elo_shifted = pl.col("opponent_elo").shift(1)
    team_elo_shifted = pl.col("team_elo").shift(1)
    elo_diff_shifted = (team_elo_shifted - opponent_elo_shifted)
    
    # Rolling mean of opponent Elo (last 5 games)
    exprs.append(
        opponent_elo_shifted
        .rolling_mean(window_size=window, min_samples=2)
        .over(gkeys)
        .alias("opponent_elo_r5")
    )
    
    # Rolling mean of Elo difference (last 5 games)
    exprs.append(
        elo_diff_shifted
        .rolling_mean(window_size=window, min_samples=2)
        .over(gkeys)
        .alias("elo_diff_r5")
    )
    
    # Rolling std of opponent Elo (last 5 games)
    exprs.append(
        opponent_elo_shifted
        .rolling_std(window_size=window, min_samples=2)
        .over(gkeys)
        .alias("opponent_elo_std_r5")
    )
    
    return exprs


# ---------- Schedule Features (Fixture Congestion) ----------

def load_european_schedule(european_csv: Path) -> pl.LazyFrame:
    """
    Load European competition schedule from FBRef CSV and normalize team names.
    Returns a long-format LazyFrame with one row per team per match.
    Only includes teams that map to Big 5 leagues (non-null in mapping).
    """
    mapping_path = MAPPINGS_DIR / "fbref_to_canonical.json"
    with open(mapping_path, encoding="utf-8") as f:
        fbref_mapping = json.load(f)
    
    # Load CSV with UTF-8 encoding
    df = pl.scan_csv(european_csv, encoding="utf8")
    
    # Parse date
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d").cast(pl.Datetime).alias("date")
    )
    
    # Create home rows - cast game_id to string and use Utf8 for league columns
    # to match domestic data types (will be converted later for concat compatibility)
    home_rows = df.select(
        pl.col("game_id").cast(pl.Utf8).alias("match_id"),
        pl.col("league").cast(pl.Utf8).alias("league_id"),
        pl.col("league").cast(pl.Utf8),
        pl.col("season").cast(pl.Utf8),
        pl.col("date"),
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent"),
        pl.lit(True).alias("is_home"),
        pl.lit(True).alias("is_european"),
    )
    
    # Create away rows
    away_rows = df.select(
        pl.col("game_id").cast(pl.Utf8).alias("match_id"),
        pl.col("league").cast(pl.Utf8).alias("league_id"),
        pl.col("league").cast(pl.Utf8),
        pl.col("season").cast(pl.Utf8),
        pl.col("date"),
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent"),
        pl.lit(False).alias("is_home"),
        pl.lit(True).alias("is_european"),
    )
    
    # Combine and apply team mapping
    long_df = pl.concat([home_rows, away_rows])
    
    # Map team names to canonical (filter out non-Big-5 teams)
    # Create mapping expressions
    mapping_expr = pl.col("team").replace(fbref_mapping, default=None).alias("canonical_team")
    
    long_df = long_df.with_columns(mapping_expr)
    
    # Filter to only Big-5 teams (those with non-null canonical mapping)
    long_df = long_df.filter(pl.col("canonical_team").is_not_null())
    
    # Replace team with canonical name
    long_df = long_df.with_columns(pl.col("canonical_team").alias("team")).drop("canonical_team")
    
    return long_df


def merge_european_schedule(
    domestic_long: pl.LazyFrame,
    european_csv: Path | None = None,
) -> pl.LazyFrame:
    """
    Merge European competition games into domestic long-format data.
    European games are used only for schedule feature computation (days since last match,
    games in last 15 days) but are filtered out before final feature output.
    
    Args:
        domestic_long: Long-format domestic league data (from build_long)
        european_csv: Path to European schedule CSV. If None, returns domestic_long unchanged.
    
    Returns:
        Combined LazyFrame with is_european column to distinguish source.
    """
    if european_csv is None or not european_csv.exists():
        # No European data, add is_european=False column
        return domestic_long.with_columns(pl.lit(False).alias("is_european"))
    
    # Add is_european flag to domestic data
    domestic_long = domestic_long.with_columns(pl.lit(False).alias("is_european"))
    
    # Load European schedule
    european_long = load_european_schedule(european_csv)
    
    # Get domestic teams to filter European data
    # We only want European games for teams that exist in domestic data
    domestic_teams = domestic_long.select(pl.col("team").cast(pl.Utf8)).unique()
    
    # Filter European to only teams in domestic leagues
    european_long = european_long.join(domestic_teams, on="team", how="semi")
    
    # Select only common columns and cast to compatible types (all to Utf8/String)
    common_cols = ["match_id", "league_id", "league", "season", "date", "team", "opponent", "is_home", "is_european"]
    
    # Cast domestic categorical columns to Utf8 for concat
    domestic_casted = domestic_long.select([
        pl.col("match_id").cast(pl.Utf8),
        pl.col("league_id").cast(pl.Utf8),
        pl.col("league").cast(pl.Utf8),
        pl.col("season").cast(pl.Utf8),
        pl.col("date").cast(pl.Datetime("us")),  # Normalize datetime precision
        pl.col("team").cast(pl.Utf8),
        pl.col("opponent").cast(pl.Utf8),
        pl.col("is_home"),
        pl.col("is_european"),
    ])
    
    european_casted = european_long.select([
        pl.col("match_id").cast(pl.Utf8),
        pl.col("league_id").cast(pl.Utf8),
        pl.col("league").cast(pl.Utf8),
        pl.col("season").cast(pl.Utf8),
        pl.col("date").cast(pl.Datetime("us")),  # Normalize datetime precision
        pl.col("team").cast(pl.Utf8),
        pl.col("opponent").cast(pl.Utf8),
        pl.col("is_home"),
        pl.col("is_european"),
    ])
    
    # Concatenate
    combined = pl.concat([domestic_casted, european_casted])
    
    # Sort by team and date for proper schedule feature computation
    combined = combined.sort(["team", "date", "match_id"])
    
    return combined


def compute_schedule_features(long_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute schedule-based features:
    - days_since_last_match: Days since team's previous game (any competition)
    - games_last_15_days: Count of games in last 15 days (excluding current match)
    
    These features capture fixture congestion across all competitions.
    Groups by (season, team) only - not league_id - to capture cross-competition schedule.
    Uses shift(1) to prevent data leakage.
    """
    # Sort by team and date first
    long_df = long_df.sort(["season", "team", "date", "match_id"])
    
    # Group by season and team (across all competitions)
    gkeys = ["season", "team"]
    
    # Days since last match
    # shift(1) gets the previous game's date, then compute difference
    days_since = (
        pl.col("date") - pl.col("date").shift(1).over(gkeys)
    ).dt.total_days().alias("days_since_last_match")
    
    # Games in last 15 days
    # We need to count games where date is within 15 days before current match
    # Using a rolling window approach with shift to exclude current match
    # 
    # Polars rolling_count with a time-based window:
    # We'll use a different approach - count dates within 15 days before current
    # This requires a self-join or group_by_dynamic
    # 
    # Simpler approach: Use rolling count with shift
    # Count how many games in last 15 calendar days (shifted to exclude current)
    
    long_df = long_df.with_columns([
        days_since,
        # For games_last_15_days, we use group_by_dynamic
    ])
    
    # For games in last 15 days, we need a time-based rolling window
    # Polars group_by_dynamic can help, but it's complex with LazyFrame
    # Alternative: compute using a window function approach
    # 
    # We'll compute this by:
    # 1. For each row, count how many rows for same team have date within (current_date - 15 days, current_date)
    # This is tricky in pure Polars without collect
    # 
    # Simplified approach: Use rolling count based on row index as proxy
    # Or accept that we compute it after collect
    # 
    # For now, let's use a shifted rolling count approximation
    # Games in last N matches can proxy fixture congestion
    # But user wants last 15 DAYS specifically
    # 
    # Let's use a self-join approach in a helper
    
    # Collect to compute games_last_15_days (requires time-based logic)
    # We'll do this as a post-processing step
    
    return long_df


def compute_games_last_15_days(long_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute games_last_15_days feature using a time-based approach.
    Must be called on a collected DataFrame (not LazyFrame).
    
    For each game, counts how many games the team played in the 15 days before.
    Uses shift logic to exclude the current game.
    """
    from datetime import timedelta
    
    # Sort first
    long_df = long_df.sort(["season", "team", "date", "match_id"])
    
    # Add row number per team-season for efficient processing
    long_df = long_df.with_columns(
        pl.col("date").cum_count().over(["season", "team"]).alias("game_num")
    )
    
    # For each row, count games where:
    # - Same team and season
    # - Date is within (current_date - 15 days, current_date) - exclusive of current
    # 
    # We'll do this with a self-join
    
    result_rows = []
    
    # Process by team-season groups
    for (season, team), group in long_df.group_by(["season", "team"]):
        group = group.sort("date")
        dates = group["date"].to_list()
        counts = []
        
        for i, current_date in enumerate(dates):
            if current_date is None:
                counts.append(None)
                continue
            
            # Count games in last 15 days (excluding current)
            cutoff = current_date - timedelta(days=15)
            count = 0
            for j in range(i):  # Only look at previous games
                prev_date = dates[j]
                if prev_date is not None and prev_date >= cutoff:
                    count += 1
            counts.append(count)
        
        group = group.with_columns(pl.Series("games_last_15_days", counts))
        result_rows.append(group)
    
    if not result_rows:
        return long_df.with_columns(pl.lit(None).cast(pl.Int64).alias("games_last_15_days"))
    
    return pl.concat(result_rows).drop("game_num")


def with_side_suffix(lf: pl.LazyFrame, suffix: str) -> pl.LazyFrame:
    """
    Add a side suffix (e.g., '__h' or '__a') to all engineered feature columns in this frame.
    Includes rolling features, schedule features, and Elo rolling features.
    """
    names = lf.collect_schema().names()
    feat_cols = [c for c in names if c.startswith(("ovr__", "home__", "away__"))]
    # Also rename schedule features
    schedule_cols = [c for c in names if c in ("days_since_last_match", "games_last_15_days")]
    # Also rename Elo rolling features
    elo_feat_cols = [c for c in names if c in ("opponent_elo_r5", "elo_diff_r5", "opponent_elo_std_r5")]
    all_feat_cols = feat_cols + schedule_cols + elo_feat_cols
    rename_map = {c: f"{c}{suffix}" for c in all_feat_cols}
    return lf.rename(rename_map)


def build_match_level(
    base_matches: pl.LazyFrame, long_with_feats: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Join features back to match level.
    We tag home features with '__h' and away with '__a' BEFORE the join to avoid collisions.
    Adds target 'Over'.
    Filters out European-only matches (keeps only domestic league games).
    """
    # Minimal feature frames per side
    feat_cols = [
        c
        for c in long_with_feats.collect_schema().names()
        if c.startswith(("ovr__", "home__", "away__")) or c in ("days_since_last_match", "games_last_15_days", "opponent_elo_r5", "elo_diff_r5", "opponent_elo_std_r5")
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
