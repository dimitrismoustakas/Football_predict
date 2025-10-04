# understat_dump.py
# Usage:
#   python understat_dump.py --root data/understat
#   python understat_dump.py --root data/understat --leagues EPL La_liga --seasons 2019 2020
#
# It will save:
#   data/understat/<league>/<season>/shots.parquet
#   data/understat/<league>/<season>/matches.parquet
#   data/understat/<league>/<season>/rosters.parquet
import argparse
import os
import re
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ScraperFC as sfc
from tqdm import tqdm
import time


LEAGUES_DEFAULT = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "RFPL"]

def sanitize_league(league: str) -> str:
    return league.replace(" ", "_")

def sanitize_season(season: str) -> str:
    return re.sub(r"[\\/]", "-", season)

def match_id_from_link(link: str) -> str:
    # Typical understat match links end with /match/<id>
    m = re.search(r"/match/(\d+)", link)
    return m.group(1) if m else re.sub(r"\W+", "_", link)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def coerce_df(x) -> pd.DataFrame:
    """Be permissive: convert dict/list/None to DataFrame safely."""
    if isinstance(x, pd.DataFrame):
        return x
    if x is None:
        return pd.DataFrame()
    if isinstance(x, dict):
        # If dict of columns or dict of records
        try:
            return pd.DataFrame.from_dict(x)
        except Exception:
            return pd.DataFrame([x])
    if isinstance(x, (list, tuple)):
        return pd.DataFrame(list(x))
    # Fallback
    return pd.DataFrame()

def normalize_match_value(val: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    The docs say scrape_match returns (shots_data, match_info, rosters_data).
    scrape_matches returns a dict per link. Support both.
    """
    if isinstance(val, dict):
        shots = coerce_df(val.get("shots_data"))
        match_info = coerce_df(val.get("match_info"))
        rosters = coerce_df(val.get("rosters_data"))
    elif isinstance(val, (list, tuple)) and len(val) == 3:
        shots, match_info, rosters = map(coerce_df, val)
    else:
        # Unknown shape; try best-effort
        shots = coerce_df(val)
        match_info = pd.DataFrame()
        rosters = pd.DataFrame()
    return shots, match_info, rosters

def add_partition_cols(df: pd.DataFrame, league: str, season: str, link: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["league"] = league
    df["season"] = season
    df["match_link"] = link
    df["match_id"] = match_id_from_link(link)
    return df

def concat_and_sort(dfs) -> pd.DataFrame:
    dfs = [d for d in dfs if d is not None and not d.empty]
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # Try to sort by a sensible time column if present
    for col in ["datetime", "date", "kickoff_time", "match_date", "time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce", utc=True).dt.date
            out = out.sort_values(col, kind="mergesort")
            break
    return out

def write_parquet(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        # Still write an empty file with schema for consistency
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path)
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)

def scrape_league_season(us: sfc.Understat, league: str, season: str, out_root: Path, sleep_s: float = 1.0) -> None:
    league_norm = sanitize_league(league)
    season_norm = sanitize_season(season)
    out_dir = out_root / league_norm / season_norm
    ensure_dir(out_dir)

    # Pull all matches for league-season as DataFrames
    # If rate-limited/fragile network, retry once.
    for attempt in range(2):
        try:
            matches: Dict[str, Any] = us.scrape_matches(year=season, league=league, as_df=True)
            break
        except Exception as e:
            if attempt == 1:
                raise
            time.sleep(3)

    shots_rows, info_rows, roster_rows = [], [], []

    for link, match_val in tqdm(matches.items(), desc=f"{league} {season}"):
        try:
            shots, match_info, rosters = normalize_match_value(match_val)
            shots = add_partition_cols(shots, league, season_norm, link)
            match_info = add_partition_cols(match_info, league, season_norm, link)
            rosters = add_partition_cols(rosters, league, season_norm, link)

            shots_rows.append(shots)
            info_rows.append(match_info)
            roster_rows.append(rosters)

        except Exception:
            # Keep going; you can inspect logs later
            continue

        # Gentle pacing
        time.sleep(sleep_s)

    shots_all = concat_and_sort(shots_rows)
    matches_all = concat_and_sort(info_rows)
    rosters_all = concat_and_sort(roster_rows)

    write_parquet(shots_all, out_dir / "shots.parquet")
    write_parquet(matches_all, out_dir / "matches.parquet")
    write_parquet(rosters_all, out_dir / "rosters.parquet")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/understat")
    parser.add_argument("--leagues", nargs="*", default=LEAGUES_DEFAULT)
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Optional season filter. If omitted, will use Understat.get_valid_seasons per league.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep between matches to be gentle.")
    args = parser.parse_args()

    out_root = Path(args.root)
    ensure_dir(out_root)

    us = sfc.Understat()

    for league in args.leagues:
        if args.seasons:
            seasons = args.seasons
        else:
            seasons = us.get_valid_seasons(league=league)

        for season in seasons:
            try:
                scrape_league_season(us, league, season, out_root, sleep_s=args.sleep)
            except Exception as e:
                # Log and continue to next season/league
                print(f"[WARN] {league} {season} failed: {e}")

if __name__ == "__main__":
    main()
