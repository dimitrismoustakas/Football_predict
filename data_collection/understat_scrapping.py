# understat_dump.py
# Usage examples:
#   python understat_dump.py --root data/understat
#   python understat_dump.py --root data/understat --leagues EPL La_Liga --seasons 2019 2020
#   python understat_dump.py --root data/understat --leagues "La Liga" --seasons 2017 --only-failed
#
# It writes per-match shards:
#   data/understat/<league>/<season>/{shots,matches,rosters}/<match_id>.parquet
# and consolidates to:
#   data/understat/<league>/<season>/shots.parquet
#   data/understat/<league>/<season>/matches.parquet
#   data/understat/<league>/<season>/rosters.parquet

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ScraperFC as sfc
from tqdm import tqdm

# Transient network exceptions to catch
try:
    from http.client import RemoteDisconnected
except Exception:
    class RemoteDisconnected(Exception): ...
try:
    from urllib3.exceptions import ProtocolError
except Exception:
    class ProtocolError(Exception): ...
try:
    from requests.exceptions import ConnectionError as RequestsConnectionError
except Exception:
    class RequestsConnectionError(Exception): ...

LEAGUES_DEFAULT = ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "RFPL"]
PER_MATCH_DIRS = ["shots", "matches", "rosters"]


# --------------------- helpers ---------------------
def sanitize_league(league: str) -> str:
    return league.replace(" ", "_")


def sanitize_season(season: str) -> str:
    return re.sub(r"[\\/]", "-", str(season))


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
        try:
            return pd.DataFrame.from_dict(x)
        except Exception:
            return pd.DataFrame([x])
    if isinstance(x, (list, tuple)):
        return pd.DataFrame(list(x))
    return pd.DataFrame()


def normalize_match_value(val: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ScraperFC.Understat.scrape_matches returns a dict:
      link -> {"shots_data": df|list|dict, "match_info": df|..., "rosters_data": df|...}
    Also support a 3-tuple (shots, match_info, rosters).
    """
    if isinstance(val, dict):
        shots = coerce_df(val.get("shots_data"))
        match_info = coerce_df(val.get("match_info"))
        rosters = coerce_df(val.get("rosters_data"))
    elif isinstance(val, (list, tuple)) and len(val) == 3:
        shots, match_info, rosters = map(coerce_df, val)
    else:
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


def concat_and_sort(dfs: Iterable[pd.DataFrame]) -> pd.DataFrame:
    dfs = [d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty]
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    # Try to sort by a sensible time-like column if present
    # for col in ["datetime", "date", "kickoff_time", "match_date", "time"]:
    #     if col in out.columns:
    #         # out[col] = pd.to_datetime(out[col], format="%Y-%m-%d", errors="coerce", utc=True).dt.date
    #         out = out.sort_values(col, kind="mergesort")
    #         break
    return out


def retry(fn, max_tries=5, base_sleep=1.0, exceptions=(Exception,), ctx=""):
    """Retry helper with exponential backoff and light jitter."""
    last_exc = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            if attempt == max_tries:
                break
            sleep = base_sleep * (2 ** (attempt - 1)) * (1 + 0.12 * (attempt % 3))
            print(f"[INFO] Retry {attempt}/{max_tries} for {ctx or fn.__name__}: {e}. Sleeping {sleep:.1f}s")
            time.sleep(sleep)
    raise last_exc


def save_failed(out_dir: Path, league: str, season: str, link: str, err: Exception) -> None:
    rec = {
        "league": league,
        "season": season,
        "match_link": link,
        "match_id": match_id_from_link(link),
        "error": repr(err),
        "ts": time.time(),
    }
    with open(out_dir / "failed_matches.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_failed_links(out_dir: Path) -> List[str]:
    fp = out_dir / "failed_matches.jsonl"
    if not fp.exists():
        return []
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out.append(obj.get("match_link") or "")
            except Exception:
                continue
    # de-dup while preserving order
    seen = set()
    uniq = []
    for l in out:
        if l and l not in seen:
            uniq.append(l)
            seen.add(l)
    return uniq


def rewrite_failed_excluding(out_dir: Path, exclude_match_ids: Iterable[str]) -> None:
    exclude = set(map(str, exclude_match_ids))
    src = out_dir / "failed_matches.jsonl"
    if not src.exists():
        return
    keep: List[str] = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                mid = str(obj.get("match_id", ""))
                if mid and mid in exclude:
                    continue
            except Exception:
                pass
            keep.append(line)
    tmp = out_dir / "failed_matches.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(keep)
    tmp.replace(src)


def per_match_path(out_dir: Path, kind: str, match_id: str) -> Path:
    assert kind in PER_MATCH_DIRS
    return out_dir / kind / f"{match_id}.parquet"


def match_already_done(out_dir: Path, match_id: str) -> bool:
    # consider done if any of the three files exists and is non-empty
    for k in PER_MATCH_DIRS:
        fp = per_match_path(out_dir, k, match_id)
        if fp.exists() and fp.stat().st_size > 0:
            return True
    return False


def write_single(out_dir: Path, kind: str, df: pd.DataFrame, match_id: str):
    ensure_dir(out_dir / kind)
    fp = per_match_path(out_dir, kind, match_id)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, fp)


def consolidate_dir_to_file(out_dir: Path, kind: str, season_file: str):
    shard_dir = out_dir / kind
    files = sorted([p for p in shard_dir.glob("*.parquet") if p.is_file()])
    if not files:
        # still write empty parquet with schema-less table
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(), preserve_index=False), out_dir / season_file)
        return
    # streaming concat to limit memory
    dfs: List[pd.DataFrame] = []
    batch = 0
    for fp in files:
        try:
            dfs.append(pd.read_parquet(fp))
        except Exception:
            continue
        if len(dfs) >= 64:
            batch_df = concat_and_sort(dfs)
            dfs = [batch_df]
            batch += 1
    final_df = concat_and_sort(dfs)
    pq.write_table(pa.Table.from_pandas(final_df, preserve_index=False), out_dir / season_file)


def consolidate_season(out_dir: Path):
    consolidate_dir_to_file(out_dir, "shots", "shots.parquet")
    consolidate_dir_to_file(out_dir, "matches", "matches.parquet")
    consolidate_dir_to_file(out_dir, "rosters", "rosters.parquet")


# --------------------- scraping core ---------------------
def fetch_all_matches(us: sfc.Understat, league: str, season: str) -> Dict[str, Any]:
    """Fetch whole season (mapping link -> {shots_data, match_info, rosters_data})."""
    def _f():
        return us.scrape_matches(year=season, league=league, as_df=True)
    return retry(
        _f,
        max_tries=6,
        base_sleep=1.0,
        exceptions=(RemoteDisconnected, ProtocolError, RequestsConnectionError, ConnectionError, TimeoutError, Exception),
        ctx=f"{league} {season} scrape_matches",
    )


def fetch_single_match(us: sfc.Understat, link: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch one match by link using Understat.scrape_match, normalized to (shots, match_info, rosters)."""
    def _f():
        # Prefer the dedicated single-match scraper if available
        if hasattr(us, "scrape_match"):
            val = us.scrape_match(link=link, as_df=True)
        else:
            # Fallback: scrape generic URL (ScraperFC supports this)
            val = us.scrape(link=link, as_df=True)
        return normalize_match_value(val)
    return retry(
        _f,
        max_tries=5,
        base_sleep=0.8,
        exceptions=(RemoteDisconnected, ProtocolError, RequestsConnectionError, ConnectionError, TimeoutError, Exception),
        ctx=f"match {match_id_from_link(link)}",
    )


def process_one(
    out_dir: Path,
    league: str,
    season: str,
    link: str,
    value_from_bulk: Optional[Any],
) -> bool:
    """
    Process a single match link.
    If value_from_bulk is provided (from scrape_matches), use it; else fetch single.
    Returns True if a new write occurred (or already done), False if failed this time.
    """
    mid = match_id_from_link(link)
    if match_already_done(out_dir, mid):
        return True

    try:
        if value_from_bulk is None:
            shots, match_info, rosters = fetch_single_match(sfc.Understat(), link)
        else:
            shots, match_info, rosters = normalize_match_value(value_from_bulk)

        shots = add_partition_cols(shots, league, season, link)
        match_info = add_partition_cols(match_info, league, season, link)
        rosters = add_partition_cols(rosters, league, season, link)

        # It’s valid for a match to have 0 shots; still write empty dfs so resume knows it's done
        write_single(out_dir, "shots", shots, mid)
        write_single(out_dir, "matches", match_info, mid)
        write_single(out_dir, "rosters", rosters, mid)
        return True

    except Exception as e:
        save_failed(out_dir, league, season, link, e)
        return False


def scrape_league_season(
    us: sfc.Understat,
    league: str,
    season: str,
    out_root: Path,
    sleep_s: float = 0.8,
    only_failed: bool = False,
) -> None:
    league_norm = sanitize_league(league)
    season_norm = sanitize_season(season)
    out_dir = out_root / league_norm / season_norm
    ensure_dir(out_dir)
    for sub in PER_MATCH_DIRS:
        ensure_dir(out_dir / sub)

    if only_failed:
        failed_links = read_failed_links(out_dir)
        if not failed_links:
            print(f"[INFO] No failed_matches.jsonl entries for {league} {season}. Nothing to do.")
            # still consolidate whatever exists
            consolidate_season(out_dir)
            return
        print(f"[INFO] Retrying {len(failed_links)} failed matches for {league} {season}")
        completed_now: List[str] = []
        for link in tqdm(failed_links, desc=f"{league} {season} (only-failed)"):
            mid = match_id_from_link(link)
            ok = process_one(out_dir, league, season_norm, link, value_from_bulk=None)
            if ok:
                completed_now.append(mid)
            time.sleep(sleep_s)
        # remove successes from failed list
        if completed_now:
            rewrite_failed_excluding(out_dir, completed_now)
        consolidate_season(out_dir)
        return

    # Normal bulk run
    try:
        matches: Dict[str, Any] = fetch_all_matches(us, league, season)
    except Exception as e:
        print(f"[WARN] {league} {season} failed to fetch matches: {e}")
        # If bulk fails, try to at least retry previously failed links
        failed_links = read_failed_links(out_dir)
        if failed_links:
            print(f"[INFO] Bulk failed. Retrying {len(failed_links)} previously failed matches individually...")
            completed_now: List[str] = []
            for link in tqdm(failed_links, desc=f"{league} {season} (recover-failed)"):
                mid = match_id_from_link(link)
                ok = process_one(out_dir, league, season_norm, link, value_from_bulk=None)
                if ok:
                    completed_now.append(mid)
                time.sleep(sleep_s)
            if completed_now:
                rewrite_failed_excluding(out_dir, completed_now)
        consolidate_season(out_dir)
        return

    # Iterate all matches
    completed_now: List[str] = []
    for link, match_val in tqdm(matches.items(), desc=f"{league} {season}"):
        mid = match_id_from_link(link)
        ok = process_one(out_dir, league, season_norm, link, value_from_bulk=match_val)
        if ok:
            completed_now.append(mid)
        time.sleep(sleep_s)

    if completed_now:
        # If any of these were previously logged as failures and now succeeded, drop them from the log
        rewrite_failed_excluding(out_dir, completed_now)

    consolidate_season(out_dir)


# --------------------- CLI ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data/understat")
    parser.add_argument("--leagues", nargs="*", default=LEAGUES_DEFAULT)
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Optional season filter. If omitted, uses Understat.get_valid_seasons(league).")
    parser.add_argument("--sleep", type=float, default=0.8, help="Sleep between matches to be gentle.")
    parser.add_argument("--only-failed", action="store_true",
                        help="Retry only matches listed in failed_matches.jsonl for each league/season.")
    args = parser.parse_args()

    out_root = Path(args.root)
    ensure_dir(out_root)

    us = sfc.Understat()

    for league in args.leagues:
        seasons = args.seasons or us.get_valid_seasons(league=league)
        for season in seasons:
            try:
                scrape_league_season(
                    us, league, season, out_root,
                    sleep_s=args.sleep,
                    only_failed=args.only_failed,
                )
            except Exception as e:
                print(f"[WARN] {league} {season} failed: {e}")


if __name__ == "__main__":
    main()
