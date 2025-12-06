# upcoming_games.py
from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Tuple
from zoneinfo import ZoneInfo
import pandas as pd
import soccerdata as sd

ATHENS_TZ = ZoneInfo("Europe/Athens")
UTC = ZoneInfo("UTC")


def _latest_season_for_league(league_id: str) -> str:
    """
    Query Understat seasons for a league and return the latest as a 'YYYY/YYYY+1' string
    when available, otherwise the raw season label Understat returns.
    soccerdata.Understat.read_seasons() returns a DataFrame indexed by (league, season) in most versions.
    """
    tmp = sd.Understat(leagues=league_id)
    seasons_df = tmp.read_seasons()

    # Bring index to columns robustly
    if isinstance(seasons_df.index, pd.MultiIndex):
        seasons_df = seasons_df.reset_index()
    else:
        seasons_df = seasons_df.reset_index(drop=False)

    # Understat commonly encodes seasons as strings like "2015" or "2015/2016" depending on adapter
    if "season" not in seasons_df.columns:
        # fallback: try to find a season-like column
        for c in seasons_df.columns:
            if "season" in str(c).lower():
                seasons_df = seasons_df.rename(columns={c: "season"})
                break
    if "season" not in seasons_df.columns:
        # final fallback: cannot detect -> let soccerdata auto-select by not passing season
        return None  # type: ignore

    # Normalize to a sortable integer "start year" when possible
    def _start_year(s):
        try:
            if isinstance(s, str) and "/" in s:
                return int(s.split("/")[0])
            return int(str(s))
        except Exception:
            return -1

    seasons_df["__start_year"] = seasons_df["season"].map(_start_year)
    seasons_df = seasons_df.sort_values(["__start_year", "season"])
    latest = seasons_df.iloc[-1]["season"]

    # Return as-is; soccerdata accepts both "2019" and "2019/2020" depending on backend
    return str(latest)


def fetch_current_season_schedule(leagues: List[str]) -> pd.DataFrame:
    frames = []
    for lg in leagues:
        # season = _latest_season_for_league(lg)
        reader = sd.Understat(leagues=lg, seasons="2025/2026")# if season else sd.Understat(leagues=lg)
        sch = reader.read_schedule(include_matches_without_data=True)
        sch = sch.reset_index()  # ["league","season","game", ...] back to columns
        frames.append(sch)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def localize_and_split(schedule: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if schedule.empty:
        return schedule, schedule

    if isinstance(schedule.index, pd.MultiIndex) or (schedule.index.names and any(n is not None for n in schedule.index.names)):
        schedule = schedule.reset_index()

    # Ensure datetime dtype
    if "date" not in schedule.columns:
        raise ValueError("Schedule missing 'date' column from Understat.")
    if not pd.api.types.is_datetime64_any_dtype(schedule["date"]):
        schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce")

    # Treat source times as UTC, then convert to Athens
    schedule["kickoff_utc"] = schedule["date"].dt.tz_localize(UTC, nonexistent="shift_forward", ambiguous="NaT")
    schedule["kickoff_local"] = schedule["kickoff_utc"].dt.tz_convert(ATHENS_TZ)

    now_local = datetime.now(ATHENS_TZ)
    today_local = now_local.date()
    tomorrow_local = (now_local + timedelta(days=4)).date()

    # Played so far = kickoff <= now
    played_so_far = schedule[schedule["kickoff_local"] <= now_local].copy()

    # Upcoming today & tomorrow (exclude ones already finished)
    day = schedule["kickoff_local"].dt.date
    today_tomorrow = schedule[(day.isin([today_local, tomorrow_local])) & (schedule["kickoff_local"] >= now_local)].copy()

    sort_cols = [c for c in ["kickoff_local", "league", "home_team", "away_team"] if c in schedule.columns]
    if sort_cols:
        played_so_far = played_so_far.sort_values(sort_cols, kind="mergesort")
        today_tomorrow = today_tomorrow.sort_values(sort_cols, kind="mergesort")

    return played_so_far, today_tomorrow


def main():
    leagues = sd.Understat.available_leagues()
    schedule = fetch_current_season_schedule(leagues)
    played_so_far, today_tomorrow = localize_and_split(schedule)

    keep_cols = [c for c in [
        "league","season","game","kickoff_local","home_team","away_team",
        "home_goals","away_goals","home_xg","away_xg","is_result","url"
    ] if c in today_tomorrow.columns]

    print("\n=== TODAY & TOMORROW (LOCAL Europe/Athens) — UPCOMING ONLY ===")
    if not today_tomorrow.empty:
        print(today_tomorrow[keep_cols].to_string(index=False))
    else:
        print("(no upcoming fixtures for today/tomorrow)")

if __name__ == "__main__":
    a=0
    main()
