# pip install soccerdata python-dateutil pandas pytz
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import soccerdata as sd

TZ = ZoneInfo("Europe/Athens")

def get_current_season_for_league(league_id: str) -> str:
    """Pick the most recent season string for a FotMob league."""
    tmp = sd.FotMob(leagues=league_id)
    seasons_df = tmp.read_seasons()  # columns typically include 'season' and maybe others
    # Be robust to column naming
    season_col = "season" if "season" in seasons_df.columns else seasons_df.columns[0]
    # Sort season strings "2024-25" correctly by extracting the last year-ish
    seasons_df = seasons_df.copy()
    seasons_df["_sortkey"] = seasons_df[season_col].astype(str).str.extract(r"(\d{4})").astype(int)
    current = seasons_df.sort_values("_sortkey").iloc[-1][season_col]
    return str(current)

def collect_league_slice(league_id: str, season: str, now: datetime) -> pd.DataFrame:
    """Return past-to-now matches in the season, plus today/tomorrow fixtures."""
    reader = sd.FotMob(leagues=league_id, seasons=season)
    sched = reader.read_schedule()

    # Expect a datetime or date column; make it timezone-aware.
    # FotMob schedule usually has 'kickoff' or 'date' fields; normalize:
    dt_col = "kickoff" if "kickoff" in sched.columns else ("date" if "date" in sched.columns else None)
    if dt_col is None:
        # Fallback: try to combine date/time columns if present
        raise RuntimeError(f"Couldn't find a datetime column for league {league_id}")

    sched = sched.copy()
    sched["kick_dt"] = pd.to_datetime(sched[dt_col], utc=True, errors="coerce").dt.tz_convert(TZ)
    # If tz-naive, localize:
    na_mask = sched["kick_dt"].isna() & pd.to_datetime(sched[dt_col], errors="coerce").notna()
    if na_mask.any():
        sched.loc[na_mask, "kick_dt"] = pd.to_datetime(sched.loc[na_mask, dt_col], errors="coerce").dt.tz_localize(TZ)

    today = now.date()
    tomorrow = (now + timedelta(days=1)).date()

    played_until_now = sched[sched["kick_dt"] <= now]
    today_tomorrow = sched[sched["kick_dt"].dt.date.isin([today, tomorrow])]

    out = pd.concat([played_until_now, today_tomorrow], ignore_index=True).drop_duplicates()
    out.insert(0, "league_id", league_id)
    out.insert(1, "season", season)
    return out

def main():
    now = datetime.now(TZ)

    # 1) List supported leagues
    supported_leagues = sd.FotMob.available_leagues()  # list[str]
    # Optional: restrict to a subset if you only want, say, the “major” leagues.

    # 2) For each league, pick current season and pull schedules
    frames = []
    for lg in supported_leagues:
        try:
            season = get_current_season_for_league(lg)
            df = collect_league_slice(lg, season, now)
            frames.append(df)
        except Exception as e:
            # Some leagues may fail or not expose fixtures consistently; skip cleanly
            print(f"[WARN] {lg}: {e}")
            continue

    if not frames:
        print("No data gathered.")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Keep useful columns if present:
    keep_cols = [c for c in [
        "league_id", "season", "kick_dt", "home_team", "away_team",
        "home", "away", "round", "status", "score", "venue", "match_id"
    ] if c in all_df.columns]
    all_df = all_df[keep_cols].sort_values(["kick_dt", "league_id", "season"]).reset_index(drop=True)

    # Save two outputs:
    # 1) Past-to-now for current seasons (by league)
    past_to_now = all_df[all_df["kick_dt"] <= now]
    past_to_now.to_csv("current_season_played_until_now.csv", index=False)

    # 2) Today + tomorrow fixtures across all supported leagues
    dmask = all_df["kick_dt"].dt.date.isin([now.date(), (now + timedelta(days=1)).date()])
    fixtures = all_df[dmask]
    fixtures.to_csv("fixtures_today_tomorrow.csv", index=False)

    print("Wrote current_season_played_until_now.csv and fixtures_today_tomorrow.csv")

if __name__ == "__main__":
    main()
