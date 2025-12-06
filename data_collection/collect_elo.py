# -*- coding: utf-8 -*-
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import time
import pandas as pd
from tqdm import tqdm
from soccerdata import ClubElo

warnings.simplefilter(action="ignore", category=FutureWarning)

# --------------------------
# Config
# --------------------------
START_DATE = datetime(2014, 1, 1)
END_DATE = datetime.today()
STEP_DAYS = 120
LEAGUES_TO_KEEP = {"GER", "ESP", "ENG", "ITA", "FRA"}

DATA_DIR = Path("data/eloscores")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
TEAM_UNIVERSE_PATH = DATA_DIR / "team_universe.parquet"
ELO_HISTORY_PATH = DATA_DIR / "elo_history.parquet"

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    club_elo = ClubElo()

    # --------------------------
    # 1) Snapshots and team universe
    # --------------------------
    snapshots = []
    current_date = START_DATE
    print("Collecting snapshots to build team universe...")
    while current_date <= END_DATE:
        try:
            df = club_elo.read_by_date(current_date).reset_index()
            df = df.assign(snapshot_date=pd.to_datetime(current_date))
            if "country" in df.columns:
                df = df[df["country"].isin(LEAGUES_TO_KEEP)]
            
            # Save snapshot
            snapshot_path = SNAPSHOT_DIR / f"clubelo_snapshot_{current_date.date()}.parquet"
            df.to_parquet(snapshot_path, index=False)
            
            snapshots.append(df)
        except Exception as e:
            print(f"Error fetching snapshot for {current_date.date()}: {e}")
        
        current_date += timedelta(days=STEP_DAYS)

    if not snapshots:
        print("No snapshots collected.")
        return

    snapshots_df = pd.concat(snapshots, ignore_index=True)

    # de-duplicated team universe
    cols = ["team"] + (["country"] if "country" in snapshots_df.columns else [])
    team_universe = (
        snapshots_df[cols]
        .drop_duplicates(subset=["team"])
        .sort_values("team")
        .reset_index(drop=True)
    )
    team_universe.to_parquet(TEAM_UNIVERSE_PATH, index=False)
    print(f"Saved team universe with {len(team_universe)} teams to {TEAM_UNIVERSE_PATH}")

    # --------------------------
    # 2) Pull each team's full Elo history
    # --------------------------
    def fetch_team_history(team, retries=3, backoff=1.0):
        for attempt in range(1, retries + 1):
            try:
                return club_elo.read_team_history(team, max_age=1)
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(backoff * attempt)

    histories = []
    teams_list = team_universe["team"].tolist()
    
    print("Fetching full history for each team...")
    for team in tqdm(teams_list, desc="Fetching team Elo histories"):
        try:
            hist = fetch_team_history(team)
            # Attach country if missing
            if "country" not in hist.columns and "country" in team_universe.columns:
                c = team_universe.loc[team_universe["team"] == team, "country"]
                if not c.empty:
                    hist = hist.assign(country=c.iloc[0])
            histories.append(hist)
        except Exception as e:
            print(f"Failed to fetch history for {team}: {e}")

    if not histories:
        print("No histories collected.")
        return

    elo_history = pd.concat(histories, ignore_index=True)

    # Rename 'to' to 'from' because it represents the date of the update (valid from this date)
    if "to" in elo_history.columns and "from" not in elo_history.columns:
        elo_history = elo_history.rename(columns={"to": "from"})

    # Clean and filter
    for col in ("from", "to"):
        if col in elo_history.columns:
            elo_history[col] = pd.to_datetime(elo_history[col], errors="coerce")
    if "elo" in elo_history.columns:
        elo_history["elo"] = pd.to_numeric(elo_history["elo"], errors="coerce")

    # Sort to ensure correct interval calculation
    elo_history = elo_history.sort_values(["team", "from"])

    # Generate 'to' column if missing
    if "to" not in elo_history.columns:
        # 'to' is the day before the next 'from'
        elo_history["to"] = elo_history.groupby("team")["from"].shift(-1) - pd.Timedelta(days=1)
        # Fill the last 'to' with today (or a future date)
        elo_history["to"] = elo_history["to"].fillna(pd.Timestamp.today())

    cutoff = pd.to_datetime(START_DATE)
    if {"from", "to"}.issubset(elo_history.columns):
        elo_history = elo_history[(elo_history["from"] >= cutoff) | (elo_history["to"] >= cutoff)]

    # Keep interval schema for efficient “elo before date” later
    keep = [c for c in ["team", "country", "from", "to", "elo"] if c in elo_history.columns]
    elo_history = elo_history[keep].reset_index(drop=True)

    # --------------------------
    # 3) Persist
    # --------------------------
    elo_history.to_parquet(ELO_HISTORY_PATH, index=False)
    print(f"Saved Elo history to {ELO_HISTORY_PATH}")

if __name__ == "__main__":
    main()
