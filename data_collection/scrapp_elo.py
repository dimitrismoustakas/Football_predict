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
LEAGUES_TO_KEEP = {"GER", "ESP", "ENG", "ITA", "FRA", "RUS"}

DATA_DIR = Path("data/eloscores")
SNAPSHOT_DIR = DATA_DIR / "snapshots"
TEAM_UNIVERSE_PATH = DATA_DIR / "team_universe.parquet"
ELO_HISTORY_PATH = DATA_DIR / "elo_history.parquet"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

club_elo = ClubElo()

# --------------------------
# 1) Snapshots and team universe
# --------------------------
snapshots = []
current_date = START_DATE
while current_date <= END_DATE:
    df = club_elo.read_by_date(current_date).reset_index()  # ensure 'team' column exists
    df = df.assign(snapshot_date=pd.to_datetime(current_date))
    if "country" in df.columns:
        df = df[df["country"].isin(LEAGUES_TO_KEEP)]
    # persist each snapshot for auditing
    (SNAPSHOT_DIR / f"clubelo_snapshot_{current_date.date()}.parquet").write_bytes(
        df.to_parquet(index=False)
        if hasattr(pd.DataFrame, "to_parquet")  # just to be explicit
        else b""
    )
    snapshots.append(df)
    current_date += timedelta(days=STEP_DAYS)

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

# --------------------------
# 2) Pull each team's full Elo history
# --------------------------
def fetch_team_history(team, retries=3, backoff=1.0):
    for attempt in range(1, retries + 1):
        try:
            return club_elo.read_team_history(team, max_age=1)  # use local cache when possible
        except Exception:
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)

histories = []
for team in tqdm(team_universe["team"].tolist(), desc="Fetching team Elo histories"):
    hist = fetch_team_history(team)
    # Attach country if missing
    if "country" not in hist.columns and "country" in team_universe.columns:
        c = team_universe.loc[team_universe["team"] == team, "country"]
        if not c.empty:
            hist = hist.assign(country=c.iloc[0])
    histories.append(hist)

elo_history = pd.concat(histories, ignore_index=True)

# Clean and filter
for col in ("from", "to"):
    if col in elo_history.columns:
        elo_history[col] = pd.to_datetime(elo_history[col], errors="coerce")
if "elo" in elo_history.columns:
    elo_history["elo"] = pd.to_numeric(elo_history["elo"], errors="coerce")

cutoff = pd.to_datetime(START_DATE)
if {"from", "to"}.issubset(elo_history.columns):
    elo_history = elo_history[(elo_history["from"] >= cutoff) | (elo_history["to"] >= cutoff)]

# Keep interval schema for efficient “elo before date” later
keep = [c for c in ["team", "country", "from", "to", "elo"] if c in elo_history.columns]
elo_history = elo_history[keep].reset_index(drop=True)

# --------------------------
# 3) Persist
# --------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
elo_history.to_parquet(ELO_HISTORY_PATH, index=False)
print(f"Saved team universe -> {TEAM_UNIVERSE_PATH}")
print(f"Saved Elo history  -> {ELO_HISTORY_PATH}")
print(f"Saved snapshots    -> {SNAPSHOT_DIR}")
