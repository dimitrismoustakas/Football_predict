import warnings
from datetime import datetime, timedelta

import pandas as pd
from soccerdata import ClubElo

warnings.simplefilter(action="ignore", category=FutureWarning)
club_elo = ClubElo()

start_date = datetime(2014, 1, 1)
end_date = datetime.today()

elo_frames = []

current_date = start_date
count = 0
while current_date <= end_date:
    elo_scores = club_elo.read_by_date(current_date)
    elo_frames.append(elo_scores)
    current_date += timedelta(days=120)
    count += 1
    print(count)

elo_df = pd.concat(elo_frames)

elo_df = elo_df.reset_index()
leagues_to_keep = ["GER", "ESP", "ENG", "ITA", "FRA", "RUS"]

elo_df_filtered = elo_df[elo_df["country"].isin(leagues_to_keep)]

unique_teams = elo_df_filtered["team"].unique()

team_elo_frames = []
for team in unique_teams:
    team_elo = club_elo.read_team_history(team)
    team_elo_frames.append(team_elo)

all_teams_elo_history = pd.concat(team_elo_frames)

all_teams_elo_history = all_teams_elo_history.reset_index()
all_teams_elo_history.head(5)

date_cutoff = datetime(2014, 1, 1)
filtered_elo_df = all_teams_elo_history[
    (all_teams_elo_history["from"] >= date_cutoff)
    | (all_teams_elo_history["to"] >= date_cutoff)
]
filtered_elo_df = filtered_elo_df[["team", "from", "to", "elo"]]
filtered_elo_df.to_csv("Elo_Scores.csv", sep=",")
