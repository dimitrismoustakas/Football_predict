# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:38:01 2023

@author: mouts
"""

from soccerdata import ClubElo
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Create an instance of the ClubElo class
club_elo = ClubElo()

# Define the start and end dates
start_date = datetime(2014, 1, 1)
end_date = datetime.today()

# Create a list to store the DataFrames
elo_frames = []

# Iterate through the dates and retrieve the ELO scores, incrementing by 3 months at a time
current_date = start_date
count=0
while current_date <= end_date:
    elo_scores = club_elo.read_by_date(current_date)
    elo_frames.append(elo_scores)
    current_date += timedelta(days=120)  # Increment by 3 months (approximately)
    count+=1
    print(count)

# Concatenate the DataFrames
elo_df = pd.concat(elo_frames)

elo_df = elo_df.reset_index()
# List of leagues to keep
leagues_to_keep = ['GER', 'ESP', 'ENG', 'ITA', 'FRA']

# Filter the DataFrame to keep only the rows corresponding to the specified leagues
elo_df_filtered = elo_df[elo_df['country'].isin(leagues_to_keep)]

# Get a list of unique teams from the filtered DataFrame
unique_teams = elo_df_filtered['team'].unique()

# Now, elo_df_filtered contains only the rows for the specified leagues, and unique_teams is a list of all the teams in those leagues

# Create an empty list to store the DataFrames for each team's history
team_elo_frames = []

# Iterate through the unique teams and retrieve the full ELO history for each
for team in unique_teams:
    team_elo = club_elo.read_team_history(team)
    
    # Add the team's ELO history to the list
    team_elo_frames.append(team_elo)

# Concatenate the DataFrames to create a single DataFrame with the full history for all teams
all_teams_elo_history = pd.concat(team_elo_frames)

all_teams_elo_history = all_teams_elo_history.reset_index()
all_teams_elo_history.head(5)

# Define the date after which to keep the ELO scores
date_cutoff = datetime(2014, 1, 1)

# Filter the rows to include those where 'from' is before the cutoff and 'to' is after the cutoff
filtered_elo_df = all_teams_elo_history[(all_teams_elo_history['from'] >= date_cutoff) | (all_teams_elo_history['to'] >= date_cutoff)]

# Keep only the 'team', 'from', and 'to' columns
filtered_elo_df = filtered_elo_df[['team', 'from', 'to','elo']]
# Now, all_teams_elo_history contains the filtered ELO history for all the teams in unique_teams, including only the 'team', 'from', and 'to' columns, and only the rows where 'from' is before the cutoff and 'to' is after the cutoff
filtered_elo_df.to_csv('Elo_Scores.csv',sep=',')
