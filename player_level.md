# Player-Derived Team Features Specification

## Overview

These features are computed from **player-level match statistics** sourced from Understat. They aggregate individual player performance into team-level rolling features that capture team ability,
concentration of talent, and squad depth.

## Data Source

- **Source**: Understat player match statistics
- **Granularity**: One row per (player, match)
- **Key columns used**: `minutes`, `goals`, `xg`, `xa`, `assists`, `shots`, `key_passes`, `player_id`, `team_id`, `game_id`, `match_id`

## Rolling Window Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_WINDOW` | 15 | Maximum games to look back |
| `MIN_GAMES` | 2 | Minimum games required to compute (returns null otherwise) |
| `shift` | 1 | All features are shifted by 1 game to prevent data leakage |

## Feature Categories

### 1. Team Aggregate Stats (Rolling Mean over 15 games)

These are simple sums of player stats per match, then averaged over the lookback window.

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `team_xg_r15` | `mean(sum(player_xg))` | Rolling mean of team's total xG per match |
| `team_xa_r15` | `mean(sum(player_xa))` | Rolling mean of team's total xA per match |
| `team_total_goals_r15` | `mean(sum(player_goals))` | Rolling mean of goals scored |
| `team_total_assists_r15` | `mean(sum(player_assists))` | Rolling mean of assists |
| `team_total_shots_r15` | `mean(sum(player_shots))` | Rolling mean of total shots |
| `team_total_key_passes_r15` | `mean(sum(player_key_passes))` | Rolling mean of key passes |
| `team_total_minutes_r15` | `mean(sum(player_minutes))` | Rolling mean of total minutes (should be ~990 per match for 11 players) |

### 2. Concentration Metrics (Herfindahl-Hirschman Index)

HHI measures how concentrated a statistic is among players. Range: 0 to 1.
- **HHI = 1.0**: One player accounts for 100% of the output
- **HHI = 0.1**: Output evenly distributed among 10 players
- **HHI = 0.5**: Dominated by 2 players

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `xg_hhi_r15` | `mean(sum(player_xg^2) / sum(player_xg)^2)` | Concentration of xG production. High = star-dependent |
| `xa_hhi_r15` | `mean(sum(player_xa^2) / sum(player_xa)^2)` | Concentration of xA production. High = playmaker-dependent |
| `minutes_hhi_r15` | `mean(sum(player_min^2) / sum(player_min)^2)` | Concentration of minutes. High = less rotation |

**Interpretation**:
- High `xg_hhi` = team relies on 1-2 players for goals (risky if injured)
- Low `xg_hhi` = goal threat spread across many players (more resilient)
- High `minutes_hhi` = manager uses fewer players (less rotation, possible fatigue)

### 3. Squad Depth Features

| Feature Name | Formula | Description |
|--------------|---------|-------------|
| `unique_players_r15` | `mean(count_distinct(player_id))` | Avg unique players used per match over 15 games |
| `unique_players_r5_sum` | `sum(count_distinct(player_id))` | Total unique players used in last 5 games |

**Interpretation**:
- High `unique_players_r15` = large squad rotation (depth or injuries)
- Low value = settled starting XI

## Computation Steps

### Step 1: Aggregate Player Stats to Team-Match Level

For each `(league, season, team_id, game_id)`:

```python
team_match = player_df.group_by(["league", "season", "team_id", "game_id"]).agg([
    pl.col("minutes").sum().alias("team_total_minutes"),
    pl.col("xg").sum().alias("team_total_xg"),
    pl.col("xa").sum().alias("team_total_xa"),
    pl.col("goals").sum().alias("team_total_goals"),
    pl.col("assists").sum().alias("team_total_assists"),
    pl.col("shots").sum().alias("team_total_shots"),
    pl.col("key_passes").sum().alias("team_total_key_passes"),
    pl.col("player_id").n_unique().alias("unique_players"),
    (pl.col("xg").pow(2).sum()).alias("sum_xg_squared"),
    (pl.col("xa").pow(2).sum()).alias("sum_xa_squared"),
    (pl.col("minutes").pow(2).sum()).alias("sum_minutes_squared"),
])

Step 2: Compute HHI
xg_hhi = sum_xg_squared / team_total_xg^2 (if team_total_xg > 0, else null)
xa_hhi = sum_xa_squared / team_total_xa^2 (if team_total_xa > 0, else null)
minutes_hhi = sum_minutes_squared / team_total_minutes^2

Step 3: Compute Rolling Features

For each team, sorted by date:

for stat in stats_to_roll:
    feature = stat.shift(1).rolling_mean(window_size=15, min_samples=2).over(["league", "team_id"])

Critical: The .shift(1) ensures we only use data from previous matches, preventing data leakage.

Output Schema

Final features per team-match (joined to match data as home_, and away_ prefixes):

Column	Type
home_team_xg_r15	Float64
home_team_xa_r15	Float64
home_team_total_goals_r15	Float64
home_team_total_assists_r15	Float64
home_team_total_shots_r15	Float64
home_team_total_key_passes_r15	Float64
home_team_total_minutes_r15	Float64
home_xg_hhi_r15	Float64
home_xa_hhi_r15	Float64
home_minutes_hhi_r15	Float64
home_unique_players_r15	Float64
home_unique_players_r5_sum	Float64

Same columns exist with away_ prefix.

Join Logic

Features are joined to match-level data using:

league (string)

home_team_id / away_team_id (int)

game_id (int)

Validation Results

From testing on 2014-2025 data across 5 European leagues:

Correlation with match result: home_team_xg_r15 has r=0.235 (higher than base team xG r=0.196)

Feature importance: 15 of top 20 LightGBM features are player-derived

HHI features are unique: Only -0.35 to -0.47 correlation with base xG (captures new info)

Model improvement: +0.44% accuracy, -1.2% log loss when added to base features

Known Limitations

GER-Bundesliga 2024-25: Missing due to Understat API format issue

First games per team: Null features (waiting for min_samples)

Cross-season: Rolling window resets at season boundaries (by design)