# Football Prediction Pipeline - Data Dictionary

This document provides a comprehensive overview of all data sources available in the project, their structure, columns, and usage.

---

## Table of Contents
1. [Data Sources Overview](#data-sources-overview)
2. [Elo Scores Data](#1-elo-scores-data)
3. [Understat Data](#2-understat-data)
4. [FBRef Schedule Data](#3-fbref-schedule-data)
5. [Match History / Odds Data](#4-match-history--odds-data)
6. [Training Data](#6-training-data)
7. [Team Mappings](#7-team-mappings)

---

## Data Sources Overview

| Source | Location | Format | Records | Date Range | Purpose |
|--------|----------|--------|---------|------------|---------|
| Elo Scores | `data/eloscores/` | Parquet | ~222K | 2013-12 to Present | Team strength ratings over time |
| Understat | `data/understat/<league>/<season>/` | Parquet | ~380/season/league | 2014-15 to Present | Match-level xG statistics |
| FBRef Schedule | `data/full_schedule/` | Parquet/CSV | ~6K | 2014-15 to Present | Match schedule, fixtures, results |
| Match History | `data/match_history/` | Parquet | ~20K | 2014-08 to Present | Historical odds & match statistics |
| Training | `data/training/` | Parquet | ~21K | Aggregated | Feature-engineered dataset |

---

## 1. Elo Scores Data

**Source:** ClubElo  
**Location:** `data/eloscores/`  
**Purpose:** Historical team strength ratings for ranking comparison

### 1.1 elo_history.parquet

**Shape:** 221,744 rows × 5 columns  
**Date Range:** December 2013 to Present

| Column | Type | Description |
|--------|------|-------------|
| `team` | String | Team name (ClubElo naming convention) |
| `country` | String | Country code (ENG, ESP, GER, FRA, ITA) |
| `from` | Datetime | Start date of this Elo rating period |
| `to` | Datetime | End date of this Elo rating period |
| `elo` | Float64 | Elo rating value (typically 1300-2100 range) |

**Notes:**
- Elo ratings update after each match
- Higher Elo = stronger team
- Ratings are relative within the system

### 1.2 team_universe.parquet

**Shape:** 308 rows × 2 columns  
**Purpose:** Master list of all teams tracked in Elo system

| Column | Type | Description |
|--------|------|-------------|
| `team` | String | Team name (ClubElo naming convention) |
| `country` | String | Country code (ENG, ESP, GER, FRA, ITA) |

**Covered Countries:**
- ENG (England)
- ESP (Spain)
- GER (Germany)
- FRA (France)
- ITA (Italy)

---

## 2. Understat Data

**Source:** Understat.com (via soccerdata)  
**Location:** `data/understat/<league>/<season>/matches.parquet`  
**Purpose:** Advanced match statistics with Expected Goals (xG)

### Covered Leagues
- `ENG-Premier_League` (English Premier League)
- `ESP-La_Liga` (Spanish La Liga)
- `GER-Bundesliga` (German Bundesliga)
- `FRA-Ligue_1` (French Ligue 1)
- `ITA-Serie_A` (Italian Serie A)

### Covered Seasons
2014-2015 through 2025-2026 (varies by league)

### matches.parquet

**Shape:** ~380 rows per season (varies by league format)  
**Columns:** 29

| Column | Type | Description |
|--------|------|-------------|
| **Identifiers** | | |
| `league` | String | League identifier (e.g., "ENG-Premier League") |
| `season` | String | Season code (e.g., "2425" for 2024-25) |
| `match_id` | String | Unique match identifier (date + teams) |
| `league_id` | String | Numeric league ID |
| `season_id` | Int64 | Numeric season ID |
| `game_id` | Int64 | Understat game ID |
| `date` | Datetime | Match date and kickoff time |
| **Team Info** | | |
| `home_team_id` | Int64 | Understat home team ID |
| `away_team_id` | Int64 | Understat away team ID |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |
| `home_team_code` | String | Home team abbreviation |
| `away_team_code` | String | Away team abbreviation |
| **Match Results** | | |
| `home_goals` | Int64 | Goals scored by home team |
| `away_goals` | Int64 | Goals scored by away team |
| `home_points` | Int64 | Points earned by home team (0, 1, or 3) |
| `away_points` | Int64 | Points earned by away team (0, 1, or 3) |
| **Expected Goals (xG)** | | |
| `home_xg` | Float64 | Home team's expected goals |
| `away_xg` | Float64 | Away team's expected goals |
| `home_np_xg` | Float64 | Home team's non-penalty xG |
| `away_np_xg` | Float64 | Away team's non-penalty xG |
| `home_expected_points` | Float64 | Home team's expected points (based on xG) |
| `away_expected_points` | Float64 | Away team's expected points (based on xG) |
| `home_np_xg_difference` | Float64 | Home team's non-penalty xG difference |
| `away_np_xg_difference` | Float64 | Away team's non-penalty xG difference |
| **Advanced Metrics** | | |
| `home_ppda` | Float64 | Home team's Passes Per Defensive Action |
| `away_ppda` | Float64 | Away team's Passes Per Defensive Action |
| `home_deep_completions` | Int64 | Home team's passes completed in final third |
| `away_deep_completions` | Int64 | Away team's passes completed in final third |

**Key Metrics Explained:**
- **xG (Expected Goals):** Statistical probability of scoring based on shot quality
- **npxG:** xG excluding penalty kicks
- **PPDA:** Pressing intensity metric (lower = more aggressive pressing)
- **Deep Completions:** Passes completed near the opponent's goal

---

## 3. FBRef Schedule Data

**Source:** FBRef (via soccerdata)  
**Location:** `data/full_schedule/`  
**Purpose:** Match schedule, fixtures, and basic results

### Available Files

| File | Records | Description |
|------|---------|-------------|
| `all_competitions.parquet` | 6,198 | All leagues including European competitions |
| `domestic_all.parquet` | 1,752 | Top 5 domestic leagues only |
| `domestic_upcoming.parquet` | Variable | Upcoming domestic fixtures |
| `all_upcoming.parquet` | ~97 | All upcoming fixtures |
| `european_all.parquet` | Variable | European competition matches |

### Column Schema (19 columns)

| Column | Type | Description |
|--------|------|-------------|
| **Match Info** | | |
| `league` | String | League name (e.g., "ENG-Premier League") |
| `season` | String | Season code (e.g., "2425") |
| `game` | String | Match identifier (date + home-away) |
| `game_id` | String | FBRef unique game ID (hash) |
| **Schedule** | | |
| `round` | String | Competition round (e.g., "Group stage", "Matchweek 15") |
| `week` | Int64 | Matchweek number |
| `day` | String | Day of week |
| `date` | Datetime | Match date |
| `time` | String | Kickoff time |
| **Teams & Result** | | |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |
| `score` | String | Final score (e.g., "2–1") or null if upcoming |
| **Venue Info** | | |
| `venue` | String | Stadium name |
| `attendance` | Int64 | Number of spectators |
| `referee` | String | Match referee |
| **Additional** | | |
| `match_report` | String | Link to match report |
| `notes` | String | Additional match notes |
| `home_xg` | Float64 | Home team xG (from FBRef) |
| `away_xg` | Float64 | Away team xG (from FBRef) |

**Covered Leagues:**
- ENG-Premier League
- ESP-La Liga
- GER-Bundesliga
- FRA-Ligue 1
- ITA-Serie A
- EUR-Champions League
- EUR-Europa League
- EUR-Conference League

**Season Range:** 2014-15 to 2025-26

---

## 4. Match History / Odds Data

**Source:** Football-Data.co.uk  
**Location:** `data/match_history/matches.parquet`  
**Purpose:** Historical match results with comprehensive betting odds

**Shape:** 20,483 rows × 185 columns  
**Date Range:** August 2014 to December 2025

### Core Match Columns

| Column | Type | Description |
|--------|------|-------------|
| `league` | String | League name |
| `season` | String | Season code |
| `game` | String | Match identifier |
| `date` | Datetime | Match date and time |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |

### Match Statistics

| Column | Type | Description |
|--------|------|-------------|
| **Final Score** | | |
| `FTHG` | Float64 | Full Time Home Goals |
| `FTAG` | Float64 | Full Time Away Goals |
| `FTR` | String | Full Time Result (H/D/A) |
| **Half Time** | | |
| `HTHG` | Float64 | Half Time Home Goals |
| `HTAG` | Float64 | Half Time Away Goals |
| `HTR` | String | Half Time Result (H/D/A) |
| **Shots** | | |
| `HS` | Float64 | Home Shots |
| `AS` | Float64 | Away Shots |
| `HST` | Float64 | Home Shots on Target |
| `AST` | Float64 | Away Shots on Target |
| **Fouls & Cards** | | |
| `HF` | Float64 | Home Fouls |
| `AF` | Float64 | Away Fouls |
| `HC` | Float64 | Home Corners |
| `AC` | Float64 | Away Corners |
| `HY` | Float64 | Home Yellow Cards |
| `AY` | Float64 | Away Yellow Cards |
| `HR` | Float64 | Home Red Cards |
| `AR` | Float64 | Away Red Cards |
| `referee` | String | Match referee |

### Betting Odds - Match Result (1X2)

**Opening Odds (before match):**

| Column | Description |
|--------|-------------|
| `B365H/B365D/B365A` | Bet365 - Home/Draw/Away |
| `BWH/BWD/BWA` | BetWin - Home/Draw/Away |
| `IWH/IWD/IWA` | Interwetten - Home/Draw/Away |
| `LBH/LBD/LBA` | Ladbrokes - Home/Draw/Away |
| `PSH/PSD/PSA` | Pinnacle - Home/Draw/Away |
| `WHH/WHD/WHA` | William Hill - Home/Draw/Away |
| `SJH/SJD/SJA` | Stan James - Home/Draw/Away |
| `VCH/VCD/VCA` | VC Bet - Home/Draw/Away |

**Aggregated Odds:**

| Column | Description |
|--------|-------------|
| `Bb1X2` | Number of bookmakers offering 1X2 odds |
| `BbMxH/BbMxD/BbMxA` | Maximum odds across bookmakers |
| `BbAvH/BbAvD/BbAvA` | Average odds across bookmakers |
| `MaxH/MaxD/MaxA` | Maximum opening odds |
| `AvgH/AvgD/AvgA` | Average opening odds |

**Closing Odds (at kickoff):**

| Column | Description |
|--------|-------------|
| `PSCH/PSCD/PSCA` | Pinnacle closing odds |
| `B365CH/B365CD/B365CA` | Bet365 closing odds |
| `BWCH/BWCD/BWCA` | BetWin closing odds |
| `MaxCH/MaxCD/MaxCA` | Maximum closing odds |
| `AvgCH/AvgCD/AvgCA` | Average closing odds |

### Betting Odds - Over/Under 2.5 Goals

| Column | Description |
|--------|-------------|
| `B365>2.5/B365<2.5` | Bet365 Over/Under 2.5 |
| `P>2.5/P<2.5` | Pinnacle Over/Under 2.5 |
| `BbMx>2.5/BbMx<2.5` | Maximum Over/Under odds |
| `BbAv>2.5/BbAv<2.5` | Average Over/Under odds |
| `Max>2.5/Max<2.5` | Maximum opening Over/Under |
| `Avg>2.5/Avg<2.5` | Average opening Over/Under |
| **Closing Odds** | |
| `B365C>2.5/B365C<2.5` | Bet365 closing Over/Under |
| `PC>2.5/PC<2.5` | Pinnacle closing Over/Under |
| `MaxC>2.5/MaxC<2.5` | Maximum closing Over/Under |
| `AvgC>2.5/AvgC<2.5` | Average closing Over/Under |

### Betting Odds - Asian Handicap

| Column | Description |
|--------|-------------|
| `AHh` | Asian Handicap line (e.g., -1.5, +0.25) |
| `BbAHh` | Betbrain Asian Handicap line |
| `B365AHH/B365AHA` | Bet365 Asian Handicap Home/Away |
| `PAHH/PAHA` | Pinnacle Asian Handicap Home/Away |
| `BbMxAHH/BbMxAHA` | Maximum Asian Handicap odds |
| `BbAvAHH/BbAvAHA` | Average Asian Handicap odds |
| **Closing Odds** | |
| `AHCh` | Closing Asian Handicap line |
| `B365CAHH/B365CAHA` | Bet365 closing AH odds |
| `PCAHH/PCAHA` | Pinnacle closing AH odds |

### Additional Bookmakers

| Prefix | Bookmaker |
|--------|-----------|
| `BF` | Betfair |
| `BFE` | Betfair Exchange |
| `1XB` | 1XBet |
| `BFD` | Betfred |
| `BMGM` | BetMGM |
| `BV` | Bet Victor |
| `CL` | Coral/Ladbrokes |

**Covered Leagues:**
- ENG-Premier League
- ESP-La Liga
- GER-Bundesliga
- FRA-Ligue 1
- ITA-Serie A

---

## 5. Training Data

**Location:** `data/training/understat_df.parquet`  
**Purpose:** Feature-engineered dataset ready for model training

**Shape:** 20,613 rows × 780 columns

### Base Columns

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | String | Unique match identifier |
| `league_id` | String | League identifier |
| `league` | String | League name |
| `season` | String | Season code |
| `date` | Datetime | Match date |
| `home_team` | String | Home team name |
| `away_team` | String | Away team name |
| `home_goals` | Int64 | Home team goals |
| `away_goals` | Int64 | Away team goals |
| `Over` | Boolean | Whether total goals > 2.5 |

### Rolling Feature Naming Convention

Features follow the pattern: `<scope>__<stat>__<window>__<side>`

**Scopes:**
- `ovr` - Overall stats (all matches)
- `home` - Home matches only
- `away` - Away matches only

**Statistics:**
- `xg_for` / `xg_against` - Expected goals for/against
- `npxg_for` / `npxg_against` - Non-penalty xG for/against
- `shots_for` / `shots_against` - Shots for/against
- `sot_for` / `sot_against` - Shots on target for/against
- `deep_for` / `deep_against` - Deep completions for/against
- `ppda_for` / `ppda_against` - PPDA for/against
- `gf` / `ga` - Goals for/against
- `xgd` - Expected goal difference

**Rolling Windows:**
- `r3` - Rolling mean over last 3 matches
- `r5` - Rolling mean over last 5 matches
- `r10` - Rolling mean over last 10 matches
- `sum` suffix - Rolling sum instead of mean

**Sides:**
- `__h` - Home team's feature
- `__a` - Away team's feature

**Example Features:**
```
ovr__xg_for__r5__h      # Home team's avg xG for (last 5 overall matches)
ovr__xg_for__sum__r3__a # Away team's sum of xG for (last 3 overall matches)
home__shots_for__r5__h  # Home team's avg shots (last 5 home matches)
away__ppda_for__r10__a  # Away team's avg PPDA (last 10 away matches)
```

---

## 6. Team Mappings

**Location:** `data/mappings/`  
**Purpose:** Normalize team names across different data sources

### Available Mappings

| File | Source → Target |
|------|-----------------|
| `understat_to_canonical.json` | Understat names → Canonical |
| `fbref_to_canonical.json` | FBRef names → Canonical |
| `footballdata_to_canonical.json` | Football-Data names → Canonical |
| `clubelo_to_canonical.json` | ClubElo names → Canonical |
| `theoddsapi_to_canonical.json` | The Odds API names → Canonical |

### Format

Each mapping file is a JSON dictionary:
```json
{
    "Source Team Name": "Canonical Team Name",
    "Manchester United": "Manchester United",
    "Man United": "Manchester United",
    "Wolverhampton Wanderers": "Wolverhampton"
}
```

### Common Name Variations

| Canonical | Understat | FBRef | Football-Data |
|-----------|-----------|-------|---------------|
| Manchester United | Manchester United | Manchester Utd | Man United |
| Wolverhampton | Wolverhampton Wanderers | Wolves | Wolverhampton |
| Nottingham Forest | Nottingham Forest | Nott'ham Forest | Nott'm Forest |
| Brighton | Brighton | Brighton | Brighton |

---

## Data Quality Notes

### Completeness
- **Elo Scores:** Complete daily coverage since Dec 2013
- **Understat:** No null values in core metrics
- **FBRef Schedule:** `xG` values may be null for future/old matches
- **Match History/Odds:** Some bookmaker columns have significant nulls

### Known Issues
1. **Team Name Inconsistency:** Different sources use different team names (resolved via mappings)
2. **Salary Data:** Not yet integrated into features
3. **Closing Odds:** More sparse than opening odds
4. **European Competitions:** Available in schedule but not in odds data

### Recommended Data Usage
- **Model Training:** Use `data/training/understat_df.parquet` (pre-engineered features)
- **Production Inference:** Use `data/prod/features_season.parquet`
- **Odds Analysis:** Use `data/match_history/matches.parquet` with `Avg*` or `Max*` columns
- **Team Strength:** Use `data/eloscores/elo_history.parquet` joined on date

---

## File Size Reference

| File | Approximate Size |
|------|------------------|
| `elo_history.parquet` | ~5 MB |
| `matches.parquet` (match_history) | ~15 MB |
| `understat_df.parquet` (training) | ~50 MB |
| `all_competitions.parquet` | ~1 MB |
| Individual understat season | ~50 KB |
