import polars as pl
import pandas as pd
import json
from pathlib import Path
import difflib

UNDERSTAT_FILE = Path("data/training/understat_df.parquet")
MATCH_HISTORY_FILE = Path("data/match_history/matches.parquet")
MAPPING_FILE = Path("data/team_mapping.json")

def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.lower().strip().replace(" ", "").replace(".", "").replace("-", "")

def main():
    # Load Understat teams
    if not UNDERSTAT_FILE.exists():
        print(f"Understat file not found: {UNDERSTAT_FILE}")
        return
    
    udf = pl.read_parquet(UNDERSTAT_FILE)
    u_teams = set(udf["home_team"].unique().to_list()) | set(udf["away_team"].unique().to_list())
    u_teams = sorted(list(u_teams))
    print(f"Understat teams: {len(u_teams)}")

    # Load MatchHistory teams
    if not MATCH_HISTORY_FILE.exists():
        print(f"MatchHistory file not found: {MATCH_HISTORY_FILE}")
        return
    
    mdf = pd.read_parquet(MATCH_HISTORY_FILE)
    print(f"MatchHistory columns: {mdf.columns.tolist()}")
    
    if 'home_team' in mdf.columns:
        m_teams = set(mdf['home_team'].unique()) | set(mdf['away_team'].unique())
    else:
        print("Could not find home_team/away_team columns in MatchHistory.")
        return

    m_teams = sorted(list(m_teams))
    print(f"MatchHistory teams: {len(m_teams)}")

    # Mapping: MatchHistory Name -> Understat Name
    mapping = {}
    
    # 1. Exact match
    u_teams_set = set(u_teams)
    unmatched_m = []
    
    for m_team in m_teams:
        if m_team in u_teams_set:
            mapping[m_team] = m_team
        else:
            unmatched_m.append(m_team)
            
    print(f"Exact matches: {len(mapping)}")
    print(f"Unmatched MatchHistory teams: {len(unmatched_m)}")
    
    # 2. Normalized match
    norm_u = {normalize(t): t for t in u_teams}
    
    still_unmatched = []
    for m_team in unmatched_m:
        nm = normalize(m_team)
        if nm in norm_u:
            mapping[m_team] = norm_u[nm]
        else:
            still_unmatched.append(m_team)
            
    print(f"After normalization: {len(mapping)} matched. {len(still_unmatched)} still unmatched.")
    
    # 3. Fuzzy match
    # Manual fixes for known issues
    manual_fixes = {
        "Ath Bilbao": "Athletic Club",
        "Ath Madrid": "Atletico Madrid",
        "Celta": "Celta Vigo",
        "Espanyol": "Espanyol",
        "M'gladbach": "Borussia M.Gladbach",
        "Paris SG": "Paris Saint Germain",
        "West Brom": "West Bromwich Albion",
        "Wolves": "Wolverhampton Wanderers",
        "QPR": "Queens Park Rangers",
        "Spal": "SPAL 2013",
        "Spezia": "Spezia",
        "Ajaccio GFCO": "GFC Ajaccio",
        "Nott'm Forest": "Nottingham Forest",
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Newcastle": "Newcastle United",
        "St Etienne": "Saint-Etienne",
        "Stuttgart": "VfB Stuttgart",
        "Valladolid": "Real Valladolid",
        "Vallecano": "Rayo Vallecano",
        "Sociedad": "Real Sociedad",
        "Betis": "Real Betis",
        "La Coruna": "Deportivo La Coruna",
        "Leverkusen": "Bayer Leverkusen",
        "Mainz": "Mainz 05",
        "Hertha": "Hertha Berlin",
        "Hamburg": "Hamburger SV",
        "Hannover": "Hannover 96",
        "FC Koln": "FC Cologne",
        "Dortmund": "Borussia Dortmund",
        "Bielefeld": "Arminia Bielefeld",
        "Augsburg": "Augsburg", # Likely exact
        "Hoffenheim": "Hoffenheim", # Likely exact
        "Schalke 04": "Schalke 04",
        "Werder Bremen": "Werder Bremen",
        "Wolfsburg": "Wolfsburg",
        "Freiburg": "Freiburg",
        "Inter": "Inter",
        "Milan": "AC Milan",
        "Roma": "Roma",
        "Lazio": "Lazio",
        "Napoli": "Napoli",
        "Juventus": "Juventus",
        "Fiorentina": "Fiorentina",
        "Atalanta": "Atalanta",
        "Torino": "Torino",
        "Sampdoria": "Sampdoria",
        "Genoa": "Genoa",
        "Udinese": "Udinese",
        "Bologna": "Bologna",
        "Cagliari": "Cagliari",
        "Chievo": "Chievo",
        "Empoli": "Empoli",
        "Sassuolo": "Sassuolo",
        "Verona": "Verona",
        "Palermo": "Palermo",
        "Frosinone": "Frosinone",
        "Carpi": "Carpi",
        "Benevento": "Benevento",
        "Crotone": "Crotone",
        "Pescara": "Pescara",
        "Salernitana": "Salernitana",
        "Monza": "Monza",
        "Lecce": "Lecce",
        "Cremonese": "Cremonese",
        "Como": "Como",
        "Venezia": "Venezia",
        "Parma": "Parma",
        "Brescia": "Brescia",
        "Livorno": "Livorno",
        "Cesena": "Cesena",
        "Novara": "Novara",
        "Siena": "Siena",
        "Catania": "Catania",
        "Reggina": "Reggina",
        "Messina": "Messina",
        "Ascoli": "Ascoli",
        "Treviso": "Treviso",
        "Almeria": "Almeria",
        "Granada": "Granada",
        "Mallorca": "Mallorca",
        "Osasuna": "Osasuna",
        "Sevilla": "Sevilla",
        "Valencia": "Valencia",
        "Villarreal": "Villarreal",
        "Getafe": "Getafe",
        "Levante": "Levante",
        "Malaga": "Malaga",
        "Racing Santander": "Racing Santander",
        "Sporting Gijon": "Sporting Gijon",
        "Tenerife": "Tenerife",
        "Xerez": "Xerez",
        "Hercules": "Hercules",
        "Elche": "Elche",
        "Eibar": "Eibar",
        "Cordoba": "Cordoba",
        "Las Palmas": "Las Palmas",
        "Leganes": "Leganes",
        "Girona": "Girona",
        "Cadiz": "Cadiz",
        "Alaves": "Alaves", # Fix Wolves -> Alaves
    }

    print("\nApplying manual fixes and fuzzy matching...")
    for m_team in still_unmatched:
        if m_team in manual_fixes:
             # Check if target exists in u_teams
             target = manual_fixes[m_team]
             if target in u_teams:
                 mapping[m_team] = target
                 print(f"  Manual: '{m_team}' -> '{target}'")
                 continue
             else:
                 print(f"  Manual target '{target}' for '{m_team}' NOT found in Understat teams. Trying fuzzy...")

        # Find closest match in u_teams
        # cutoff=0.6 is a bit loose, but let's see.
        matches = difflib.get_close_matches(m_team, u_teams, n=1, cutoff=0.6)
        if matches:
            best_match = matches[0]
            print(f"  '{m_team}' -> '{best_match}'")
            mapping[m_team] = best_match
        else:
            print(f"  No match found for '{m_team}'")
            mapping[m_team] = None 

    # Save mapping
    with open(MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
        
    print(f"Mapping saved to {MAPPING_FILE}")

if __name__ == "__main__":
    main()
