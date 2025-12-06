import polars as pl
import pandas as pd
import json
from pathlib import Path
import difflib

UNDERSTAT_FILE = Path("data/training/understat_df.parquet")
MATCH_HISTORY_FILE = Path("data/match_history/matches.parquet")
UNDERSTAT_MAPPING_FILE = Path("data/mappings/understat_to_canonical.json")
FOOTBALLDATA_MAPPING_FILE = Path("data/mappings/footballdata_to_canonical.json")

# Define Canonical Names for tricky cases
# Format: "Raw Name": "Canonical Name"
# This list acts as the "Source of Truth" for normalization
MANUAL_OVERRIDES = {
    # Understat variants
    "Parma Calcio 1913": "Parma",
    "Parma": "Parma",
    "SPAL 2013": "SPAL",
    "Milan": "AC Milan",
    "Inter": "Inter",
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
    
    # Football-Data variants
    "Spal": "SPAL",
    "Spezia": "Spezia",
    "Ajaccio GFCO": "GFC Ajaccio",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico Madrid",
    "Celta": "Celta Vigo",
    "Espanyol": "Espanyol",
    "M'gladbach": "Borussia M.Gladbach",
    "Paris SG": "Paris Saint Germain",
    "West Brom": "West Bromwich Albion",
    "Wolves": "Wolverhampton Wanderers",
    "QPR": "Queens Park Rangers",
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
    "Schalke 04": "Schalke 04",
    "Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "Freiburg": "Freiburg",
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
    "Alaves": "Alaves",
}

def normalize(name):
    if not isinstance(name, str):
        return ""
    return name.lower().strip().replace(" ", "").replace(".", "").replace("-", "")

def main():
    print("Generating Unified Team Mappings...")
    
    # 1. Load all raw names
    udf = pl.read_parquet(UNDERSTAT_FILE)
    u_teams = sorted(list(set(udf["home_team"].unique().to_list()) | set(udf["away_team"].unique().to_list())))
    print(f"Found {len(u_teams)} unique teams in Understat.")
    
    mdf = pd.read_parquet(MATCH_HISTORY_FILE)
    m_teams = sorted(list(set(mdf['home_team'].unique()) | set(mdf['away_team'].unique())))
    print(f"Found {len(m_teams)} unique teams in MatchHistory.")
    
    # 2. Build Understat Mapping
    # Goal: Map every Understat name to a Canonical Name
    u_mapping = {}
    canonical_names = set()
    
    for team in u_teams:
        if team in MANUAL_OVERRIDES:
            canonical = MANUAL_OVERRIDES[team]
        else:
            # Default: The name itself is canonical
            canonical = team
            
        u_mapping[team] = canonical
        canonical_names.add(canonical)
        
    # 3. Build Football-Data Mapping
    # Goal: Map every Football-Data name to a Canonical Name (must exist in canonical_names set if possible)
    m_mapping = {}
    
    # Create a normalized lookup for canonical names
    norm_canonical = {normalize(c): c for c in canonical_names}
    
    for team in m_teams:
        if team in MANUAL_OVERRIDES:
            canonical = MANUAL_OVERRIDES[team]
        elif team in canonical_names:
            canonical = team
        else:
            # Try normalized match
            nm = normalize(team)
            if nm in norm_canonical:
                canonical = norm_canonical[nm]
            else:
                # Fuzzy match against canonical names
                matches = difflib.get_close_matches(team, list(canonical_names), n=1, cutoff=0.6)
                if matches:
                    canonical = matches[0]
                    print(f"Fuzzy match: '{team}' -> '{canonical}'")
                else:
                    print(f"WARNING: No match found for '{team}'. Using raw name as canonical.")
                    canonical = team
        
        m_mapping[team] = canonical

    # 4. Save Mappings
    with open(UNDERSTAT_MAPPING_FILE, "w") as f:
        json.dump(u_mapping, f, indent=4)
    print(f"Saved Understat mapping to {UNDERSTAT_MAPPING_FILE}")
    
    with open(FOOTBALLDATA_MAPPING_FILE, "w") as f:
        json.dump(m_mapping, f, indent=4)
    print(f"Saved Football-Data mapping to {FOOTBALLDATA_MAPPING_FILE}")

if __name__ == "__main__":
    main()
