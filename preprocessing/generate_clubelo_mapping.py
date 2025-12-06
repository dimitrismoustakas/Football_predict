import json
import difflib
from pathlib import Path
import pandas as pd

MAPPINGS_DIR = Path("data/mappings")
ELO_UNIVERSE_PATH = Path("data/eloscores/team_universe.parquet")
UNDERSTAT_MAPPING_PATH = MAPPINGS_DIR / "understat_to_canonical.json"
OUTPUT_MAPPING_PATH = MAPPINGS_DIR / "clubelo_to_canonical.json"

def load_canonical_teams():
    if not UNDERSTAT_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {UNDERSTAT_MAPPING_PATH}")
    
    with open(UNDERSTAT_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    
    # The values are the canonical names
    return set(mapping.values())

def load_clubelo_teams():
    if not ELO_UNIVERSE_PATH.exists():
        print(f"Warning: {ELO_UNIVERSE_PATH} not found. Please run data_collection/collect_elo.py first.")
        return []
    
    df = pd.read_parquet(ELO_UNIVERSE_PATH)
    return df["team"].tolist()

def main():
    canonical_teams = load_canonical_teams()
    clubelo_teams = load_clubelo_teams()
    
    if not clubelo_teams:
        return

    mapping = {}
    unmatched = []

    # Manual overrides for known tricky ones
    manual_map = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Gladbach": "Borussia M.Gladbach",
        "Paris SG": "Paris Saint Germain",
        "Atletico": "Atletico Madrid",
        "Bilbao": "Athletic Club",
        "Spurs": "Tottenham",
        "Wolves": "Wolverhampton Wanderers",
        "Nurnberg": "Nuernberg",
        "Dusseldorf": "Fortuna Duesseldorf",
        "Mainz": "Mainz 05",
        "Schalke": "Schalke 04",
        "Hertha": "Hertha Berlin",
        "Koln": "FC Cologne",
        "Koeln": "FC Cologne",
        "Betis": "Real Betis",
        "Sociedad": "Real Sociedad",
        "Celta": "Celta Vigo",
        "Vallecano": "Rayo Vallecano",
        "Espanyol": "Espanyol",
        "Gijon": "Sporting Gijon",
        "La Coruna": "Deportivo La Coruna",
        "Depor": "Deportivo La Coruna",
        "Las Palmas": "Las Palmas",
        "Leganes": "Leganes",
        "Osasuna": "Osasuna",
        "Villarreal": "Villarreal",
        "Sevilla": "Sevilla",
        "Valencia": "Valencia",
        "Getafe": "Getafe",
        "Levante": "Levante",
        "Eibar": "Eibar",
        "Malaga": "Malaga",
        "Granada": "Granada",
        "Alaves": "Alaves",
        "Girona": "Girona",
        "Valladolid": "Real Valladolid",
        "Huesca": "SD Huesca",
        "Mallorca": "Mallorca",
        "Cadiz": "Cadiz",
        "Elche": "Elche",
        "Almeria": "Almeria",
        "Cordoba": "Cordoba",
        "Hercules": "Hercules",
        "Numancia": "Numancia",
        "Recreativo": "Recreativo",
        "Tenerife": "Tenerife",
        "Xerez": "Xerez",
        "Zaragoza": "Real Zaragoza",
        "Ajaccio": "Ajaccio",
        "Auxerre": "Auxerre",
        "Bastia": "SC Bastia",
        "Bordeaux": "Bordeaux",
        "Brest": "Brest",
        "Caen": "Caen",
        "Dijon": "Dijon",
        "Evian": "Evian Thonon Gaillard",
        "Evian TG": "Evian Thonon Gaillard",
        "Guingamp": "Guingamp",
        "Lens": "Lens",
        "Lille": "Lille",
        "Lorient": "Lorient",
        "Lyon": "Lyon",
        "Marseille": "Marseille",
        "Metz": "Metz",
        "Monaco": "Monaco",
        "Montpellier": "Montpellier",
        "Nancy": "Nancy",
        "Nantes": "Nantes",
        "Nice": "Nice",
        "Reims": "Reims",
        "Rennes": "Rennes",
        "Saint-Etienne": "Saint-Etienne",
        "Sochaux": "Sochaux",
        "Strasbourg": "Strasbourg",
        "Toulouse": "Toulouse",
        "Troyes": "Troyes",
        "Valenciennes": "Valenciennes",
        "Angers": "Angers",
        "Gazelec Ajaccio": "GFC Ajaccio",
        "Gazelec": "GFC Ajaccio",
        "Amiens": "Amiens",
        "Nimes": "Nimes",
        "Clermont": "Clermont Foot",
        "Le Havre": "Le Havre",
        "Fuerth": "Greuther Fuerth",
        "QPR": "Queens Park Rangers",
        "Leipzig": "RasenBallsport Leipzig",
        "Frankfurt": "Eintracht Frankfurt",
        "Hamburg": "Hamburger SV",
        "Hannover": "Hannover 96",
        "Stuttgart": "VfB Stuttgart",
        "Bremen": "Werder Bremen",
        "West Brom": "West Bromwich Albion",
        "Newcastle": "Newcastle United",
        "Spal": "SPAL",
        "Verona": "Verona",
        "Parma": "Parma",
        "Inter": "Inter",
        "Milan": "AC Milan",
        "Roma": "Roma",
        "Lazio": "Lazio",
        "Udinese": "Udinese",
        "Sampdoria": "Sampdoria",
        "Genoa": "Genoa",
        "Torino": "Torino",
        "Fiorentina": "Fiorentina",
        "Napoli": "Napoli",
        "Salernitana": "Salernitana",
        "Empoli": "Empoli",
        "Sassuolo": "Sassuolo",
        "Frosinone": "Frosinone",
        "Cagliari": "Cagliari",
        "Bologna": "Bologna",
        "Atalanta": "Atalanta",
        "Monza": "Monza",
        "Lecce": "Lecce",
        "Spezia": "Spezia",
        "Cremonese": "Cremonese",
        "Venezia": "Venezia",
        "Benevento": "Benevento",
        "Crotone": "Crotone",
        "Pescara": "Pescara",
        "Carpi": "Carpi",
        "Cesena": "Cesena",
        "Palermo": "Palermo",
        "Chievo": "Chievo",
        "Como": "Como",
        "Darmstadt": "Darmstadt",
        "Heidenheim": "FC Heidenheim",
        "Hoffenheim": "Hoffenheim",
        "Kiel": "Holstein Kiel",
        "Huddersfield": "Huddersfield",
        "Hull": "Hull",
        "Ingolstadt": "Ingolstadt",
        "Ipswich": "Ipswich",
        "Leeds": "Leeds",
        "Leicester": "Leicester",
        "Liverpool": "Liverpool",
        "Luton": "Luton",
        "Middlesbrough": "Middlesbrough",
        "Norwich": "Norwich",
        "Forest": "Nottingham Forest",
        "Paderborn": "Paderborn",
        "Paris FC": "Paris FC",
        "Sheffield United": "Sheffield United",
        "Southampton": "Southampton",
        "St Pauli": "St. Pauli",
        "Stoke": "Stoke",
        "Sunderland": "Sunderland",
        "Swansea": "Swansea",
        "Watford": "Watford",
        "West Ham": "West Ham",
        "Wolfsburg": "Wolfsburg",
        "Union Berlin": "Union Berlin",
        "Freiburg": "Freiburg",
        "Bayer Leverkusen": "Bayer Leverkusen",
        "Leverkusen": "Bayer Leverkusen",
        "Bayern": "Bayern Munich",
        "Dortmund": "Borussia Dortmund",
        "Bochum": "Bochum",
        "Bielefeld": "Arminia Bielefeld",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "RB Leipzig": "RasenBallsport Leipzig",
        "Werder": "Werder Bremen",
        "Holstein": "Holstein Kiel",
        "Oviedo": "Real Oviedo",
        "Duesseldorf": "Fortuna Duesseldorf",
    }

    for team in clubelo_teams:
        if team in canonical_teams:
            mapping[team] = team
        elif team in manual_map:
            if manual_map[team] in canonical_teams:
                mapping[team] = manual_map[team]
            else:
                # Try to find close match for manual map result?
                # Or just assume manual map is correct but maybe canonical list is incomplete?
                # For now, trust manual map if target is in canonical
                pass
        else:
            # Fuzzy match - Stricter cutoff
            matches = difflib.get_close_matches(team, list(canonical_teams), n=1, cutoff=0.85)
            if matches:
                mapping[team] = matches[0]
                # print(f"Fuzzy match: {team} -> {matches[0]}")
            else:
                unmatched.append(team)

    # Save mapping
    with open(OUTPUT_MAPPING_PATH, "w") as f:
        json.dump(mapping, f, indent=4, sort_keys=True)
    
    print(f"Generated mapping for {len(mapping)} teams.")
    print(f"Saved to {OUTPUT_MAPPING_PATH}")
    
    if unmatched:
        print(f"\nUnmatched teams ({len(unmatched)}):")
        for t in sorted(unmatched):
            print(f"  {t}")

if __name__ == "__main__":
    main()
