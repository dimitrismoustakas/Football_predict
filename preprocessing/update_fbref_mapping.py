"""
Update FBRef to canonical (Understat) team name mapping.
Maps all FBRef domestic league team names to their Understat equivalents.
"""

import polars as pl
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MAPPINGS_DIR = DATA_DIR / "mappings"


def main():
	# Get all FBRef domestic teams
	fbref = pl.read_csv(DATA_DIR / "full_schedule" / "domestic_all.csv")
	fbref_teams = set(fbref["home_team"].unique().to_list()) | set(fbref["away_team"].unique().to_list())
	print(f"FBRef domestic teams: {len(fbref_teams)}")

	# Get all Understat teams (these are our canonical names)
	understat = pl.scan_parquet(str(DATA_DIR / "understat/*/*/matches.parquet")).select(["home_team", "away_team"]).collect()
	understat_teams = set(understat["home_team"].unique().to_list()) | set(understat["away_team"].unique().to_list())
	print(f"Understat teams: {len(understat_teams)}")

	# Start mapping - exact matches first
	mapping = {}
	for t in fbref_teams:
		if t in understat_teams:
			mapping[t] = t  # Exact match
	
	print(f"Exact matches: {len(mapping)}")

	# Manual mappings for teams that don't match exactly
	# FBRef name -> Understat name
	manual_mappings = {
		# English
		"Manchester Utd": "Manchester United",
		"Newcastle Utd": "Newcastle United",
		"Leeds United": "Leeds",
		"Leicester City": "Leicester",
		"Stoke City": "Stoke",
		"Swansea City": "Swansea",
		"Hull City": "Hull",
		"Cardiff City": "Cardiff",
		"Norwich City": "Norwich",
		"Ipswich Town": "Ipswich",
		"Luton Town": "Luton",
		"Sheffield Utd": "Sheffield United",
		"Nott'ham Forest": "Nottingham Forest",
		"West Brom": "West Bromwich Albion",
		"Wolves": "Wolverhampton Wanderers",
		"QPR": "Queens Park Rangers",
		"Tottenham Hotspur": "Tottenham",
		"West Ham United": "West Ham",
		
		# Spanish
		"Atlético Madrid": "Atletico Madrid",
		"Alavés": "Alaves",
		"Almería": "Almeria",
		"Cádiz": "Cadiz",
		"Córdoba": "Cordoba",
		"Málaga": "Malaga",
		"La Coruña": "Deportivo La Coruna",
		"Betis": "Real Betis",
		"Sporting Gijón": "Sporting Gijon",
		"Leganés": "Leganes",
		"Valladolid": "Real Valladolid",
		"Huesca": "SD Huesca",
		"Oviedo": "Real Oviedo",
		"Hellas Verona": "Verona",
		
		# German
		"Dortmund": "Borussia Dortmund",
		"Leverkusen": "Bayer Leverkusen",
		"Stuttgart": "VfB Stuttgart",
		"Eint Frankfurt": "Eintracht Frankfurt",
		"RB Leipzig": "RasenBallsport Leipzig",
		"Gladbach": "Borussia M.Gladbach",
		"Hertha BSC": "Hertha Berlin",
		"Köln": "FC Cologne",
		"Nürnberg": "Nuernberg",
		"Düsseldorf": "Fortuna Duesseldorf",
		"Arminia": "Arminia Bielefeld",
		"Paderborn 07": "Paderborn",
		"Ingolstadt 04": "Ingolstadt",
		"Darmstadt 98": "Darmstadt",
		"Greuther Fürth": "Greuther Fuerth",
		"BTSV": "Eintracht Braunschweig",  # Not in Understat - will be skipped
		"Heidenheim": "FC Heidenheim",
		"Karlsruher": "Karlsruher SC",  # Not in Understat - will be skipped
		"Elversberg": "SV Elversberg",  # Not in Understat - will be skipped
		"St Pauli": "St. Pauli",
		
		# Italian
		"Milan": "AC Milan",
		"SPAL": "SPAL 2013",
		
		# French
		"Paris S-G": "Paris Saint Germain",
		"Saint-Étienne": "Saint-Etienne",
		"Nîmes": "Nimes",
		"Bastia": "SC Bastia",
		"Evian": "Evian Thonon Gaillard",
		"Gazélec Ajaccio": "GFC Ajaccio",
	}

	# Apply manual mappings
	for fbref_name, understat_name in manual_mappings.items():
		if fbref_name in fbref_teams:
			if understat_name in understat_teams:
				mapping[fbref_name] = understat_name
			else:
				print(f"WARNING: Manual mapping target '{understat_name}' not found in Understat teams (for '{fbref_name}')")

	# Find still unmapped teams
	unmapped = fbref_teams - set(mapping.keys())
	if unmapped:
		print(f"\nStill unmapped ({len(unmapped)} teams):")
		for t in sorted(unmapped):
			print(f"  {t}")

	# Load existing mapping and merge
	existing_path = MAPPINGS_DIR / "fbref_to_canonical.json"
	if existing_path.exists():
		with open(existing_path, "r", encoding="utf-8") as f:
			existing = json.load(f)
		# Keep existing mappings for non-domestic teams (European competitions)
		for k, v in existing.items():
			if k not in mapping and v is not None:
				mapping[k] = v

	# Save updated mapping
	MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
	output_path = MAPPINGS_DIR / "fbref_to_canonical.json"
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(mapping, f, indent=2, ensure_ascii=False)
	
	print(f"\nSaved {len(mapping)} mappings to {output_path}")


if __name__ == "__main__":
	main()
