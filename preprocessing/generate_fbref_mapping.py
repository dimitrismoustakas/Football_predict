"""
Generate comprehensive FBRef to Understat team name mapping.
Covers all Big-5 league teams that have played in European competitions.
"""

import polars as pl
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import DATA_DIR, MAPPINGS_DIR


def main():
	# Get all unique teams from Understat data  
	understat = pl.scan_parquet(str(DATA_DIR / "understat/**/matches.parquet")).select(["home_team", "away_team"]).collect()
	understat_teams = set(understat["home_team"].unique().to_list()) | set(understat["away_team"].unique().to_list())
	print(f"Understat teams: {len(understat_teams)}")

	# Get European teams
	european = pl.read_csv(DATA_DIR / "full_schedule" / "european_all.csv")
	european_teams = set(european["home_team"].unique().to_list()) | set(european["away_team"].unique().to_list())
	print(f"European teams: {len(european_teams)}")

	# Start with exact matches
	mapping = {}
	for t in european_teams:
		if t in understat_teams:
			mapping[t] = t  # Exact match
		else:
			mapping[t] = None  # No match yet

	# Known manual mappings based on name patterns
	manual_mappings = {
		# Spanish
		"Atlético Madrid": "Atletico Madrid",
		"Atletico": "Atletico Madrid", 
		"Betis": "Real Betis",
		"Real Sociedad": "Real Sociedad",
		"Sporting Gijón": "Sporting Gijon",
		"Deportivo La Coruña": "Deportivo La Coruna",
		# German
		"Dortmund": "Borussia Dortmund",
		"Leverkusen": "Bayer Leverkusen", 
		"Stuttgart": "VfB Stuttgart",
		"Eint Frankfurt": "Eintracht Frankfurt",
		"RB Leipzig": "RasenBallsport Leipzig",
		"Schalke": "Schalke 04",
		"Schalke 04": "Schalke 04",
		"Gladbach": "Borussia M.Gladbach",
		"M'Gladbach": "Borussia M.Gladbach",
		"Werder Bremen": "Werder Bremen",
		"Wolfsburg": "Wolfsburg",
		"Hoffenheim": "Hoffenheim",
		"Köln": "FC Cologne",
		"Hertha BSC": "Hertha Berlin",
		"Mainz": "Mainz 05",
		"Mainz 05": "Mainz 05",
		# Italian
		"AC Milan": "AC Milan",
		"Milan": "AC Milan",
		"Inter Milan": "Inter",
		"Internazionale": "Inter",
		"Lazio": "Lazio",
		"Roma": "Roma",
		"Napoli": "Napoli",
		"Juventus": "Juventus",
		"Fiorentina": "Fiorentina",
		"Sampdoria": "Sampdoria",
		"Sassuolo": "Sassuolo",
		"Torino": "Torino",
		"Udinese": "Udinese",
		"Genoa": "Genoa",
		"Atalanta": "Atalanta",
		"Bologna": "Bologna",
		"Verona": "Verona",
		"Empoli": "Empoli",
		"Cagliari": "Cagliari",
		"Parma": "Parma",
		# French
		"Paris S-G": "Paris Saint Germain",
		"PSG": "Paris Saint Germain",
		"Saint-Étienne": "Saint-Etienne",
		"St Etienne": "Saint-Etienne",
		"Marseille": "Marseille",
		"Lyon": "Lyon",
		"Monaco": "Monaco",
		"Lille": "Lille",
		"Nice": "Nice",
		"Rennes": "Rennes",
		"Bordeaux": "Bordeaux",
		"Montpellier": "Montpellier",
		"Nantes": "Nantes",
		"Toulouse": "Toulouse",
		"Strasbourg": "Strasbourg",
		"Lens": "Lens",
		"Reims": "Reims",
		"Brest": "Brest",
		"Lorient": "Lorient",
		"Angers": "Angers",
		"Metz": "Metz",
		"Guingamp": "Guingamp",
		"Dijon": "Dijon",
		# English
		"Newcastle Utd": "Newcastle United",
		"Nott'ham Forest": "Nottingham Forest",
		"Wolves": "Wolverhampton Wanderers",
		"Man City": "Manchester City",
		"Man United": "Manchester United",
		"Man Utd": "Manchester United",
		"Spurs": "Tottenham",
		"Sheffield Utd": "Sheffield United",
		"West Brom": "West Bromwich Albion",
		"West Ham Utd": "West Ham",
		"Leicester City": "Leicester",
		"Arsenal": "Arsenal",
		"Chelsea": "Chelsea",
		"Liverpool": "Liverpool",
		"Manchester City": "Manchester City",
		"Manchester United": "Manchester United",
		"Tottenham": "Tottenham",
		"Everton": "Everton",
		"Aston Villa": "Aston Villa",
		"Brighton": "Brighton",
		"Crystal Palace": "Crystal Palace",
		"Fulham": "Fulham",
		"Southampton": "Southampton",
		"Bournemouth": "Bournemouth",
		"Burnley": "Burnley",
		"Leeds": "Leeds",
		"Leeds United": "Leeds",
		"Brentford": "Brentford",
		"Watford": "Watford",
		"Norwich": "Norwich",
		"Norwich City": "Norwich",
		"Stoke": "Stoke",
		"Stoke City": "Stoke",
		"Swansea": "Swansea",
		"Swansea City": "Swansea",
		"West Ham": "West Ham",
		"Hull": "Hull",
		"Hull City": "Hull",
		"Middlesbrough": "Middlesbrough",
		"Huddersfield": "Huddersfield",
		"Cardiff": "Cardiff",
		"Cardiff City": "Cardiff",
		"Sunderland": "Sunderland",
	}

	# Apply manual mappings
	for fbref_name, canonical in manual_mappings.items():
		if fbref_name in mapping:
			mapping[fbref_name] = canonical

	# Count mapped vs unmapped
	mapped = sum(1 for v in mapping.values() if v is not None)
	unmapped = sum(1 for v in mapping.values() if v is None)
	print(f"Mapped: {mapped}, Unmapped (non-Big-5): {unmapped}")

	# Show some unmapped for verification
	unmapped_teams = [k for k, v in mapping.items() if v is None]
	print(f"\nSample unmapped teams (non-Big-5 clubs):")
	for t in sorted(unmapped_teams)[:20]:
		print(f"  {t}")

	# Save mapping
	output_path = MAPPINGS_DIR / "fbref_to_canonical.json"
	MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(mapping, f, indent="\t", ensure_ascii=False)
	print(f"\nSaved mapping to {output_path}")


if __name__ == "__main__":
	main()
