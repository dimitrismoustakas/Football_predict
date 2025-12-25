"""
Odds fetching module for The-Odds-API with daily caching.
"""
import json
import requests
from pathlib import Path
from datetime import datetime, timezone

# League mapping: Understat league ID -> The-Odds-API sport key
LEAGUE_TO_SPORT_KEY = {
	"ENG-Premier League": "soccer_epl",
	"FRA-Ligue 1": "soccer_france_ligue_one",
	"GER-Bundesliga": "soccer_germany_bundesliga",
	"ITA-Serie A": "soccer_italy_serie_a",
	"ESP-La Liga": "soccer_spain_la_liga",
}

# Reverse mapping for convenience
SPORT_KEY_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_SPORT_KEY.items()}

# Cache directory
CACHE_DIR = Path("data/prod/odds")

# Team name mapping file
MAPPINGS_DIR = Path("data/mappings")
TEAM_MAPPING_PATH = MAPPINGS_DIR / "theoddsapi_to_canonical.json"

# Load team mapping at module level
def _load_team_mapping() -> dict:
	"""Load team name mapping from JSON file."""
	if not TEAM_MAPPING_PATH.exists():
		raise FileNotFoundError(f"Team mapping file not found: {TEAM_MAPPING_PATH}")
	with open(TEAM_MAPPING_PATH, "r", encoding="utf-8") as f:
		return json.load(f)

TEAM_MAPPING = _load_team_mapping()


def get_cache_path(sport_key: str, date_str: str) -> Path:
	"""Get the cache file path for a given sport key and date."""
	return CACHE_DIR / f"{date_str}_{sport_key}.json"


def fetch_league_odds(sport_key: str, api_key: str) -> list[dict]:
	"""
	Fetch odds for a single league from The-Odds-API.
	
	Args:
		sport_key: The-Odds-API sport identifier (e.g., "soccer_epl")
		api_key: API key for The-Odds-API
	
	Returns:
		List of game dicts from the API response
	
	Raises:
		requests.HTTPError: If API call fails
	"""
	url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
	params = {
		"apiKey": api_key,
		"regions": "eu",
		"markets": "h2h,totals",
	}
	
	response = requests.get(url, params=params, timeout=30)
	response.raise_for_status()
	
	return response.json()


def get_cached_or_fetch(sport_key: str, api_key: str, date_str: str) -> list[dict]:
	"""
	Get odds from cache if available for today, otherwise fetch from API.
	
	Args:
		sport_key: The-Odds-API sport identifier
		api_key: API key for The-Odds-API
		date_str: Date string in YYYY-MM-DD format
	
	Returns:
		List of game dicts
	"""
	cache_path = get_cache_path(sport_key, date_str)
	
	# Check if cache exists for today
	if cache_path.exists():
		print(f"  Using cached odds for {sport_key} from {cache_path.name}")
		with open(cache_path, "r", encoding="utf-8") as f:
			return json.load(f)
	
	# Fetch from API
	print(f"  Fetching odds for {sport_key} from API...")
	data = fetch_league_odds(sport_key, api_key)
	
	# Save to cache
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	with open(cache_path, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=2)
	print(f"  Saved {len(data)} games to {cache_path.name}")
	
	return data


def get_all_leagues_odds(api_key: str) -> list[dict]:
	"""
	Fetch odds for all supported leagues, using cache when available.
	
	Args:
		api_key: API key for The-Odds-API
	
	Returns:
		Combined list of game dicts from all leagues
	"""
	today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
	all_games = []
	
	for league_id, sport_key in LEAGUE_TO_SPORT_KEY.items():
		games = get_cached_or_fetch(sport_key, api_key, today_str)
		# Tag each game with the league ID for later filtering
		for game in games:
			game["league_id"] = league_id
		all_games.extend(games)
	
	return all_games


def parse_odds_data(games: list[dict]) -> list[dict]:
	"""
	Parse raw API response into structured odds data.
	Extracts best over/under 2.5 odds across all bookmakers.
	
	Args:
		games: List of game dicts from The-Odds-API
	
	Returns:
		List of dicts with: home_team, away_team, league_id, commence_time, odds_over, odds_under
	"""
	parsed = []
	
	for game in games:
		home_team_raw = game["home_team"]
		away_team_raw = game["away_team"]
		commence_time = game["commence_time"]
		league_id = game.get("league_id", SPORT_KEY_TO_LEAGUE.get(game.get("sport_key", ""), ""))
		
		# Map team names to canonical names
		home_team = TEAM_MAPPING.get(home_team_raw, home_team_raw)
		away_team = TEAM_MAPPING.get(away_team_raw, away_team_raw)
		
		# Find best over/under odds across all bookmakers
		best_over = None
		best_under = None
		
		for bookmaker in game.get("bookmakers", []):
			for market in bookmaker.get("markets", []):
				if market["key"] == "totals":
					for outcome in market["outcomes"]:
						if outcome.get("point") == 2.5:
							if outcome["name"] == "Over":
								if best_over is None or outcome["price"] > best_over:
									best_over = outcome["price"]
							elif outcome["name"] == "Under":
								if best_under is None or outcome["price"] > best_under:
									best_under = outcome["price"]
		
		if best_over is not None and best_under is not None:
			parsed.append({
				"home_team": home_team,
				"away_team": away_team,
				"home_team_raw": home_team_raw,
				"away_team_raw": away_team_raw,
				"league_id": league_id,
				"commence_time": commence_time,
				"odds_over": best_over,
				"odds_under": best_under,
			})
	
	return parsed
