"""
Odds fetching module for The-Odds-API with daily caching.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from utils.paths import DATA_DIR, MAPPINGS_DIR

LEAGUE_TO_SPORT_KEY = {
	"ENG-Premier League": "soccer_epl",
	"FRA-Ligue 1": "soccer_france_ligue_one",
	"GER-Bundesliga": "soccer_germany_bundesliga",
	"ITA-Serie A": "soccer_italy_serie_a",
	"ESP-La Liga": "soccer_spain_la_liga",
}
SPORT_KEY_TO_LEAGUE = {value: key for key, value in LEAGUE_TO_SPORT_KEY.items()}
CACHE_DIR = DATA_DIR / "prod" / "odds"
TEAM_MAPPING_PATH = MAPPINGS_DIR / "theoddsapi_to_canonical.json"
PREFERRED_BOOKMAKER_KEYS = ("betsson", "williamhill")


def _load_team_mapping() -> dict:
	"""Load the team-name mapping file."""

	if not TEAM_MAPPING_PATH.exists():
		raise FileNotFoundError(f"Team mapping file not found: {TEAM_MAPPING_PATH}")
	with open(TEAM_MAPPING_PATH, "r", encoding="utf-8") as file:
		return json.load(file)


TEAM_MAPPING = _load_team_mapping()


def get_cache_path(sport_key: str, date_str: str) -> Path:
	"""Return the cache path for a sport and date."""

	return CACHE_DIR / f"{date_str}_{sport_key}.json"


def fetch_league_odds(sport_key: str, api_key: str) -> list[dict]:
	"""Fetch home/draw/away odds for one league."""

	url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
	params = {
		"apiKey": api_key,
		"regions": "eu",
		"markets": "h2h",
		"bookmakers": ",".join(PREFERRED_BOOKMAKER_KEYS),
	}
	response = requests.get(url, params=params, timeout=30)
	response.raise_for_status()
	return response.json()


def get_cached_or_fetch(sport_key: str, api_key: str | None, date_str: str) -> list[dict]:
	"""Get odds from cache or fetch them if needed."""

	cache_path = get_cache_path(sport_key, date_str)
	if cache_path.exists():
		print(f"  Using cached odds for {sport_key} from {cache_path.name}")
		with open(cache_path, "r", encoding="utf-8") as file:
			return json.load(file)
	if not api_key:
		raise RuntimeError(f"No cached odds found for {sport_key} on {date_str} and ODDS_API_KEY is not set.")
	print(f"  Fetching odds for {sport_key} from API...")
	data = fetch_league_odds(sport_key, api_key)
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	with open(cache_path, "w", encoding="utf-8") as file:
		json.dump(data, file, indent=2)
	print(f"  Saved {len(data)} games to {cache_path.name}")
	return data


def get_all_leagues_odds(api_key: str | None) -> list[dict]:
	"""Fetch odds for all supported leagues."""

	today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
	all_games = []
	for league_id, sport_key in LEAGUE_TO_SPORT_KEY.items():
		games = get_cached_or_fetch(sport_key, api_key, today_str)
		for game in games:
			game["league_id"] = league_id
		all_games.extend(games)
	return all_games


def parse_odds_data(games: list[dict]) -> list[dict]:
	"""Extract result odds from Betsson, falling back to William Hill."""

	parsed = []
	for game in games:
		home_team_raw = game["home_team"]
		away_team_raw = game["away_team"]
		commence_time = game["commence_time"]
		league_id = game.get("league_id", SPORT_KEY_TO_LEAGUE.get(game.get("sport_key", ""), ""))
		home_team = TEAM_MAPPING.get(home_team_raw, home_team_raw)
		away_team = TEAM_MAPPING.get(away_team_raw, away_team_raw)

		selected_bookmaker = None
		bookmakers = {bookmaker.get("key"): bookmaker for bookmaker in game.get("bookmakers", [])}
		for bookmaker_key in PREFERRED_BOOKMAKER_KEYS:
			bookmaker = bookmakers.get(bookmaker_key)
			if bookmaker is not None:
				selected_bookmaker = bookmaker
				break

		if selected_bookmaker is None:
			continue

		selected_home = None
		selected_draw = None
		selected_away = None
		for market in selected_bookmaker.get("markets", []):
			if market.get("key") != "h2h":
				continue
			for outcome in market.get("outcomes", []):
				price = outcome.get("price")
				if price is None:
					continue
				if outcome["name"] == home_team_raw:
					selected_home = price
				elif outcome["name"] == away_team_raw:
					selected_away = price
				elif outcome["name"].lower() == "draw":
					selected_draw = price

		if all(value is not None for value in [selected_home, selected_draw, selected_away]):
			parsed.append({
				"home_team": home_team,
				"away_team": away_team,
				"home_team_raw": home_team_raw,
				"away_team_raw": away_team_raw,
				"league_id": league_id,
				"commence_time": commence_time,
				"odds_home": selected_home,
				"odds_draw": selected_draw,
				"odds_away": selected_away,
			})
	return parsed
