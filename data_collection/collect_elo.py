"""Fill required historical Elo dates and fetch current ratings from clubelo.com."""

import json
import re
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from utils.paths import DATA_DIR, MAPPINGS_DIR

ELO_HISTORY_PATH = DATA_DIR / "eloscores" / "elo_history.parquet"
TEAM_UNIVERSE_PATH = DATA_DIR / "eloscores" / "team_universe.parquet"
UNDERSTAT_GLOB = str(DATA_DIR / "understat" / "*" / "*" / "matches.parquet")
PAGE_SLUGS = {
	"Atalanta": "atalanta",
	"Atletico": "atletico",
	"Bilbao": "athletic-club",
	"Celta": "rc-celta",
	"Parma": "parma-calcio-1913",
	"Real Madrid": "realmadrid",
	"Saint-Etienne": "SaintEtienne",
	"Venezia": "venezia-fc",
}


def _club_slug(team: str) -> str:
	return PAGE_SLUGS.get(team, team.replace(" ", ""))


def _read_page(slug: str) -> str:
	url = f"https://clubelo.com/{quote(slug)}"
	with urlopen(url, timeout=20) as response:
		if response.geturl().rstrip("/") != url:
			raise ValueError(f"ClubElo redirected {url} to {response.geturl()}; no club data returned")
		return response.read().decode("utf-8")


def _vega_rows(page: str, fields: set[str]) -> list[dict]:
	match = re.search(r"\bvar\s+vegaJson\s*=\s*", page)
	if match is None:
		raise ValueError("ClubElo page has no vegaJson data")
	chart = json.JSONDecoder().raw_decode(page[match.end():].lstrip())[0]
	for rows in chart["datasets"].values():
		if rows and fields.issubset(rows[0]):
			return rows
	raise ValueError(f"ClubElo page has no rating data with fields {sorted(fields)}")


def fetch_team_history(team: str) -> pd.DataFrame:
	page = _read_page(_club_slug(team))
	points = pd.DataFrame(_vega_rows(page, {"Date", "Elo"}))
	points = points.rename(columns={"Date": "date", "Elo": "elo"})
	points["date"] = pd.to_datetime(points["date"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
	return points[["date", "elo"]].sort_values("date")


def fetch_current_elo(teams: pd.DataFrame) -> pl.DataFrame:
	current = []
	for country, group in teams.groupby("country", sort=True):
		page = _read_page(country)
		ratings = {row["Name"]: row["Elo"] for row in _vega_rows(page, {"Name", "Elo"})}
		names = {
			slug: unescape(name)
			for slug, name in re.findall(r'<a href="/([^"/]+)">([^<]+)', page)
		}
		for team in group["team"]:
			name = names.get(_club_slug(team))
			if name not in ratings:
				raise ValueError(f"ClubElo has no full-precision current rating for {team} in {country}")
			current.append({"team_clubelo": team, "elo": ratings[name]})
	return pl.DataFrame(current, schema={"team_clubelo": pl.String, "elo": pl.Float64})


def collect_elo(matches: pl.DataFrame, history_path: Path = ELO_HISTORY_PATH) -> pl.DataFrame:
	"""Keep saved ratings, append uncovered past dates, and return live future-fixture ratings."""
	history = pd.read_parquet(history_path)
	with open(MAPPINGS_DIR / "clubelo_to_canonical.json", encoding="utf-8") as file:
		club_names = {canonical: club for club, canonical in json.load(file).items()}
	frame = matches.select("league", "date", "home_team", "away_team").to_pandas()
	requests = pd.concat([
		frame[["league", "date", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
		for side in ("home", "away")
	], ignore_index=True)
	unmapped = sorted(set(requests["team"]) - club_names.keys())
	if unmapped:
		raise ValueError(f"Missing ClubElo mappings for: {', '.join(unmapped)}")
	requests["team"] = requests["team"].map(club_names)
	requests["country"] = requests["league"].str.split("-").str[0]
	requests["date"] = (
		pd.to_datetime(requests["date"], utc=True).dt.tz_localize(None)
		.astype("datetime64[ns]").dt.normalize() - pd.Timedelta(days=1)
	)
	requests = requests[["team", "country", "date"]].drop_duplicates()
	today = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).normalize()
	additions = []
	for team, wanted in requests[requests["date"] < today].groupby("team", sort=True):
		saved = history.loc[history["team"] == team, ["from", "to", "elo"]].sort_values("from")
		covered = pd.merge_asof(wanted.sort_values("date"), saved, left_on="date", right_on="from")
		missing = covered.loc[
			covered["elo"].isna() | (covered["date"] > covered["to"]), ["date", "country"]
		]
		if missing.empty:
			continue
		points = fetch_team_history(team)
		values = pd.merge_asof(missing, points, on="date", direction="backward")
		if values["elo"].isna().any():
			dates = values.loc[values["elo"].isna(), "date"].dt.strftime("%Y-%m-%d").tolist()
			raise ValueError(f"ClubElo's available history does not cover {team}: {', '.join(dates)}")
		values["team"] = team
		values["from"] = values["to"] = values["date"]
		additions.append(values[["team", "country", "from", "to", "elo"]])
		print(f"Fetched {len(values)} missing Elo dates for {team}")

	# Today's rating is still changing; never cache it as a completed historical day.
	live_teams = requests.loc[requests["date"] >= today, ["team", "country"]].drop_duplicates()
	current = fetch_current_elo(live_teams)
	if additions:
		updated = pd.concat([history, *additions], ignore_index=True).sort_values(["team", "from"])
		updated.to_parquet(history_path, index=False)
		print(f"Added {sum(len(rows) for rows in additions)} Elo observations to {history_path}")
	else:
		print("All required historical Elo dates are already saved.")
	return current


def main():
	matches = pl.scan_parquet(UNDERSTAT_GLOB).select("league", "date", "home_team", "away_team").collect()
	with open(MAPPINGS_DIR / "understat_to_canonical.json", encoding="utf-8") as file:
		mapping = json.load(file)
	matches = matches.with_columns(pl.col("home_team", "away_team").replace(mapping))
	collect_elo(matches)
	history = pd.read_parquet(ELO_HISTORY_PATH, columns=["team", "country"])
	history.drop_duplicates("team").sort_values("team").to_parquet(TEAM_UNIVERSE_PATH, index=False)


if __name__ == "__main__":
	main()
