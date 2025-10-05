# -*- coding: utf-8 -*-
"""
Robust production ClubElo fetcher.

- Stays inside ClubElo (no cross-site joins).
- Filters by country (reliable), not by league (often missing for RUS, etc.).
- If 'level' is present, takes top division (level == 1).
- If 'level' is missing, falls back to selecting the expected number of
  top-division teams per country by best available ordering (rank asc, then elo desc).
- Writes compact as-of snapshot and optional per-team histories to data/prod/eloscores.
- Safe to import & call from other scripts.

Example:
    from elo_prod import build_prod_elo
    paths = build_prod_elo(
        target_countries={"ENG","ESP","GER","ITA","FRA","RUS"},
        as_of=None,
        write_histories=False,
    )
    print(paths)
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import pandas as pd
from soccerdata import ClubElo


# --------------------------
# Paths / Defaults
# --------------------------
PROD_DIR = Path("data/prod/eloscores")
PROD_DIR.mkdir(parents=True, exist_ok=True)

# Your six target countries
DEFAULT_COUNTRIES = {"ENG", "ESP", "GER", "ITA", "FRA", "RUS"}

# Fallback top-division team counts when 'level' column is absent.
# Adjust if needed.
DEFAULT_TOP_COUNTS = {
    "ENG": 20,  # Premier League
    "ESP": 20,  # La Liga
    "GER": 18,  # Bundesliga
    "ITA": 20,  # Serie A
    "FRA": 18,  # Ligue 1 (since 2023-24)
    "RUS": 16,  # Russian Premier Liga
}


# --------------------------
# Utilities
# --------------------------
def _parse_as_of(as_of: Optional[str | datetime]) -> datetime:
    if as_of is None:
        return datetime.today()
    if isinstance(as_of, datetime):
        return as_of
    return datetime.strptime(as_of, "%Y-%m-%d")


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _select_top_division_like(
    df: pd.DataFrame,
    target_countries: Iterable[str],
    top_counts: Dict[str, int],
) -> pd.DataFrame:
    """
    Select a plausible top-division cohort per country when 'level' is missing.
    Strategy: within each country select N teams using:
      1) ascending rank if 'rank' exists (1 is best),
      2) else descending elo if 'elo' exists,
      3) else stable name sort as last resort.
    """
    need_cols = {"team", "country"}
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"ClubElo table is missing required columns: {missing}")

    keep = df[df["country"].isin(set(target_countries))].copy()
    if keep.empty:
        raise RuntimeError(
            f"No rows after country filter. Available countries: "
            f"{sorted(df.get('country', pd.Series(dtype=str)).dropna().unique().tolist())}"
        )

    # Prepare sort keys
    if "rank" in keep.columns:
        keep["rank"] = _safe_numeric(keep["rank"])
    if "elo" in keep.columns:
        keep["elo"] = _safe_numeric(keep["elo"])

    out_parts: List[pd.DataFrame] = []
    for cc in sorted(set(target_countries)):
        block = keep[keep["country"] == cc].copy()
        if block.empty:
            continue

        # Primary: rank asc
        if "rank" in block.columns and block["rank"].notna().any():
            block = block.sort_values(["rank", "elo"], ascending=[True, False], na_position="last")
        # Secondary: elo desc
        elif "elo" in block.columns and block["elo"].notna().any():
            block = block.sort_values(["elo"], ascending=[False], na_position="last")
        # Tertiary: name
        else:
            block = block.sort_values(["team"])

        n = int(top_counts.get(cc, 20))
        out_parts.append(block.head(n))

    if not out_parts:
        raise RuntimeError("Could not form any country blocks under top-division fallback.")

    return pd.concat(out_parts, ignore_index=True)


def _pick_top_division(
    table: pd.DataFrame,
    target_countries: Iterable[str],
    top_counts: Dict[str, int],
) -> pd.DataFrame:
    """
    Prefer 'level == 1' if available; otherwise fall back to heuristic selection.
    """
    # Basic guard
    if "country" not in table.columns:
        raise RuntimeError("ClubElo table lacks 'country' column—unexpected for ClubElo.")
    if "team" not in table.columns:
        # read_by_date puts name in index in some versions; reset_index in caller should handle this.
        raise RuntimeError("ClubElo table lacks 'team' column after reset_index().")

    # Filter to target countries first
    base = table[table["country"].isin(set(target_countries))].copy()
    if base.empty:
        raise RuntimeError(
            f"No rows for target countries {sorted(target_countries)}. "
            f"Found countries: {sorted(table['country'].dropna().unique().tolist())}"
        )

    # If 'level' exists and not entirely NA, keep level == 1
    if "level" in base.columns and base["level"].notna().any():
        return base[base["level"] == 1].copy()

    # Else, fallback by expected team counts per country
    return _select_top_division_like(base, target_countries, top_counts)


def _prepare_asof_df(sel: pd.DataFrame, as_of_dt: datetime) -> pd.DataFrame:
    # Lean schema and consistent names
    cols = [c for c in ["team", "country", "rank", "elo", "from", "to"] if c in sel.columns]
    out = sel[cols].copy()
    out = out.rename(columns={"team": "team_clubelo"})
    # Datetime columns may be object; coerce
    for c in ("from", "to"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    if "rank" in out.columns:
        out["rank"] = _safe_numeric(out["rank"])
    if "elo" in out.columns:
        out["elo"] = _safe_numeric(out["elo"])
    out["as_of_date"] = as_of_dt.date().isoformat()
    return out.sort_values(["country", "team_clubelo"]).reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)


# --------------------------
# Public API
# --------------------------
def build_prod_elo(
    target_countries: Iterable[str] = DEFAULT_COUNTRIES,
    as_of: Optional[str | datetime] = None,
    write_histories: bool = False,
    out_dir: Path = PROD_DIR,
    top_counts_override: Optional[Dict[str, int]] = None,
) -> Dict[str, str]:
    """
    Build minimal Elo payload for production.

    Args:
        target_countries: iterable of 3-letter country codes (ClubElo style), e.g., {"ENG","ESP","GER","ITA","FRA","RUS"}.
        as_of: None for today or 'YYYY-MM-DD' or datetime.
        write_histories: if True, also persist per-team full Elo histories for selected teams.
        out_dir: output directory (default data/prod/eloscores).
        top_counts_override: optional per-country dict of expected top-flight team counts when 'level' is missing.

    Returns:
        dict with keys: 'elo_asof', 'roster', and optionally 'elo_history'.
    """
    as_of_dt = _parse_as_of(as_of)
    out_dir.mkdir(parents=True, exist_ok=True)

    ce = ClubElo()
    table = ce.read_by_date(as_of_dt).reset_index()  # ensure 'team' exists
    if "team" not in table.columns:
        raise RuntimeError("ClubElo.read_by_date(...).reset_index() did not expose 'team' column.")

    # Selection (robust to missing 'league')
    top_counts = dict(DEFAULT_TOP_COUNTS)
    if top_counts_override:
        top_counts.update(top_counts_override)

    sel = _pick_top_division(table, target_countries, top_counts)
    if sel.empty:
        raise RuntimeError("Selection produced an empty set; check countries and counts.")

    # Prepare stable as-of dataset
    asof_df = _prepare_asof_df(sel, as_of_dt)

    # Write as-of parquet
    asof_path = out_dir / f"elo_asof_{as_of_dt.date().isoformat()}.parquet"
    paths: Dict[str, str] = {"elo_asof": _write_parquet(asof_df, asof_path)}

    # Write a compact roster for auditing; 'league' may be missing—use a group label
    roster = asof_df[["country", "team_clubelo"]].copy()
    roster["group_label"] = roster["country"] + "-Top"
    roster_path = out_dir / f"roster_{as_of_dt.date().isoformat()}.parquet"
    paths["roster"] = _write_parquet(roster, roster_path)

    # Optional histories only for the selected teams
    if write_histories:
        parts: List[pd.DataFrame] = []
        for team in asof_df["team_clubelo"].unique().tolist():
            hist = ce.read_team_history(team, max_age=1)
            # keep minimal schema
            keep = [c for c in ["from", "to", "elo"] if c in hist.columns]
            if not keep:
                continue
            h = hist[keep].copy()
            for c in ("from", "to"):
                if c in h.columns:
                    h[c] = pd.to_datetime(h[c], errors="coerce")
            h["team_clubelo"] = team
            # bring country (stable as of selection)
            country = asof_df.loc[asof_df["team_clubelo"] == team, "country"].iloc[0]
            h["country"] = country
            parts.append(h)

        if parts:
            elo_hist = pd.concat(parts, ignore_index=True)
            elo_hist = elo_hist[["team_clubelo", "country", "from", "to", "elo"]]
            hist_path = out_dir / f"elo_history_selected_{as_of_dt.date().isoformat()}.parquet"
            paths["elo_history"] = _write_parquet(elo_hist, hist_path)

    return paths


# --------------------------
# CLI entry (optional)
# --------------------------
if __name__ == "__main__":
    # Default run: today, default 6 countries, no histories
    out = build_prod_elo(write_histories=False)
    print(out)