import polars as pl
from pathlib import Path

# Load the generated features file
FEATURES_PARQUET = Path("data/prod/features_season.parquet")

print(f"--- Inspecting Features from {FEATURES_PARQUET} ---")

try:
    df = pl.read_parquet(FEATURES_PARQUET)

    print("\n--- Overall R5 Features ---")

    # Select r5 overall features for home and away teams
    r5_cols_h = [c for c in df.columns if c.startswith('ovr__') and c.endswith('__r5__h')]
    r5_cols_a = [c for c in df.columns if c.startswith('ovr__') and c.endswith('__r5__a')]
    r5_cols = r5_cols_h + r5_cols_a

    if r5_cols:
        print("\n--- First 5 Rows of R5 Overall Features ---")
        print(df.select(['home_team', 'away_team'] + r5_cols).head())

        print("\n--- Summary Statistics for R5 Overall Features ---")
        print(df.select(r5_cols).describe())

        print("\n--- Null Counts for R5 Overall Features ---")
        print(df.select(r5_cols).select(pl.all().is_null().sum()).to_pandas().transpose())
    else:
        print("No R5 overall features found in the data.")

except FileNotFoundError:
    print(f"ERROR: The file {FEATURES_PARQUET} was not found.")
    print("Please run the main.py script first to generate the features.")