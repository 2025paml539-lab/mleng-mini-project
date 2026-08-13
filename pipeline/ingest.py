"""
pipeline/ingest.py
Loads raw train.csv and prints a quick profile — shape, dtypes, null counts.
Run: python pipeline/ingest.py
"""

import pandas as pd
import sys
import os

RAW_PATH = os.path.join("data", "raw", "train.csv")


def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[INGEST] ERROR — file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    return df


def profile(df: pd.DataFrame) -> None:
    print(f"[INGEST] Loaded     : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"[INGEST] Columns    : {list(df.columns)}")

    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls == 0:
        print(f"[INGEST] Nulls      : 0 — all columns complete ✓")
    else:
        print(f"[INGEST] Nulls      : {total_nulls} found")
        print(null_counts[null_counts > 0])

    print(f"[INGEST] trip_duration — min: {df['trip_duration'].min()}s  "
          f"max: {df['trip_duration'].max()}s  "
          f"mean: {df['trip_duration'].mean():.0f}s")
    print(f"[INGEST] Date range : {df['pickup_datetime'].min()} → {df['pickup_datetime'].max()}")


if __name__ == "__main__":
    df = load_data()
    profile(df)
    print("[INGEST] Done ✓")
