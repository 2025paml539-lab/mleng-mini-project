"""
pipeline/features.py
Engineers 5 features from raw trip data. Saves processed dataset and
feature_schema.json (used identically at serving time — prevents skew).
Run: python pipeline/features.py
"""

import pandas as pd
import numpy as np
import json
import os
import sys

RAW_PATH      = os.path.join("data", "raw", "train.csv")
PROCESSED_DIR = os.path.join("data", "processed")
PROCESSED_PATH = os.path.join(PROCESSED_DIR, "features.csv")
SCHEMA_PATH   = os.path.join("artifacts", "feature_schema.json")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("artifacts", exist_ok=True)


def haversine_km(lat1: pd.Series, lon1: pd.Series,
                 lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Haversine distance in km between two GPS points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def hour_bin(hour: int) -> str:
    if 0 <= hour < 6:   return "night"
    if 6 <= hour < 12:  return "morning"
    if 12 <= hour < 18: return "afternoon"
    return "evening"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[FEATURES] Engineering features...")

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # 1. hour_of_day
    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    print(f"  hour_of_day    : range {df['hour_of_day'].min()}–{df['hour_of_day'].max()} ✓")

    # 2. day_of_week (0=Mon, 6=Sun)
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    print(f"  day_of_week    : range {df['day_of_week'].min()}–{df['day_of_week'].max()} ✓")

    # 3. is_weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    print(f"  is_weekend     : {df['is_weekend'].sum():,} weekend trips ✓")

    # 4. distance_km (Haversine)
    df["distance_km"] = haversine_km(
        df["pickup_latitude"],  df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    ).round(4)
    print(f"  distance_km    : mean={df['distance_km'].mean():.2f}km  "
          f"max={df['distance_km'].max():.2f}km ✓")

    # 5. pickup_hour_bin
    df["pickup_hour_bin"] = df["hour_of_day"].apply(hour_bin)
    print(f"  pickup_hour_bin: {df['pickup_hour_bin'].value_counts().to_dict()} ✓")

    return df


def save_schema(df: pd.DataFrame) -> None:
    """Save transformation params — reused at serving time."""
    schema = {
        "distance_km_mean": round(float(df["distance_km"].mean()), 4),
        "distance_km_std":  round(float(df["distance_km"].std()),  4),
        "distance_km_min":  round(float(df["distance_km"].min()),  4),
        "distance_km_max":  round(float(df["distance_km"].max()),  4),
        "hour_bins":        {"night": "0-5", "morning": "6-11",
                             "afternoon": "12-17", "evening": "18-23"},
        "feature_columns":  ["hour_of_day", "day_of_week", "is_weekend",
                             "distance_km", "pickup_hour_bin", "trip_duration"]
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"  Schema saved   : {SCHEMA_PATH} ✓")


def main() -> None:
    if not os.path.exists(RAW_PATH):
        print(f"[FEATURES] ERROR — {RAW_PATH} not found. Run ingest first.")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    input_rows = len(df)

    df = engineer_features(df)

    # Keep only model-relevant columns — drop cols not available at serving time
    keep_cols = ["hour_of_day", "day_of_week", "is_weekend",
                 "distance_km", "pickup_hour_bin", "trip_duration"]
    df = df[keep_cols]

    df.to_csv(PROCESSED_PATH, index=False)
    save_schema(df)

    print(f"\n[FEATURES] Input : {input_rows:,} rows")
    print(f"[FEATURES] Output: {len(df):,} rows x {len(df.columns)} cols")
    print(f"[FEATURES] Saved : {PROCESSED_PATH}")
    print("[FEATURES] Done ✓")


if __name__ == "__main__":
    main()
