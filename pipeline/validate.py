"""
pipeline/validate.py
4-level data validation. Prints [PASS] or [WARN: n rows] for outliers.
Outliers are flagged and removed — pipeline continues with clean data.
Schema failures (L1) are hard stops.
Run: python pipeline/validate.py
"""

import pandas as pd
import sys
import os

RAW_PATH = os.path.join("data", "raw", "train.csv")

EXPECTED_COLUMNS = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime",
    "passenger_count", "pickup_longitude", "pickup_latitude",
    "dropoff_longitude", "dropoff_latitude", "store_and_fwd_flag", "trip_duration"
]

# NYC bounding box
LAT_MIN, LAT_MAX = 40.4, 41.0
LON_MIN, LON_MAX = -74.3, -73.6

schema_failed = False


def check_hard(label: str, condition: bool, detail: str = "") -> None:
    """Hard failure — stops pipeline."""
    global schema_failed
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{' — ' + detail if detail else ''}")
        schema_failed = True


def check_soft(label: str, n_invalid: int, total: int) -> None:
    """Soft check — warns and reports count, pipeline continues."""
    if n_invalid == 0:
        print(f"  [PASS] {label}")
    else:
        pct = round(100 * n_invalid / total, 3)
        print(f"  [WARN] {label} — {n_invalid} rows ({pct}%) removed as outliers")


def run_validation(path: str = RAW_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[VALIDATE] ERROR — file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path, parse_dates=["pickup_datetime", "dropoff_datetime"])
    original_len = len(df)

    # L1 — Schema (hard stop on failure)
    print("\n[L1 SCHEMA]")
    check_hard("All 11 columns present", list(df.columns) == EXPECTED_COLUMNS)
    check_hard("pickup_datetime parsed as datetime",
               pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]))
    check_hard("trip_duration is integer",
               pd.api.types.is_integer_dtype(df["trip_duration"]))
    check_hard("No null values in any column", df.isnull().sum().sum() == 0,
               f"{df.isnull().sum().sum()} nulls found")

    if schema_failed:
        print("[VALIDATE] Schema check FAILED — cannot continue.")
        sys.exit(1)

    # L2 — Range & Domain (soft — remove outliers)
    print("\n[L2 RANGE & DOMAIN]")
    mask = (
        df["passenger_count"].between(1, 6) &
        df["pickup_latitude"].between(LAT_MIN, LAT_MAX) &
        df["pickup_longitude"].between(LON_MIN, LON_MAX) &
        df["dropoff_latitude"].between(LAT_MIN, LAT_MAX) &
        df["dropoff_longitude"].between(LON_MIN, LON_MAX) &
        df["trip_duration"].between(1, 86400)
    )
    check_soft("passenger_count 1–6",
               (~df["passenger_count"].between(1, 6)).sum(), original_len)
    check_soft("Coordinates within NYC bounds",
               (~(df["pickup_latitude"].between(LAT_MIN, LAT_MAX) &
                  df["pickup_longitude"].between(LON_MIN, LON_MAX))).sum(), original_len)
    check_soft("trip_duration 1s–86400s",
               (~df["trip_duration"].between(1, 86400)).sum(), original_len)

    df = df[mask].reset_index(drop=True)

    # L3 — Statistical
    print("\n[L3 STATISTICAL]")
    check_hard("Row count > 1,000,000 after cleaning", len(df) > 1_000_000,
               f"got {len(df):,}")
    mean_dur = df["trip_duration"].mean()
    check_hard("Mean trip_duration between 300s–1500s",
               300 <= mean_dur <= 1500, f"got {mean_dur:.0f}s")

    # L4 — Business Rules
    print("\n[L4 BUSINESS RULES]")
    invalid_times = (df["dropoff_datetime"] <= df["pickup_datetime"]).sum()
    check_soft("dropoff_datetime > pickup_datetime", invalid_times, len(df))
    df = df[df["dropoff_datetime"] > df["pickup_datetime"]].reset_index(drop=True)

    removed = original_len - len(df)
    print(f"\n[VALIDATE] Input : {original_len:,} rows")
    print(f"[VALIDATE] Removed: {removed} outlier rows ({round(100*removed/original_len,3)}%)")
    print(f"[VALIDATE] Output : {len(df):,} clean rows")
    print("[VALIDATE] All checks passed ✓")
    return df


if __name__ == "__main__":
    run_validation()
