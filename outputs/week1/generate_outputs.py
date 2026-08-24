"""
outputs/week1/generate_outputs.py
Week 1 visual outputs - validation results, feature distributions, GPS heatmap.
Run from project root: python outputs/week1/generate_outputs.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

RAW_PATH      = os.path.join("data", "raw", "train.csv")
FEATURES_PATH = os.path.join("data", "processed", "features.csv")
SCHEMA_PATH   = os.path.join("artifacts", "feature_schema.json")
OUT_DIR       = os.path.join("outputs", "week1")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
raw  = pd.read_csv(RAW_PATH, parse_dates=["pickup_datetime", "dropoff_datetime"])
feat = pd.read_csv(FEATURES_PATH)
print(f"Raw: {len(raw):,} rows | Features: {feat.shape}")


# Plot 1 - Validation summary
def plot_validation_summary():
    fig, ax = plt.subplots(figsize=(9, 4))
    checks = [
        "L1 Schema: all 11 cols, types, 0 nulls",
        "L2 Range: passenger, GPS bounds, duration",
        "L3 Statistical: row count, mean duration",
        "L4 Business: dropoff > pickup"
    ]
    results = ["PASS", "WARN (689 rows removed)", "PASS", "PASS"]
    colors  = ["green" if r == "PASS" else "orange" for r in results]

    bars = ax.barh(checks, [1, 1, 1, 1], color=colors, height=0.5, edgecolor="gray")
    for bar, res in zip(bars, results):
        ax.text(0.02, bar.get_y() + bar.get_height()/2,
                res, va="center", color="white", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 1.4)
    ax.set_xticks([])
    ax.set_title("Week 1 - Data Validation Results (4-Level Check)")
    ax.text(0.99, -0.15,
            "Input: 1,458,644 rows | Removed: 689 outliers (0.047%) | Clean: 1,457,955",
            transform=ax.transAxes, ha="right", fontsize=8, color="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_validation_summary.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 01_validation_summary.png")


# Plot 2 - Trip duration distribution
def plot_trip_duration():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    dur       = feat["trip_duration"]
    dur_clean = dur[dur < dur.quantile(0.99)]

    axes[0].hist(dur_clean / 60, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(dur.mean() / 60, color="red", linestyle="--",
                    label=f"Mean: {dur.mean()/60:.1f} min")
    axes[0].axvline(dur.median() / 60, color="orange", linestyle="--",
                    label=f"Median: {dur.median()/60:.1f} min")
    axes[0].set_xlabel("Trip Duration (minutes)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Trip Duration Distribution (99th pct)")
    axes[0].legend()

    axes[1].hist(np.log1p(dur), bins=60, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("log(1 + trip_duration)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log-Transformed Duration (used as model target)")

    plt.suptitle("Week 1 - Trip Duration Analysis", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_trip_duration_distribution.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 02_trip_duration_distribution.png")


# Plot 3 - Engineered features overview
def plot_features_overview():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # trips by hour
    hc = feat["hour_of_day"].value_counts().sort_index()
    axes[0].bar(hc.index, hc.values, color="steelblue")
    axes[0].set_title("Trips by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Count")

    # trips by day of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dc   = feat["day_of_week"].value_counts().sort_index()
    bar_cols = ["salmon" if i >= 5 else "steelblue" for i in range(7)]
    axes[1].bar(days, dc.values, color=bar_cols)
    axes[1].set_title("Trips by Day (red = weekend)")
    axes[1].set_xlabel("Day")
    axes[1].set_ylabel("Count")

    # distance distribution
    dist = feat["distance_km"][feat["distance_km"] < feat["distance_km"].quantile(0.99)]
    axes[2].hist(dist, bins=60, color="darkorange", edgecolor="white")
    axes[2].set_title(f"Distance (km) - mean={feat['distance_km'].mean():.2f} km")
    axes[2].set_xlabel("Distance (km)")
    axes[2].set_ylabel("Count")

    # hour bin
    bin_order = ["morning", "afternoon", "evening", "night"]
    bin_vals  = [feat["pickup_hour_bin"].value_counts().get(b, 0) for b in bin_order]
    axes[3].bar(bin_order, bin_vals, color="mediumpurple")
    axes[3].set_title("Trips by Hour Bin")
    axes[3].set_xlabel("Bin")
    axes[3].set_ylabel("Count")

    # weekend split
    wk = feat["is_weekend"].value_counts()
    axes[4].pie([wk.get(0, 0), wk.get(1, 0)],
                labels=["Weekday", "Weekend"],
                autopct="%1.1f%%",
                colors=["steelblue", "salmon"])
    axes[4].set_title("Weekend vs Weekday")

    # distance vs duration scatter
    s = feat.sample(3000, random_state=42)
    axes[5].scatter(s["distance_km"], s["trip_duration"] / 60,
                    alpha=0.3, s=5, color="teal")
    axes[5].set_xlim(0, 25)
    axes[5].set_ylim(0, 90)
    axes[5].set_title("Distance vs Duration (3k sample)")
    axes[5].set_xlabel("Distance (km)")
    axes[5].set_ylabel("Duration (min)")

    plt.suptitle("Week 1 - Engineered Features Overview", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_engineered_features_overview.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 03_engineered_features_overview.png")


# Plot 4 - NYC pickup heatmap
def plot_gps_heatmap():
    sample = raw[
        raw["pickup_latitude"].between(40.6, 40.9) &
        raw["pickup_longitude"].between(-74.05, -73.75)
    ].sample(40000, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.hist2d(sample["pickup_longitude"], sample["pickup_latitude"],
              bins=180, cmap="hot")
    ax.set_title("Week 1 - NYC Pickup Locations (40k sample)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_nyc_pickup_heatmap.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 04_nyc_pickup_heatmap.png")


if __name__ == "__main__":
    plot_validation_summary()
    plot_trip_duration()
    plot_features_overview()
    plot_gps_heatmap()
    print("\nAll Week 1 outputs saved to", OUT_DIR)
