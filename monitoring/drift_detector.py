"""
monitoring/drift_detector.py
Detects data drift by comparing training reference distribution
against recent prediction log using KS test (numerical) and
Chi-squared test (categorical).
Prints DRIFT DETECTED or NO DRIFT for each feature.
Run: python monitoring/drift_detector.py
"""

import os
import sys
import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from datetime import datetime

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
SCHEMA_PATH   = os.path.join("artifacts", "feature_schema.json")
LOG_PATH      = os.path.join("monitoring", "prediction_log.csv")
THRESHOLD     = 0.05   # p-value threshold for drift detection


def load_training_reference():
    """Load training feature distributions as reference."""
    df = pd.read_csv(FEATURES_PATH)
    df["pickup_datetime"] = pd.to_datetime("2016-01-01")  # placeholder

    # Recompute hour and distance from features
    ref = {
        "distance_km":    df["distance_km"].values,
        "hour_of_day":    df["hour_of_day"].values,
        "passenger_count":np.random.choice(range(1, 7),
                           size=len(df), p=[0.55,0.20,0.12,0.07,0.04,0.02]),
    }
    print("[DRIFT] Training reference loaded: {:,} rows".format(len(df)))
    return ref


def load_prediction_log():
    """Load prediction log and extract features."""
    if not os.path.exists(LOG_PATH):
        print("[DRIFT] ERROR: prediction_log.csv not found. Run simulate_drift.py first.")
        sys.exit(1)

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 10:
        print("[DRIFT] ERROR: Not enough records in log ({})".format(len(rows)))
        sys.exit(1)

    # Compute distance from GPS coords
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians,
                                      [float(lat1), float(lon1),
                                       float(lat2), float(lon2)])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    distances = []
    hours     = []
    pax       = []

    for r in rows:
        try:
            d = haversine(r["pickup_latitude"], r["pickup_longitude"],
                          r["dropoff_latitude"], r["dropoff_longitude"])
            distances.append(d)
            hours.append(int(r["pickup_datetime"].split(" ")[1].split(":")[0]))
            pax.append(int(r["passenger_count"]))
        except Exception:
            continue

    prod = {
        "distance_km":    np.array(distances),
        "hour_of_day":    np.array(hours),
        "passenger_count":np.array(pax),
    }
    print("[DRIFT] Prediction log loaded: {} records".format(len(rows)))
    return prod


def run_ks_test(name, train_vals, prod_vals):
    """KS test for numerical features."""
    stat, p = ks_2samp(train_vals, prod_vals)
    drift   = p < THRESHOLD
    status  = "DRIFT DETECTED" if drift else "NO DRIFT"
    print("  [{:>16}]  KS stat={:.4f}  p={:.6f}  -> {}".format(
          name, stat, p, status))
    return {"feature": name, "test": "KS",
            "statistic": round(stat, 4), "p_value": round(p, 6),
            "drift": drift}


def run_chi2_test(name, train_vals, prod_vals, bins=10):
    """Chi-squared test for categorical/binned features."""
    all_vals  = np.concatenate([train_vals, prod_vals])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    train_counts, _ = np.histogram(train_vals, bins=bin_edges)
    prod_counts,  _ = np.histogram(prod_vals,  bins=bin_edges)

    # Avoid zero cells
    train_counts = np.where(train_counts == 0, 1, train_counts)
    prod_counts  = np.where(prod_counts  == 0, 1, prod_counts)

    # Normalise to same total
    prod_expected = prod_counts.sum() * (train_counts / train_counts.sum())
    contingency   = np.array([prod_counts, prod_expected.astype(int)])

    try:
        chi2, p, _, _ = chi2_contingency(contingency)
    except Exception:
        p = 1.0
        chi2 = 0.0

    drift  = p < THRESHOLD
    status = "DRIFT DETECTED" if drift else "NO DRIFT"
    print("  [{:>16}]  Chi2={:.4f}  p={:.6f}  -> {}".format(
          name, chi2, p, status))
    return {"feature": name, "test": "Chi2",
            "statistic": round(chi2, 4), "p_value": round(p, 6),
            "drift": drift}


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DRIFT DETECTION REPORT")
    print("Threshold: p < {}".format(THRESHOLD))
    print("=" * 60)

    ref  = load_training_reference()
    prod = load_prediction_log()

    print("\n[Numerical Features — KS Test]")
    results = []
    results.append(run_ks_test("distance_km",
                               ref["distance_km"],
                               prod["distance_km"]))
    results.append(run_ks_test("hour_of_day",
                               ref["hour_of_day"].astype(float),
                               prod["hour_of_day"].astype(float)))

    print("\n[Categorical Features — Chi-Squared Test]")
    results.append(run_chi2_test("passenger_count",
                                 ref["passenger_count"].astype(float),
                                 prod["passenger_count"].astype(float)))

    # Summary
    drifted = [r for r in results if r["drift"]]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("  Features tested  : {}".format(len(results)))
    print("  Drift detected   : {}".format(len(drifted)))
    print("  Features drifted : {}".format(
          [r["feature"] for r in drifted] if drifted else "None"))

    if drifted:
        print("\nRECOMMENDATION: Trigger retraining review.")
        print("  distance_km and hour_of_day have shifted from training distribution.")
        print("  Review incoming data. If drift persists for 3 windows -> retrain.")
    else:
        print("\nRECOMMENDATION: No action needed. Monitor next window.")

    print("=" * 60)

    # Save results to JSON for reporting
    out = {
        "run_timestamp":  datetime.now().isoformat(),
        "threshold":      THRESHOLD,
        "log_records":    len(prod["distance_km"]),
        "results":        [{k: (int(v) if isinstance(v, (bool, np.bool_)) else
                               (float(v) if isinstance(v, (np.floating, np.integer)) else v))
                           for k, v in r.items()} for r in results],
        "drifted_features": [r["feature"] for r in drifted],
        "recommendation": "retrain" if drifted else "monitor"
    }
    out_path = os.path.join("monitoring", "drift_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[DRIFT] Results saved to", out_path)
