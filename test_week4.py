"""
test_week4.py - Unit tests for Week 4 monitoring pipeline.
Run: python test_week4.py
"""
import os, sys, csv, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))
        failed += 1


print("=" * 55)
print("Week 4 Unit Tests")
print("=" * 55)

# T1 - Monitoring files exist
print("\n[T1] Monitoring files")
for f in ["monitoring/logger.py", "monitoring/simulate_drift.py",
          "monitoring/drift_detector.py", "monitoring/__init__.py"]:
    check(f"{f} exists", os.path.exists(f))

# T2 - Logger writes correct CSV structure
print("\n[T2] Logger functionality")
from monitoring.logger import log_prediction, get_log

test_log = "monitoring/test_log.csv"
if os.path.exists(test_log):
    os.remove(test_log)

import monitoring.logger as logger_module
original_path = logger_module.LOG_PATH
logger_module.LOG_PATH = test_log

log_prediction(
    {"pickup_datetime": "2016-06-30 23:59:58",
     "pickup_latitude": 40.767937, "pickup_longitude": -73.982155,
     "dropoff_latitude": 40.765602, "dropoff_longitude": -73.964630,
     "passenger_count": 1},
    427.84, 7.13, "v1.0-week3"
)

logger_module.LOG_PATH = original_path
with open(test_log, "r") as f:
    rows = list(csv.DictReader(f))
os.remove(test_log)

check("logger writes 1 row",              len(rows) == 1, f"got {len(rows)}")
check("timestamp present",                "timestamp" in rows[0])
check("predicted_duration_seconds=427.84",rows[0]["predicted_duration_seconds"] == "427.84")
check("model_version logged",             rows[0]["model_version"] == "v1.0-week3")

# T3 - Prediction log exists with correct structure
print("\n[T3] Prediction log")
log_path = "monitoring/prediction_log.csv"
check("prediction_log.csv exists", os.path.exists(log_path))
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        log_rows = list(csv.DictReader(f))
    check("log has >= 600 records",       len(log_rows) >= 600,
          f"got {len(log_rows)}")
    check("all expected columns present", all(
        col in log_rows[0] for col in
        ["timestamp","pickup_datetime","passenger_count",
         "predicted_duration_seconds","model_version"]
    ))
    pax_vals = [int(r["passenger_count"]) for r in log_rows]
    check("passenger_count 1-6 in log",   all(1 <= p <= 6 for p in pax_vals))

# T4 - Drift results JSON exists and has correct structure
print("\n[T4] Drift detection results")
drift_path = "monitoring/drift_results.json"
check("drift_results.json exists", os.path.exists(drift_path))
if os.path.exists(drift_path):
    with open(drift_path) as f:
        dr = json.load(f)
    check("results has 3 features",       len(dr["results"]) == 3)
    check("drifted_features not empty",   len(dr["drifted_features"]) > 0)
    check("recommendation is retrain",    dr["recommendation"] == "retrain")
    check("distance_km drift detected",
          any(r["feature"] == "distance_km" and r["drift"]
              for r in dr["results"]))
    check("hour_of_day drift detected",
          any(r["feature"] == "hour_of_day" and r["drift"]
              for r in dr["results"]))

# T5 - KS test gives correct direction
print("\n[T5] KS test statistical correctness")
from scipy.stats import ks_2samp
import pandas as pd

feat = pd.read_csv("data/processed/features.csv")
train_dist = feat["distance_km"].values

# Two samples from same training distribution should have p > 0.05
np.random.seed(42)
sample_a = feat["distance_km"].sample(1000, random_state=1).values
sample_b = feat["distance_km"].sample(1000, random_state=2).values
_, p_same = ks_2samp(sample_a, sample_b)
check("two training samples: no drift (p > 0.05)", p_same > 0.05,
      f"got p={p_same:.4f}")

# Drifted sample SHOULD drift
drifted_sample = np.random.normal(12.5, 4.0, 1000)
_, p_drifted = ks_2samp(sample_a, drifted_sample)
check("drifted sample: drift detected (p < 0.05)", p_drifted < 0.05,
      f"got p={p_drifted:.6f}")

# T6 - Report files exist
print("\n[T6] Report files")
check("reports/drift_report.md exists",
      os.path.exists("reports/drift_report.md"))
check("outputs/week4/01_distance_drift.png exists",
      os.path.exists("outputs/week4/01_distance_drift.png"))
check("outputs/week4/02_hour_drift.png exists",
      os.path.exists("outputs/week4/02_hour_drift.png"))
check("outputs/week4/03_drift_detection_summary.png exists",
      os.path.exists("outputs/week4/03_drift_detection_summary.png"))

print("\n" + "=" * 55)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED - Week 4 is fully working")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
