"""
monitoring/simulate_drift.py
Simulates data drift by injecting two realistic drift scenarios:
  Scenario 1 - Rush-hour surge: longer distances, peak-hour timestamps
  Scenario 2 - Festival/holiday: short distances, unusual hours, high pax
Appends drifted records to monitoring/prediction_log.csv.
Run: python monitoring/simulate_drift.py
"""

import os
import sys
import random
import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from monitoring.logger import log_prediction

random.seed(42)
np.random.seed(42)

NORMAL_DISTANCE_MEAN = 3.44
NORMAL_DISTANCE_STD  = 3.10
NORMAL_HOUR_DIST_RAW = [2,1,1,1,2,3,6,7,6,5,5,5,6,6,6,6,6,7,7,6,5,5,4,3]
total = sum(NORMAL_HOUR_DIST_RAW)
NORMAL_HOUR_DIST = [v/total for v in NORMAL_HOUR_DIST_RAW]


def simulate_normal_batch(n=300):
    """Log n normal predictions matching training distribution."""
    print("[DRIFT] Logging {} normal predictions...".format(n))
    count = 0
    for _ in range(n):
        hour     = np.random.choice(range(24), p=NORMAL_HOUR_DIST)
        distance = max(0.5, np.random.normal(NORMAL_DISTANCE_MEAN,
                                             NORMAL_DISTANCE_STD))
        secs     = max(60, distance * 180 + np.random.normal(0, 60))
        mins     = round(secs / 60, 2)
        req = {
            "pickup_datetime":  "2016-06-{:02d} {:02d}:{:02d}:00".format(
                random.randint(1, 28), hour, random.randint(0, 59)),
            "pickup_latitude":  round(40.70 + random.uniform(0, 0.08), 6),
            "pickup_longitude": round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "dropoff_latitude": round(40.70 + random.uniform(0, 0.08), 6),
            "dropoff_longitude":round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "passenger_count":  random.randint(1, 3),
        }
        log_prediction(req, round(secs, 2), mins, "v1.0-week3")
        count += 1
    print("[DRIFT] {} normal records logged".format(count))


def simulate_rush_hour_surge(n=150):
    """
    Scenario 1 - Rush-hour surge.
    Inject trips with longer distances and peak-hour timestamps.
    Simulates a traffic surge event (e.g. city marathon, major concert).
    """
    print("[DRIFT] Scenario 1: Rush-hour surge ({} records)...".format(n))
    count = 0
    for _ in range(n):
        hour     = random.choice([7, 8, 9, 17, 18, 19])   # peak hours only
        distance = max(1.0, np.random.normal(12.5, 4.0))   # much longer trips
        secs     = max(300, distance * 400 + np.random.normal(0, 120))
        mins     = round(secs / 60, 2)
        req = {
            "pickup_datetime":  "2016-06-{:02d} {:02d}:{:02d}:00".format(
                random.randint(1, 28), hour, random.randint(0, 59)),
            "pickup_latitude":  round(40.70 + random.uniform(0, 0.08), 6),
            "pickup_longitude": round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "dropoff_latitude": round(40.70 + random.uniform(0, 0.08), 6),
            "dropoff_longitude":round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "passenger_count":  random.randint(1, 2),
        }
        log_prediction(req, round(secs, 2), mins, "v1.0-week3")
        count += 1
    print("[DRIFT] Scenario 1: {} rush-hour records injected".format(count))


def simulate_festival_pattern(n=150):
    """
    Scenario 2 - Festival/holiday pattern.
    Short distances, unusual hours (late night), high passenger counts.
    Simulates e.g. New Year's Eve or a major sports event.
    """
    print("[DRIFT] Scenario 2: Festival/holiday ({} records)...".format(n))
    count = 0
    for _ in range(n):
        hour     = random.choice([22, 23, 0, 1, 2])        # late night only
        distance = max(0.3, np.random.normal(1.2, 0.5))    # very short trips
        secs     = max(120, distance * 300 + np.random.normal(0, 60))
        mins     = round(secs / 60, 2)
        req = {
            "pickup_datetime":  "2016-06-{:02d} {:02d}:{:02d}:00".format(
                random.randint(1, 28), hour, random.randint(0, 59)),
            "pickup_latitude":  round(40.70 + random.uniform(0, 0.08), 6),
            "pickup_longitude": round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "dropoff_latitude": round(40.70 + random.uniform(0, 0.08), 6),
            "dropoff_longitude":round(-73.98 + random.uniform(-0.05, 0.05), 6),
            "passenger_count":  random.randint(3, 6),       # groups
        }
        log_prediction(req, round(secs, 2), mins, "v1.0-week3")
        count += 1
    print("[DRIFT] Scenario 2: {} festival records injected".format(count))


if __name__ == "__main__":
    from monitoring.logger import get_log

    # Clear old log first
    log_path = os.path.join("monitoring", "prediction_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)
        print("[DRIFT] Cleared old prediction log")

    simulate_normal_batch(300)
    simulate_rush_hour_surge(150)
    simulate_festival_pattern(150)

    log = get_log()
    print("\n[DRIFT] Total records in log: {}".format(len(log)))
    print("[DRIFT] prediction_log.csv ready for drift detection")
