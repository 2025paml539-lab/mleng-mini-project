"""
monitoring/logger.py
Logs every prediction to monitoring/prediction_log.csv.
Called by serving/api.py on every /predict request.
Run standalone: python monitoring/logger.py  (runs a demo log)
"""

import os
import csv
import json
from datetime import datetime

LOG_PATH = os.path.join("monitoring", "prediction_log.csv")

FIELDNAMES = [
    "timestamp", "model_version",
    "pickup_datetime", "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude", "passenger_count",
    "predicted_duration_seconds", "predicted_duration_minutes"
]


def log_prediction(request_dict: dict, prediction_seconds: float,
                   prediction_minutes: float, model_version: str) -> None:
    """Append one prediction record to the log CSV."""
    os.makedirs("monitoring", exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    row = {
        "timestamp":                   datetime.utcnow().isoformat(),
        "model_version":               model_version,
        "pickup_datetime":             request_dict.get("pickup_datetime", ""),
        "pickup_latitude":             request_dict.get("pickup_latitude", ""),
        "pickup_longitude":            request_dict.get("pickup_longitude", ""),
        "dropoff_latitude":            request_dict.get("dropoff_latitude", ""),
        "dropoff_longitude":           request_dict.get("dropoff_longitude", ""),
        "passenger_count":             request_dict.get("passenger_count", ""),
        "predicted_duration_seconds":  round(prediction_seconds, 2),
        "predicted_duration_minutes":  round(prediction_minutes, 2),
    }

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_log() -> list:
    """Load prediction log as list of dicts."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    # Demo — log 5 sample predictions
    import sys
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    samples = [
        ({"pickup_datetime":"2016-06-30 23:59:58","pickup_latitude":40.767937,
          "pickup_longitude":-73.982155,"dropoff_latitude":40.765602,
          "dropoff_longitude":-73.964630,"passenger_count":1}, 427.84, 7.13),
        ({"pickup_datetime":"2016-03-14 08:30:00","pickup_latitude":40.758896,
          "pickup_longitude":-73.985130,"dropoff_latitude":40.748817,
          "dropoff_longitude":-73.940271,"passenger_count":2}, 1101.86, 18.36),
        ({"pickup_datetime":"2016-01-16 14:20:00","pickup_latitude":40.752781,
          "pickup_longitude":-73.978271,"dropoff_latitude":40.731151,
          "dropoff_longitude":-73.921640,"passenger_count":3}, 1320.36, 22.01),
        ({"pickup_datetime":"2016-05-20 19:45:00","pickup_latitude":40.748817,
          "pickup_longitude":-74.002320,"dropoff_latitude":40.773994,
          "dropoff_longitude":-73.875612,"passenger_count":1}, 1803.75, 30.06),
        ({"pickup_datetime":"2016-04-10 12:00:00","pickup_latitude":40.755234,
          "pickup_longitude":-73.979012,"dropoff_latitude":40.742312,
          "dropoff_longitude":-73.935671,"passenger_count":2}, 980.50, 16.34),
    ]

    for req, secs, mins in samples:
        log_prediction(req, secs, mins, "v1.0-week3")

    log = get_log()
    print("[LOGGER] Logged {} predictions to {}".format(len(log), LOG_PATH))
    print("[LOGGER] Columns:", list(log[0].keys()) if log else "none")
    print("[LOGGER] Sample row:", json.dumps(log[-1], indent=2) if log else "none")
