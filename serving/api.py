"""
serving/api.py
FastAPI serving endpoint for NYC Taxi ETA prediction.
Loads model once at startup. Validates every request before prediction.
Run: uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from serving.schemas import PredictRequest, PredictResponse, HealthResponse

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger(__name__)

# paths
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_VERSION = "v1.0-week3"

# load artifacts once at startup
logger.info("Loading model artifacts...")
model  = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "feature_schema.json")) as f:
    schema = json.load(f)

logger.info("Model loaded successfully — %s", MODEL_VERSION)

app = FastAPI(
    title="NYC Taxi ETA Predictor",
    description="Predicts trip duration from pickup/dropoff GPS and time features.",
    version="1.0.0"
)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))


def hour_bin(hour):
    if 0 <= hour < 6:   return "night"
    if 6 <= hour < 12:  return "morning"
    if 12 <= hour < 18: return "afternoon"
    return "evening"


def build_features(req: PredictRequest) -> pd.DataFrame:
    dt = datetime.strptime(req.pickup_datetime, "%Y-%m-%d %H:%M:%S")
    hour      = dt.hour
    dow       = dt.weekday()
    is_weekend = 1 if dow >= 5 else 0
    distance  = round(haversine_km(
        req.pickup_latitude, req.pickup_longitude,
        req.dropoff_latitude, req.dropoff_longitude
    ), 4)
    bin_label = hour_bin(hour)

    try:
        bin_enc = int(le.transform([bin_label])[0])
    except Exception:
        bin_enc = 0

    df = pd.DataFrame([{
        "hour_of_day":          hour,
        "day_of_week":          dow,
        "is_weekend":           is_weekend,
        "distance_km":          distance,
        "pickup_hour_bin_enc":  bin_enc
    }])

    df_sc = pd.DataFrame(scaler.transform(df), columns=df.columns)
    return df_sc


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=MODEL_VERSION
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    logger.info("Predict request: %s -> %s | passengers: %d",
                req.pickup_datetime,
                f"({req.dropoff_latitude:.4f},{req.dropoff_longitude:.4f})",
                req.passenger_count)
    try:
        features = build_features(req)
        log_pred = float(model.predict(features)[0])
        duration_seconds = round(float(np.expm1(log_pred)), 2)
        duration_minutes = round(duration_seconds / 60, 2)

        logger.info("Prediction: %.1f seconds (%.1f min)",
                    duration_seconds, duration_minutes)

        return PredictResponse(
            predicted_duration_seconds=duration_seconds,
            predicted_duration_minutes=duration_minutes,
            model_version=MODEL_VERSION,
            status="ok"
        )
    except Exception as e:
        logger.error("Prediction failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
