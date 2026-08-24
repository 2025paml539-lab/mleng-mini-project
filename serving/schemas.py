"""
serving/schemas.py
Pydantic request and response models for the /predict endpoint.
Input validation runs automatically before the model is called.
"""

from pydantic import BaseModel, field_validator
from typing import Optional


class PredictRequest(BaseModel):
    pickup_datetime: str       # e.g. "2016-06-30 23:59:58"
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

    @field_validator("passenger_count")
    @classmethod
    def passenger_count_valid(cls, v):
        if not (1 <= v <= 6):
            raise ValueError("passenger_count must be between 1 and 6")
        return v

    @field_validator("pickup_latitude", "dropoff_latitude")
    @classmethod
    def latitude_in_nyc(cls, v):
        if not (40.4 <= v <= 41.0):
            raise ValueError(f"Latitude {v} is outside NYC bounds (40.4 to 41.0)")
        return v

    @field_validator("pickup_longitude", "dropoff_longitude")
    @classmethod
    def longitude_in_nyc(cls, v):
        if not (-74.3 <= v <= -73.6):
            raise ValueError(f"Longitude {v} is outside NYC bounds (-74.3 to -73.6)")
        return v


class PredictResponse(BaseModel):
    predicted_duration_seconds: float
    predicted_duration_minutes: float
    model_version: str
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
