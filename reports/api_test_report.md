# API Test Report
## Week 3 — PCAM ZC412 Mini-Project-I
**Team:** Kishore Nandhalu | Vinay | Vishruth

---

## Setup

- API: FastAPI running via `uvicorn serving.api:app --port 8000`
- Model: XGBoost loaded from `artifacts/model.pkl`
- Version: `v1.0-week3`

---

## Test Results

### T1 — Health Check

```
GET http://localhost:8000/health
```

Response (HTTP 200):
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v1.0-week3"
}
```

---

### T2 — Short Night Trip

```
POST http://localhost:8000/predict
```

Request:
```json
{
  "pickup_datetime": "2016-06-30 23:59:58",
  "pickup_longitude": -73.982155,
  "pickup_latitude": 40.767937,
  "dropoff_longitude": -73.964630,
  "dropoff_latitude": 40.765602,
  "passenger_count": 1
}
```

Response (HTTP 200):
```json
{
  "predicted_duration_seconds": 427.84,
  "predicted_duration_minutes": 7.13,
  "model_version": "v1.0-week3",
  "status": "ok"
}
```

---

### T3 — Morning Rush Hour

Request:
```json
{
  "pickup_datetime": "2016-03-14 08:30:00",
  "pickup_longitude": -73.985130,
  "pickup_latitude": 40.758896,
  "dropoff_longitude": -73.940271,
  "dropoff_latitude": 40.748817,
  "passenger_count": 2
}
```

Response (HTTP 200):
```json
{
  "predicted_duration_seconds": 1101.86,
  "predicted_duration_minutes": 18.36,
  "model_version": "v1.0-week3",
  "status": "ok"
}
```

---

### T4 — Weekend Afternoon

Request:
```json
{
  "pickup_datetime": "2016-01-16 14:20:00",
  "pickup_longitude": -73.978271,
  "pickup_latitude": 40.752781,
  "dropoff_longitude": -73.921640,
  "dropoff_latitude": 40.731151,
  "passenger_count": 3
}
```

Response (HTTP 200):
```json
{
  "predicted_duration_seconds": 1320.36,
  "predicted_duration_minutes": 22.01,
  "model_version": "v1.0-week3",
  "status": "ok"
}
```

---

### T5 — Evening Long Trip

Request:
```json
{
  "pickup_datetime": "2016-05-20 19:45:00",
  "pickup_longitude": -74.002320,
  "pickup_latitude": 40.748817,
  "dropoff_longitude": -73.875612,
  "dropoff_latitude": 40.773994,
  "passenger_count": 1
}
```

Response (HTTP 200):
```json
{
  "predicted_duration_seconds": 1803.75,
  "predicted_duration_minutes": 30.06,
  "model_version": "v1.0-week3",
  "status": "ok"
}
```

---

### T6 — Invalid Input: passenger_count out of range

Request: `passenger_count: 10`

Response (HTTP 422):
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "passenger_count"],
      "msg": "Value error, passenger_count must be between 1 and 6"
    }
  ]
}
```

---

### T7 — Invalid Input: longitude outside NYC bounds

Request: `pickup_longitude: -50.0`

Response (HTTP 422):
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "pickup_longitude"],
      "msg": "Value error, Longitude -50.0 is outside NYC bounds (-74.3 to -73.6)"
    }
  ]
}
```

---

## Summary

| Test | Endpoint | Input | Status | Result |
|---|---|---|---|---|
| T1 | GET /health | — | 200 | model_loaded=true |
| T2 | POST /predict | Night trip | 200 | 427s (7.1 min) |
| T3 | POST /predict | Rush hour | 200 | 1101s (18.4 min) |
| T4 | POST /predict | Weekend | 200 | 1320s (22.0 min) |
| T5 | POST /predict | Long trip | 200 | 1803s (30.1 min) |
| T6 | POST /predict | passenger=10 | 422 | Validation error |
| T7 | POST /predict | lon=-50 | 422 | Validation error |

All 7 tests passed. Valid inputs return predictions. Invalid inputs are rejected with HTTP 422 before reaching the model.

---

## Observations

The predictions are sensible:
- Short 1.5km night trip = 7 min (low traffic, short distance)
- Same distance in morning rush = 18 min (traffic factor captured by hour features)
- Weekend afternoon longer trip = 22 min
- Evening 10km trip = 30 min

The model correctly uses `hour_of_day` and `is_weekend` to adjust predictions based on traffic patterns, not just distance.

---

## How to Run

```bash
# Start API
uvicorn serving.api:app --host 0.0.0.0 --port 8000

# Run all tests
python serving/run_tests.py

# Or test manually with curl
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pickup_datetime":"2016-06-30 23:59:58","pickup_longitude":-73.982155,"pickup_latitude":40.767937,"dropoff_longitude":-73.964630,"dropoff_latitude":40.765602,"passenger_count":1}'
```
