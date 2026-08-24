"""
serving/run_tests.py
Runs all API test calls and prints results.
Run: python serving/run_tests.py
(API must be running: uvicorn serving.api:app --port 8000)
"""
import urllib.request
import json

BASE = "http://localhost:8000"

def post(path, body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

def get(path):
    req  = urllib.request.Request(BASE + path)
    resp = urllib.request.urlopen(req, timeout=5)
    return resp.status, json.loads(resp.read())

print("=" * 60)
print("Week 3 — API Test Results")
print("=" * 60)

# T1 - health
status, body = get("/health")
print("\nT1 GET /health")
print("  Status :", status)
print("  Body   :", json.dumps(body))

# T2-T5 - valid predictions
trips = [
    ("Short night trip",
     {"pickup_datetime":"2016-06-30 23:59:58",
      "pickup_longitude":-73.982155,"pickup_latitude":40.767937,
      "dropoff_longitude":-73.964630,"dropoff_latitude":40.765602,
      "passenger_count":1}),
    ("Morning rush hour",
     {"pickup_datetime":"2016-03-14 08:30:00",
      "pickup_longitude":-73.985130,"pickup_latitude":40.758896,
      "dropoff_longitude":-73.940271,"dropoff_latitude":40.748817,
      "passenger_count":2}),
    ("Weekend afternoon",
     {"pickup_datetime":"2016-01-16 14:20:00",
      "pickup_longitude":-73.978271,"pickup_latitude":40.752781,
      "dropoff_longitude":-73.921640,"dropoff_latitude":40.731151,
      "passenger_count":3}),
    ("Evening long trip",
     {"pickup_datetime":"2016-05-20 19:45:00",
      "pickup_longitude":-74.002320,"pickup_latitude":40.748817,
      "dropoff_longitude":-73.875612,"dropoff_latitude":40.773994,
      "passenger_count":1}),
]

for i, (name, body) in enumerate(trips, start=2):
    status, resp = post("/predict", body)
    secs = resp.get("predicted_duration_seconds", "N/A")
    mins = resp.get("predicted_duration_minutes", "N/A")
    print("\nT{} POST /predict — {}".format(i, name))
    print("  Status    :", status)
    print("  Seconds   :", secs)
    print("  Minutes   :", mins)
    print("  Model ver :", resp.get("model_version", "N/A"))

# T6 - invalid passenger count
print("\nT6 POST /predict — invalid passenger_count=10")
status, resp = post("/predict", {
    "pickup_datetime":"2016-06-30 08:00:00",
    "pickup_longitude":-73.982155,"pickup_latitude":40.767937,
    "dropoff_longitude":-73.964630,"dropoff_latitude":40.765602,
    "passenger_count":10
})
print("  Status :", status, "(expected 422)")
print("  Error  :", resp.get("detail","")[0].get("msg","") if isinstance(resp.get("detail"), list) else str(resp))

# T7 - invalid longitude
print("\nT7 POST /predict — invalid longitude=-50.0")
status, resp = post("/predict", {
    "pickup_datetime":"2016-06-30 08:00:00",
    "pickup_longitude":-50.0,"pickup_latitude":40.767937,
    "dropoff_longitude":-73.964630,"dropoff_latitude":40.765602,
    "passenger_count":1
})
print("  Status :", status, "(expected 422)")
print("  Error  :", resp.get("detail","")[0].get("msg","") if isinstance(resp.get("detail"), list) else str(resp))

print("\n" + "=" * 60)
print("All API tests complete")
print("=" * 60)
