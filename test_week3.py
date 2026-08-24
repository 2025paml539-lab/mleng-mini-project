"""
test_week3.py - Unit tests for Week 3
Tests API endpoints directly (API must be running on port 8000).
Run: python test_week3.py
"""
import os, sys, json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"

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


def post(body):
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        BASE + "/predict", data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def get(path):
    try:
        resp = urllib.request.urlopen(BASE + path, timeout=5)
        return resp.status, json.loads(resp.read())
    except Exception as e:
        return 0, str(e)


VALID = {
    "pickup_datetime":   "2016-06-30 23:59:58",
    "pickup_longitude":  -73.982155,
    "pickup_latitude":   40.767937,
    "dropoff_longitude": -73.964630,
    "dropoff_latitude":  40.765602,
    "passenger_count":   1
}

print("=" * 55)
print("Week 3 Unit Tests")
print("=" * 55)

# T1 - API files exist
print("\n[T1] Serving files")
for f in ["serving/api.py", "serving/schemas.py", "serving/__init__.py", "Dockerfile"]:
    check(f"{f} exists", os.path.exists(f))

# T2 - Health endpoint
print("\n[T2] GET /health")
status, body = get("/health")
check("returns HTTP 200",     status == 200,       f"got {status}")
check("status is ok",         body.get("status") == "ok")
check("model_loaded is true", body.get("model_loaded") is True)
check("model_version present",bool(body.get("model_version")))

# T3 - Valid prediction
print("\n[T3] POST /predict — valid request")
status, body = post(VALID)
check("returns HTTP 200",                    status == 200, f"got {status}")
check("predicted_duration_seconds present",  "predicted_duration_seconds" in body)
check("predicted_duration_minutes present",  "predicted_duration_minutes" in body)
check("prediction > 0",                      body.get("predicted_duration_seconds", 0) > 0)
check("prediction < 7200s (2 hours)",        body.get("predicted_duration_seconds", 9999) < 7200)
check("model_version is v1.0-week3",         body.get("model_version") == "v1.0-week3")
if "predicted_duration_seconds" in body:
    print(f"         Prediction: {body['predicted_duration_seconds']}s ({body['predicted_duration_minutes']} min)")

# T4 - Multiple valid trips
print("\n[T4] POST /predict — multiple trip types")
trips = [
    ("Rush hour",  {**VALID, "pickup_datetime": "2016-03-14 08:30:00", "passenger_count": 2}),
    ("Weekend",    {**VALID, "pickup_datetime": "2016-01-16 14:20:00", "passenger_count": 3}),
    ("Night",      {**VALID, "pickup_datetime": "2016-05-20 02:00:00", "passenger_count": 1}),
]
for name, body_trip in trips:
    s, r = post(body_trip)
    check(f"{name} returns 200", s == 200, f"got {s}")
    if s == 200:
        print(f"         {name}: {r.get('predicted_duration_seconds')}s")

# T5 - Input validation
print("\n[T5] POST /predict — invalid inputs rejected")
invalid_cases = [
    ("passenger_count=10",      {**VALID, "passenger_count": 10}),
    ("passenger_count=0",       {**VALID, "passenger_count": 0}),
    ("longitude out of bounds", {**VALID, "pickup_longitude": -50.0}),
    ("latitude out of bounds",  {**VALID, "pickup_latitude": 10.0}),
]
for name, bad_body in invalid_cases:
    s, r = post(bad_body)
    check(f"{name} returns 422", s == 422, f"got {s}")

# T6 - Reports exist
print("\n[T6] Report files")
check("reports/api_test_report.md exists", os.path.exists("reports/api_test_report.md"))

print("\n" + "=" * 55)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED - Week 3 is fully working")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
