# Drift Detection Report
## Week 4 — PCAM ZC412 Mini-Project-I
**Team:** Kishore Nandhalu | Vinay | Vishruth

---

## Overview

This report documents the monitoring setup, drift simulation, and detection results for the deployed NYC Taxi ETA prediction model (XGBoost v1.0-week3).

Monitoring works in three steps:
1. Every `/predict` API call is logged to `monitoring/prediction_log.csv`
2. Drift simulation injects 300 drifted records (2 scenarios)
3. KS test and Chi-squared test compare production distribution vs training reference

---

## Drift Simulation

Two realistic drift scenarios were injected into the prediction log:

### Scenario 1 — Rush-hour Surge (150 records)
Simulates a city-wide traffic surge event (e.g. marathon, major concert).
- `distance_km`: mean shifted from 3.44km → ~12.5km (much longer trips)
- `hour_of_day`: concentrated at peak hours (7–9am, 5–7pm) only
- Effect: model sees trips very different from training distribution

### Scenario 2 — Festival / Holiday Pattern (150 records)
Simulates New Year's Eve or major sporting event.
- `distance_km`: mean shifted from 3.44km → ~1.2km (very short trips)
- `hour_of_day`: concentrated at late night (10pm–2am) only
- `passenger_count`: mostly groups (3–6 people)
- Effect: unusual combination of features not well-represented in training

**Normal baseline:** 300 records matching training distribution (logged first)

**Total records in prediction log:** 600

---

## Detection Results

### Before Drift Simulation (300 normal records only)

| Feature | Test | Statistic | p-value | Result |
|---|---|---|---|---|
| distance_km | KS | ~0.02 | > 0.05 | NO DRIFT |
| hour_of_day | KS | ~0.03 | > 0.05 | NO DRIFT |
| passenger_count | Chi2 | ~5.2 | > 0.05 | NO DRIFT |

### After Drift Simulation (600 records — 300 normal + 300 drifted)

| Feature | Test | Statistic | p-value | Result |
|---|---|---|---|---|
| **distance_km** | KS | **0.3864** | **0.000000** | **DRIFT DETECTED** |
| **hour_of_day** | KS | **0.1775** | **0.000000** | **DRIFT DETECTED** |
| **passenger_count** | Chi2 | **112.36** | **0.000000** | **DRIFT DETECTED** |

All 3 features show statistically significant drift (p < 0.05 threshold).

---

## Retraining Trigger Design

**Trigger condition:**
> `distance_km` KS p-value < 0.05 for **3 consecutive monitoring windows**

**Why 3 consecutive windows?**
A single drift detection could be a temporary spike (e.g. one unusual day). Requiring 3 consecutive windows ensures the shift is persistent before triggering the cost of retraining.

**Monitoring window:** every 500 new predictions

**Trigger action:**
1. Flag for human review — send alert to ML engineer
2. Human inspects drift report and incoming data
3. If confirmed → re-run `python training/train.py` with updated data
4. Evaluate new model — if R2 improvement > 0.02, promote to production
5. Update `MODEL_VERSION` in `serving/api.py`

**Why distance_km is the primary trigger feature:**
`distance_km` has the highest feature importance in the XGBoost model (~0.55). Drift in this feature has the largest direct impact on prediction quality compared to `hour_of_day` (importance ~0.20).

---

## Monitoring Architecture

```
Every /predict call
        ↓
monitoring/logger.py
        ↓
monitoring/prediction_log.csv
        ↓ (every 500 records)
monitoring/drift_detector.py
        ↓
monitoring/drift_results.json
        ↓
Alert if drifted features >= 1
```

---

## How to Run

```bash
# Step 1: Log predictions (demo — logs 5 sample records)
python monitoring/logger.py

# Step 2: Simulate drift (300 normal + 300 drifted records)
python monitoring/simulate_drift.py

# Step 3: Detect drift
python monitoring/drift_detector.py
```

Expected output after simulation:
```
[DRIFT] DRIFT DETECTED — distance_km  KS stat=0.3864  p=0.000000
[DRIFT] DRIFT DETECTED — hour_of_day  KS stat=0.1775  p=0.000000
[DRIFT] DRIFT DETECTED — passenger_count  Chi2=112.36  p=0.000000
RECOMMENDATION: Trigger retraining review.
```
