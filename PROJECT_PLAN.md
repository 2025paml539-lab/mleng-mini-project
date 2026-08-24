# Mini-Project-I — Project Plan
### PCAM ZC412 — Machine Learning Engineering
### Flavor A: Delivery / Ride ETA Prediction
**Due Date:** Monday, 31 August 2026, 11:59 PM
**Team Members:** Kishore Nandhalu | Vinay | Vishruth
**Repository:** https://github.com/2025paml539-lab/mleng-mini-project

---

## Project Overview

**Problem Statement:**
Build an end-to-end ML pipeline that predicts NYC taxi trip duration based on pickup/dropoff GPS coordinates, time of day, day of week, and distance. The system covers data ingestion, validation, feature engineering, model training, REST API serving, and drift monitoring.

**Dataset:** NYC Taxi Trip Duration — Kaggle
- Source: `https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data`
- 1,458,644 taxi trips (Jan–Jun 2016)
- Target: `trip_duration` (seconds)

**Tools:**
- Python 3.14.5, pandas, scikit-learn, XGBoost
- MLflow (experiment tracking), DVC (dataset versioning)
- FastAPI (serving), Docker (packaging)
- Git + GitHub (version control)

---

## Repository Structure

```
mleng-mini-project/
├── data/
│   ├── raw/train.csv.dvc         # DVC pointer — 191MB CSV not in Git
│   └── processed/features.csv   # output of pipeline/features.py
├── pipeline/
│   ├── ingest.py                 # Week 1: load + profile raw data
│   ├── validate.py               # Week 1: 4-level validation
│   └── features.py               # Week 1: feature engineering
├── training/
│   └── train.py                  # Week 2: LR + XGBoost with MLflow
├── serving/                      # Week 3: FastAPI endpoint (pending)
├── monitoring/                   # Week 4: drift detection (pending)
├── artifacts/
│   ├── feature_schema.json
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_selection.json
├── reports/
│   └── model_comparison.md       # Week 2: model comparison report
├── outputs/
│   ├── week1/                    # Week 1 charts + pipeline run log
│   ├── week2/                    # Week 2 charts + training run log
│   ├── week3/                    # Week 3 API test evidence (pending)
│   └── week4/                    # Week 4 drift report (pending)
├── params.yaml                   # all hyperparameters
├── dvc.yaml                      # pipeline stage definitions
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Weekly Plan

---

### Week 1 — Data Engineering & Feature Pipeline
**Module:** M2 | **Completed:** Aug 13, 2026

#### What Was Done

- Git repo initialised, pushed to GitHub. Python 3.14.5 and Git 2.54.0 confirmed.
- Downloaded NYC Taxi dataset from Kaggle — 1,458,644 rows, 191MB.
- `pipeline/ingest.py` — loads CSV, prints shape, null counts, date range.
- `pipeline/validate.py` — 4-level validation:
  - L1 Schema: all 11 columns, correct types, 0 nulls — PASS
  - L2 Range: 689 outlier rows (0.047%) removed — passenger count, NYC GPS bounds, duration limits
  - L3 Statistical: 1,457,955 clean rows, mean duration 959s — PASS
  - L4 Business: dropoff > pickup for all rows — PASS
- `pipeline/features.py` — 5 features engineered:
  - `hour_of_day`, `day_of_week`, `is_weekend`, `distance_km` (Haversine), `pickup_hour_bin`
  - Output saved to `data/processed/features.csv`
  - `artifacts/feature_schema.json` saved for serving-time reuse
- DVC initialised, `data/raw/train.csv.dvc` pointer committed to Git
- `dvc.yaml` and `params.yaml` created

#### Actual Terminal Output
```
python pipeline/ingest.py
[INGEST] Loaded     : 1,458,644 rows x 11 cols
[INGEST] Nulls      : 0 — all columns complete
[INGEST] trip_duration — min: 1s  max: 3526282s  mean: 959s
[INGEST] Date range : 2016-01-01 to 2016-06-30

python pipeline/validate.py
[L1 SCHEMA]   PASS
[L2 RANGE]    WARN — 689 rows removed (0.047%)
[L3 STATS]    PASS — 1,457,955 rows | mean 959s
[L4 BUSINESS] PASS
[VALIDATE] All checks passed

python pipeline/features.py
[FEATURES] Saved: data/processed/features.csv (1,458,644 rows x 6 cols)
[FEATURES] Saved: artifacts/feature_schema.json
```

**Git tag:** `v1.0-week1`

---

### Week 2 — Experimentation & Reproducibility
**Module:** M3 | **Completed:** Aug 24, 2026

#### What Was Done

- `training/train.py` built — trains Linear Regression and XGBoost, logs both to MLflow.
- MLflow experiment: `eta-prediction` with file-based tracking in `mlruns/`.
- All hyperparameters loaded from `params.yaml` — no hardcoding.
- Temporal train/test split (80/20, no shuffle) — preserves time order.
- `StandardScaler` fitted on training data only, saved to `artifacts/scaler.pkl`.
- Target: `log1p(trip_duration)` to reduce skew.
- Both models logged with params, RMSE, MAE, R2, and model artifact.
- XGBoost selected as best model — saved to `artifacts/model.pkl`.
- `reports/model_comparison.md` written with justification.

#### Results

| Metric | Linear Regression | XGBoost |
|---|---|---|
| RMSE | 282,405,528 s | **3,162 s** |
| MAE | 525,057 s | **344 s** |
| R2 | 0.3883 | **0.6532** |

XGBoost selected — R2 improvement of 0.2649 exceeds the 0.05 threshold. LR fails on this data because trip duration has a non-linear relationship with distance and time. See `reports/model_comparison.md` for full justification.

#### Actual Terminal Output
```
python training/train.py
[TRAIN] Train: 1,166,915 rows | Test: 291,729 rows
[LR]  RMSE: 282,405,528s | MAE: 525,057s | R2: 0.3883
[XGB] RMSE: 3,162s       | MAE: 344s     | R2: 0.6532
[TRAIN] Winner - XGBoost
[TRAIN] Saved -> artifacts/model.pkl
```

**Git tag:** `v1.0-week2`

---

### Week 3 — Model Packaging & Deployment
**Module:** M4 | **Target:** Aug 26, 2026

#### Plan

- `serving/schemas.py` — Pydantic request/response models
- `serving/api.py` — FastAPI with `/predict` and `/health` endpoints
  - Load model and scaler once at startup
  - Input validation — reject invalid coordinates, passenger count out of range
  - Log every request with timestamp
- `Dockerfile` — containerise the serving API
- Test with curl, document in `reports/api_test_report.md`

---

### Week 4 — Monitoring, Drift & Retraining
**Module:** M5 | **Target:** Aug 30, 2026

#### Plan

- `monitoring/logger.py` — log every prediction to CSV
- `monitoring/simulate_drift.py` — inject rush-hour surge data to simulate drift
- `monitoring/drift_detector.py` — KS test on feature distributions, alert if p < 0.05
- `reports/drift_report.md` — before/after comparison with retraining trigger design

---

## Feature Engineering Decisions

| Feature | Source | How | Why |
|---|---|---|---|
| `hour_of_day` | pickup_datetime | Extract hour 0–23 | Traffic varies strongly by hour |
| `day_of_week` | pickup_datetime | Extract weekday 0–6 | Weekday vs weekend patterns differ |
| `is_weekend` | day_of_week | 1 if Sat/Sun | Leisure vs commute trips differ |
| `distance_km` | GPS coords | Haversine formula | Strongest predictor of duration |
| `pickup_hour_bin` | hour_of_day | night/morning/afternoon/evening | Captures non-linear time effects |

All transformation parameters are saved to `artifacts/feature_schema.json` and reused at serving time to prevent training-serving skew.

---

## Submission Checklist

| # | Deliverable | Week | Status |
|---|---|---|---|
| 1 | GitHub repo with weekly commit history | Setup | ✅ Aug 13 |
| 2 | Project plan pushed to repo | Setup | ✅ Aug 13 |
| 3 | `pipeline/ingest.py` running with output | Week 1 | ✅ Aug 13 |
| 4 | `pipeline/validate.py` — 4-level validation | Week 1 | ✅ Aug 13 |
| 5 | `pipeline/features.py` — 5 features engineered | Week 1 | ✅ Aug 13 |
| 6 | Dataset versioned with DVC | Week 1 | ✅ Aug 13 |
| 7 | Tag `v1.0-week1` pushed | Week 1 | ✅ Aug 13 |
| 8 | MLflow — 2 tracked experiments | Week 2 | ✅ Aug 24 |
| 9 | `reports/model_comparison.md` | Week 2 | ✅ Aug 24 |
| 10 | Tag `v1.0-week2` pushed | Week 2 | ✅ Aug 24 |
| 11 | FastAPI `/predict` endpoint working | Week 3 | ⬜ Pending |
| 12 | Docker container builds and runs | Week 3 | ⬜ Pending |
| 13 | `reports/api_test_report.md` with curl tests | Week 3 | ⬜ Pending |
| 14 | Prediction log CSV | Week 4 | ⬜ Pending |
| 15 | `reports/drift_report.md` | Week 4 | ⬜ Pending |
| 16 | Final README with architecture diagram | Week 4 | ⬜ Pending |
| 17 | 5–7 min demo video recorded | Aug 30 | ⬜ Pending |
| 18 | GitHub link submitted on BITS portal | Aug 31 | ⬜ Pending |

---

## Progress Log

| Date | What Was Done |
|---|---|
| Aug 13, 2026 | Project brief read. Flavor A selected. Group formed: Kishore Nandhalu, Vinay, Vishruth. GitHub account and repo created. Dataset downloaded from Kaggle (1,458,644 rows). Week 1 pipeline built and run — ingest, validate, features all working with verified output. DVC versioning set up. Tag v1.0-week1 pushed. |
| Aug 24, 2026 | Week 2 done. Trained Linear Regression and XGBoost with MLflow tracking. XGBoost selected (R2=0.6532, RMSE=3162s). model.pkl, scaler.pkl, label_encoder.pkl saved. model_comparison.md written. Output charts in outputs/week2/. Tag v1.0-week2 pushed. |
| Aug 26, 2026 | *(to be updated — Week 3)* |
| Aug 30, 2026 | *(to be updated — Week 4 + demo)* |
| Aug 31, 2026 | *(to be updated — submitted)* |

---

## References

- T1: Crowe, R. et al. *Machine Learning Production Systems*. O'Reilly, 2024.
- T2: Burkov, A. *Machine Learning Engineering*. 2020.
- Dataset: NYC Taxi Trip Duration — Kaggle
