# Mini-Project-I — Project Plan
### PCAM ZC412 — Machine Learning Engineering
### Flavor A: Delivery / Ride ETA Prediction
**Due Date:** Monday, 24 August 2026
**Team Members:** Kishore Nandhalu | Vinay | Vishruth
**Repository:** https://github.com/2025paml539-lab/mleng-mini-project

---

## Project Overview

**Problem Statement:**
Build an end-to-end ML pipeline that predicts taxi trip duration based on pickup/dropoff location, time of day, day of week, and distance. The system will ingest raw trip data, engineer time- and location-based features, train and compare regression models, deploy the best model as a REST API, and monitor it for accuracy drift as traffic patterns change.

**Dataset:** NYC Taxi Trip Duration — Kaggle Competition Dataset
- Source: `https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data`
- ~1.4 million NYC taxi trips
- Target variable: `trip_duration` (seconds)

**Tools & Technologies:**
- Python 3.14.5, pandas 2.2.2, scikit-learn 1.5.0, XGBoost 2.0.3
- MLflow 2.13.2 (experiment tracking)
- DVC 3.51.2 (dataset versioning)
- FastAPI 0.111.0 (model serving)
- Docker (packaging)
- Git + GitHub (version control)
- Pandera 0.19.3 (data validation)

---

## Repository Structure

```
mleng-mini-project/
├── data/
│   ├── raw/
│   │   └── train.csv.dvc         # DVC pointer — 1,458,644 rows (191MB, not in Git)
│   └── processed/                # features.csv written by pipeline/features.py
├── pipeline/
│   ├── ingest.py                 # ✅ Loads CSV, prints shape/nulls/date range
│   ├── validate.py               # ✅ 4-level validation — removed 689 outliers (0.047%)
│   └── features.py               # ✅ 5 features engineered, saves features.csv + schema
├── training/                     # Week 2 — to be built
├── serving/                      # Week 3 — to be built
├── monitoring/                   # Week 4 — to be built
├── artifacts/
│   └── feature_schema.json       # ✅ Transformation params saved for serving-time reuse
├── reports/                      # Week 2–4 — to be built
├── params.yaml                   # ✅ All hyperparameters (single source of truth)
├── dvc.yaml                      # ✅ Pipeline stage definitions
├── requirements.txt              # ✅ All dependencies pinned
├── .gitignore                    # ✅ Excludes raw data, artifacts, __pycache__
└── README.md                     # Week 4 — final version with architecture diagram
```

---

## Weekly Plan

---

### Week 1 — Data Engineering & Feature Pipeline
**Module:** M2 | **Dates:** Aug 2 – Aug 8, 2026

#### Objective
Set up the full data ingestion, validation, and feature engineering pipeline. Version the dataset using DVC.

#### What Was Done

**Setup & Data Ingestion** ✅
- [x] Git repository initialised, connected to GitHub, initial commit pushed
- [x] Python 3.14.5 and Git 2.54.0 verified on local machine
- [x] `requirements.txt` created with all pinned dependencies
- [x] NYC Taxi dataset downloaded from Kaggle — 1,458,644 rows, 191MB
- [x] `pipeline/ingest.py` — loads CSV, prints shape, null counts, date range, duration stats

**Data Validation** ✅
- [x] `pipeline/validate.py` — 4-level validation implemented and verified:
  - L1 Schema: all 11 columns present, correct dtypes, 0 nulls — **PASS**
  - L2 Range: removed 689 outlier rows (0.047%) — passenger count, NYC GPS bounds, duration 1–86400s
  - L3 Statistical: 1,457,955 clean rows, mean duration 959s — **PASS**
  - L4 Business rule: dropoff > pickup for all rows — **PASS**

**Feature Engineering** ✅
- [x] `pipeline/features.py` — 5 features engineered and verified:
  - `hour_of_day`: range 0–23
  - `day_of_week`: range 0–6
  - `is_weekend`: 416,234 weekend trips
  - `distance_km`: mean 3.44km, max 1,240km (Haversine formula)
  - `pickup_hour_bin`: evening(499k), afternoon(430k), morning(357k), night(171k)
- [x] Processed dataset saved: `data/processed/features.csv` — 1,458,644 rows x 6 cols
- [x] `artifacts/feature_schema.json` saved — reused at serving time to prevent training-serving skew

**Dataset Versioning** ✅
- [x] DVC 3.51.2 installed and initialised
- [x] `data/raw/train.csv.dvc` — DVC pointer file committed (MD5 hash + file size, not the 191MB CSV)
- [x] `dvc.yaml` — pipeline stages defined (ingest → validate → featurize)
- [x] `params.yaml` — all hyperparameters in one file

#### Week 1 Actual Output
```
python pipeline/ingest.py
→ [INGEST] Loaded: 1,458,644 rows x 11 cols | Nulls: 0 | Duration: 1s–3,526,282s | mean: 959s

python pipeline/validate.py
→ [L1 SCHEMA]   PASS — all 11 columns, correct types, 0 nulls
→ [L2 RANGE]    WARN — 689 outlier rows (0.047%) removed
→ [L3 STATS]    PASS — 1,457,955 rows | mean 959s
→ [L4 BUSINESS] PASS — dropoff > pickup for all rows
→ [VALIDATE] All checks passed ✓

python pipeline/features.py
→ [FEATURES] 5 features engineered
→ [FEATURES] Saved: data/processed/features.csv (1,458,644 rows x 6 cols)
→ [FEATURES] Saved: artifacts/feature_schema.json ✓
```

#### Week 1 Git Commit
```
commit: "Week 1: data ingestion, 4-level validation, feature engineering pipeline — DVC dataset versioned"
tag:    v1.0-week1
files:  13 files changed, 381 insertions
```

---

### Week 2 — Experimentation & Reproducibility
**Module:** M3 | **Dates:** Aug 9 – Aug 15, 2026

#### Objective
Train at least two models, track all experiments with MLflow, select the best model with documented justification, and ensure full reproducibility.

#### What I Will Do

**Day 1–2: Experiment Tracking Setup**
- [ ] Install and configure MLflow locally (`mlflow server` or file-based tracking)
- [ ] Write `training/params.yaml` — all hyperparameters in one file (no hardcoding)
- [ ] Set up MLflow experiment: `mlflow.set_experiment("eta-prediction")`

**Day 3–4: Train Model 1 — Linear Regression (Baseline)**
- [ ] Write `training/train.py` with:
  - Load features from `data/processed/`
  - Temporal train/test split (no random shuffle — respect time order)
  - `StandardScaler` fitted on training data only, saved to `artifacts/scaler.pkl`
  - Train `LinearRegression` with fixed `random_state=42`
  - Log to MLflow: params, RMSE, MAE, R², model artifact

**Day 5–6: Train Model 2 — XGBoost**
- [ ] Add XGBoost run to `training/train.py`:
  - Same train/test split as baseline
  - Same scaler (loaded from `artifacts/scaler.pkl`)
  - Train `XGBRegressor` with tuned hyperparameters from `params.yaml`
  - Log to MLflow: same metrics for fair comparison

**Day 7: Model Comparison & Selection**
- [ ] Compare both runs in MLflow UI
- [ ] Write `reports/model_comparison.md` — table comparing RMSE, MAE, R² for both models
- [ ] Select best model with written justification (not just "higher accuracy" — explain why)
- [ ] Save best model to `artifacts/model.pkl`
- [ ] Tag best model in MLflow Model Registry as `production-candidate`

#### Reproducibility Check
- [ ] Delete `artifacts/model.pkl` and re-run `python training/train.py`
- [ ] Confirm same metrics are produced (fixed seed, fixed data version)

#### Week 2 Deliverable
- MLflow UI showing 2+ tracked runs with full parameters and metrics
- `reports/model_comparison.md` with model selection justification
- Reproducible training pipeline

#### Week 2 Git Commit
```bash
git add training/ reports/model_comparison.md artifacts/
git commit -m "Week 2: MLflow experiment tracking, Linear Regression vs XGBoost, model selected"
git push origin main
git tag v1.0-week2
git push --tags
```

---

### Week 3 — Model Packaging & Deployment
**Module:** M4 | **Dates:** Aug 16 – Aug 20, 2026

#### Objective
Package the trained model as a Docker container and expose it as a FastAPI REST endpoint with proper input validation and error handling.

#### What I Will Do

**Day 1–2: FastAPI Serving Endpoint**
- [ ] Write `serving/schemas.py` — Pydantic models for request and response:
  - Request: `pickup_datetime`, `pickup_longitude`, `pickup_latitude`, `dropoff_longitude`, `dropoff_latitude`, `passenger_count`
  - Response: `predicted_duration_seconds`, `predicted_duration_minutes`, `model_version`
- [ ] Write `serving/api.py`:
  - Load model and scaler once at startup (not per request)
  - `POST /predict` — accepts trip details, returns ETA
  - `GET /health` — returns model version and status
  - Input validation via Pydantic (invalid coordinates, negative passenger count → HTTP 422)
  - Log every request with timestamp

**Day 3: Containerization**
- [ ] Write `Dockerfile`:
  - Base image: `python:3.11-slim`
  - Copy requirements, install dependencies (layer caching)
  - Copy serving code and artifacts
  - Non-root user for security
  - Expose port 8000
- [ ] Build image: `docker build -t eta-predictor:v1 .`
- [ ] Run container: `docker run -p 8000:8000 eta-predictor:v1`

**Day 4–5: API Testing & Documentation**
- [ ] Test valid request with `curl`:
  ```bash
  curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"pickup_datetime":"2016-06-30 23:59:58","pickup_longitude":-73.982155,"pickup_latitude":40.767937,"dropoff_longitude":-73.964630,"dropoff_latitude":40.765602,"passenger_count":1}'
  ```
- [ ] Test invalid inputs and confirm proper error responses (HTTP 422)
- [ ] Write `reports/api_test_report.md` — document all test cases with request/response

#### Week 3 Deliverable
- Docker container running locally on port 8000
- Working `/predict` endpoint tested with sample inputs
- `reports/api_test_report.md` with curl test evidence

#### Week 3 Git Commit
```bash
git add serving/ Dockerfile reports/api_test_report.md
git commit -m "Week 3: FastAPI endpoint, Docker packaging, input validation, API tested"
git push origin main
git tag v1.0-week3
git push --tags
```

---

### Week 4 — Monitoring, Drift & Retraining
**Module:** M5 | **Dates:** Aug 21 – Aug 23, 2026

#### Objective
Implement prediction logging, simulate data drift, detect it with statistical tests, and design a retraining trigger strategy.

#### What I Will Do

**Day 1: Prediction Logging**
- [ ] Write `monitoring/logger.py`:
  - Log every prediction: timestamp, all input features, predicted value, model version
  - Append to `monitoring/prediction_log.csv`
  - Wire into `serving/api.py` so every `/predict` call is automatically logged

**Day 2: Drift Simulation**
- [ ] Write `monitoring/simulate_drift.py`:
  - Scenario 1 — Rush-hour surge: inject trips with unusually long distances and peak-hour timestamps
  - Scenario 2 — Festival/holiday pattern: inject unusually high passenger counts with short distances
  - Generate 500 synthetic drifted records and append to prediction log

**Day 3: Drift Detection & Retraining Design**
- [ ] Write `monitoring/drift_detector.py`:
  - Load training reference distribution
  - Load recent prediction log window (last 500 predictions)
  - Run KS test on each numerical feature: `distance_km`, `hour_of_day`, `passenger_count`
  - Run Chi-squared test on categorical features: `day_of_week`, `is_weekend`
  - Print drift alert if p-value < 0.05
- [ ] Write `reports/drift_report.md`:
  - Show KS test results before drift simulation (p-values, no drift detected)
  - Show KS test results after drift simulation (p-values, drift detected)
  - Document which features drifted and by how much

**Retraining Trigger Design (documented in drift_report.md):**
- Trigger condition: `distance_km` KS p-value < 0.05 for 3 consecutive monitoring windows
- Action: flag for human review → approve → re-run `python training/train.py` with new data
- Rationale: distance is the most predictive feature; drift here has the highest business impact

#### Week 4 Deliverable
- Prediction logger integrated into the API
- Drift simulation producing measurable distribution shift
- Drift detector identifying the shift with statistical evidence
- `reports/drift_report.md` showing before/after comparison

#### Week 4 Git Commit
```bash
git add monitoring/ reports/drift_report.md README.md
git commit -m "Week 4: prediction logging, drift simulation, KS-test monitoring, final README"
git push origin main
git tag v1.0-final
git push --tags
```

---

## Model Selection Justification

| Metric | Linear Regression | XGBoost | Winner |
|---|---|---|---|
| RMSE (seconds) | TBD after training | TBD after training | TBD |
| MAE (seconds) | TBD after training | TBD after training | TBD |
| R² | TBD after training | TBD after training | TBD |
| Training time | Fast | Moderate | LR faster |
| Inference latency | Very fast | Fast | LR faster |

**Decision criteria:**
- If XGBoost R² improvement over Linear Regression is > 5%, use XGBoost (the accuracy gain justifies the added complexity)
- If improvement < 5%, use Linear Regression (simpler, faster, more maintainable)
- Either way: both models are logged in MLflow for full traceability

---

## Feature Engineering Decisions

| Feature | Source | Transformation | Justification |
|---|---|---|---|
| `hour_of_day` | pickup_datetime | Extract hour (0–23) | Traffic patterns differ strongly by hour |
| `day_of_week` | pickup_datetime | Extract weekday (0–6) | Weekday vs weekend has different trip patterns |
| `is_weekend` | day_of_week | Binary (Sat/Sun = 1) | Captures weekend leisure vs weekday commute |
| `distance_km` | GPS coordinates | Haversine formula | Most direct predictor of trip duration |
| `pickup_hour_bin` | hour_of_day | 4 bins: morning/afternoon/evening/night | Non-linear time effects |

**All transformation parameters saved to `artifacts/feature_schema.json` and reused identically at serving time to prevent training-serving skew.**

---

## Submission Checklist

| # | Deliverable | Week | Status |
|---|---|---|---|
| 1 | GitHub repository with weekly commit history | Setup | ✅ Done — Aug 13 |
| 2 | Project plan committed and pushed | Setup | ✅ Done — Aug 13 |
| 3 | `pipeline/ingest.py` — data loading verified | Week 1 | ✅ Done — Aug 13 |
| 4 | `pipeline/validate.py` — 4-level validation complete | Week 1 | ✅ Done — Aug 13 |
| 5 | `pipeline/features.py` — 5 features engineered | Week 1 | ✅ Done — Aug 13 |
| 6 | `data/raw/train.csv.dvc` — dataset DVC versioned | Week 1 | ✅ Done — Aug 13 |
| 7 | `artifacts/feature_schema.json` committed | Week 1 | ✅ Done — Aug 13 |
| 8 | Tag `v1.0-week1` pushed | Week 1 | ✅ Done — Aug 13 |
| 9 | MLflow experiment logs — 2+ tracked runs | Week 2 | ⬜ Pending |
| 10 | `reports/model_comparison.md` | Week 2 | ⬜ Pending |
| 11 | Working FastAPI `/predict` endpoint | Week 3 | ⬜ Pending |
| 12 | Docker container builds and runs | Week 3 | ⬜ Pending |
| 13 | `reports/api_test_report.md` with curl evidence | Week 3 | ⬜ Pending |
| 14 | Prediction log CSV | Week 4 | ⬜ Pending |
| 15 | `reports/drift_report.md` with KS test results | Week 4 | ⬜ Pending |
| 16 | Final `README.md` with architecture diagram | Week 4 | ⬜ Pending |
| 17 | 5–7 minute demo video | Aug 23 | ⬜ Pending |
| 18 | Submitted GitHub link on BITS portal | Aug 24 | ⬜ Pending |

---

## Progress Log

| Date | What Was Done |
|---|---|
| Aug 13, 2026 | Project brief analysed. Flavor A (ETA Prediction) selected. Group formed: Kishore Nandhalu, Vinay, Vishruth. GitHub account (`2025paml539-lab`) and repository `mleng-mini-project` created. Python 3.14.5 and Git 2.54.0 verified. `PROJECT_PLAN.md` committed with full 4-week plan. NYC Taxi dataset (1,458,644 rows, 191MB) downloaded from Kaggle and extracted. DVC 3.51.2 installed. `pipeline/ingest.py`, `pipeline/validate.py`, `pipeline/features.py` built and executed — all 3 scripts produce verified output. 4-level validation passed (689 outliers removed, 0.047%). 5 features engineered and saved to `data/processed/features.csv`. `artifacts/feature_schema.json` saved. `dvc.yaml`, `params.yaml`, `requirements.txt`, `.gitignore` committed. Tag `v1.0-week1` pushed. Repo has 5 commits with clear weekly history. |
| Aug 15, 2026 | *(to be updated — Week 2 complete)* |
| Aug 20, 2026 | *(to be updated — Week 3 complete)* |
| Aug 23, 2026 | *(to be updated — Week 4 complete, demo recorded)* |
| Aug 24, 2026 | *(to be updated — submitted)* |

---

## References

- T1: Crowe, R. et al. *Machine Learning Production Systems*. O'Reilly, 2024.
- T2: Burkov, A. *Machine Learning Engineering*. 2020.
- R1: McMahon, A.P. *Machine Learning Engineering with Python*, 2nd Ed. Packt, 2023.
- Dataset: NYC Taxi Trip Duration, Kaggle — `https://www.kaggle.com/competitions/nyc-taxi-trip-duration`

---

*PCAM ZC412 | Mini-Project-I | Flavor A — ETA Prediction | BITS Pilani WILP*
