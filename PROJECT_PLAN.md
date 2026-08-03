# Mini-Project-I — Project Plan
### PCAM ZC412 — Machine Learning Engineering
### Flavor A: Delivery / Ride ETA Prediction
**Due Date:** Monday, 24 August 2026
**Submitted By:** [Your Name] | [Student ID]

---

## Project Overview

**Problem Statement:**
Build an end-to-end ML pipeline that predicts taxi trip duration based on pickup/dropoff location, time of day, day of week, and distance. The system will ingest raw trip data, engineer time- and location-based features, train and compare regression models, deploy the best model as a REST API, and monitor it for accuracy drift as traffic patterns change.

**Dataset:** NYC Taxi Trip Duration — Kaggle Competition Dataset
- Source: `https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data`
- ~1.4 million NYC taxi trips
- Target variable: `trip_duration` (seconds)

**Tools & Technologies:**
- Python 3.11, pandas, scikit-learn, XGBoost
- MLflow (experiment tracking)
- DVC (dataset versioning)
- FastAPI (model serving)
- Docker (packaging)
- Git + GitHub (version control)
- Pandera / Great Expectations (data validation)

---

## Repository Structure

```
mleng-mini-project/
├── data/
│   ├── raw/                  # Original downloaded dataset (not committed — DVC tracked)
│   └── processed/            # Feature-engineered dataset
├── pipeline/
│   ├── ingest.py             # Data loading and schema validation
│   ├── features.py           # Feature engineering transformations
│   └── validate.py           # Statistical and business-rule validation
├── training/
│   ├── train.py              # Model training with MLflow logging
│   └── params.yaml           # Hyperparameters (single source of truth)
├── serving/
│   ├── api.py                # FastAPI prediction endpoint
│   └── schemas.py            # Pydantic request/response validation
├── monitoring/
│   ├── logger.py             # Prediction logging
│   ├── drift_detector.py     # KS-test based drift detection
│   └── simulate_drift.py     # Drift simulation (rush-hour surge)
├── artifacts/                # Saved model, scaler, feature schema
├── reports/                  # Model comparison report, drift report
├── Dockerfile
├── requirements.txt
├── .gitignore
├── dvc.yaml
└── README.md
```

---

## Weekly Plan

---

### Week 1 — Data Engineering & Feature Pipeline
**Module:** M2 | **Dates:** Aug 2 – Aug 8, 2026

#### Objective
Set up the full data ingestion, validation, and feature engineering pipeline. Version the dataset using DVC.

#### What I Will Do

**Day 1–2: Setup & Data Ingestion**
- [ ] Initialize Git repository and push initial commit
- [ ] Set up Python virtual environment (`venv`)
- [ ] Install all required libraries (`requirements.txt`)
- [ ] Download NYC Taxi dataset from Kaggle
- [ ] Write `pipeline/ingest.py` — load CSV, basic null checks, schema enforcement

**Day 3–4: Data Validation**
- [ ] Write `pipeline/validate.py` with 4-level validation:
  - L1 — Schema: column names, data types, no nulls in required fields
  - L2 — Range: passenger count 1–6, coordinates within NYC bounds, duration > 0
  - L3 — Statistical: row count check, mean trip duration within historical range
  - L4 — Business rules: dropoff timestamp must be after pickup timestamp
- [ ] Log validation results to console with clear PASS/FAIL messages

**Day 5–6: Feature Engineering**
- [ ] Write `pipeline/features.py` with these transformations:
  - `hour_of_day` — extracted from pickup_datetime (0–23)
  - `day_of_week` — Monday=0 to Sunday=6
  - `is_weekend` — binary flag (Saturday/Sunday = 1)
  - `distance_km` — Haversine formula from pickup/dropoff GPS coordinates
  - `pickup_hour_bin` — morning/afternoon/evening/night bins
- [ ] Save all transformation parameters to `artifacts/feature_schema.json`
- [ ] Save processed dataset to `data/processed/`

**Day 7: Dataset Versioning**
- [ ] Initialize DVC: `dvc init`
- [ ] Track dataset: `dvc add data/raw/train.csv`
- [ ] Commit pointer file to Git (not the raw data file)
- [ ] Tag dataset version: `git tag v1.0-dataset`

#### Week 1 Deliverable
- Working pipeline that runs end-to-end: raw CSV → validated → feature-engineered → versioned
- Git commit with all pipeline code pushed

#### Week 1 Git Commit
```bash
git add pipeline/ requirements.txt .gitignore dvc.yaml data/raw/train.csv.dvc
git commit -m "Week 1: data ingestion, validation, feature engineering pipeline complete"
git push origin main
git tag v1.0-week1
git push --tags
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
| 1 | GitHub repository with weekly commit history | All weeks | ⬜ |
| 2 | DVC-versioned dataset with pointer in Git | Week 1 | ⬜ |
| 3 | MLflow experiment logs — 2+ tracked runs | Week 2 | ⬜ |
| 4 | `reports/model_comparison.md` | Week 2 | ⬜ |
| 5 | Working FastAPI `/predict` endpoint | Week 3 | ⬜ |
| 6 | Docker container builds and runs | Week 3 | ⬜ |
| 7 | `reports/api_test_report.md` with curl evidence | Week 3 | ⬜ |
| 8 | Prediction log CSV | Week 4 | ⬜ |
| 9 | `reports/drift_report.md` with KS test results | Week 4 | ⬜ |
| 10 | Final `README.md` with architecture diagram | Week 4 | ⬜ |
| 11 | 5–7 minute demo video | Aug 23 | ⬜ |
| 12 | Submitted GitHub link on BITS portal | Aug 24 | ⬜ |

---

## Progress Log

| Date | What Was Done |
|---|---|
| Aug 2, 2026 | Project brief read. Flavor A selected. Group formed. GitHub repository created. Project plan drafted and committed. |
| Aug 3, 2026 | *(to be updated)* |
| Aug 8, 2026 | *(to be updated — Week 1 complete)* |
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
