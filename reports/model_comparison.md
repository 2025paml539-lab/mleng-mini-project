# Model Comparison Report
## Week 2 — PCAM ZC412 Mini-Project-I
**Team:** Kishore Nandhalu | Vinay | Vishruth

---

## Setup

We trained two models on the engineered features from Week 1 and tracked both runs using MLflow under the experiment name `eta-prediction`.

- **Dataset:** `data/processed/features.csv` — 1,458,644 rows
- **Split:** 80/20 temporal split (no shuffle — preserves time order of trips)
- **Train:** 1,166,915 rows | **Test:** 291,729 rows
- **Target:** `log1p(trip_duration)` — log-transformed to reduce the effect of extreme values
- **Random seed:** 42 fixed across both runs

---

## Features

| Feature | Description |
|---|---|
| `hour_of_day` | Hour extracted from pickup time (0–23) |
| `day_of_week` | 0 = Monday, 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |
| `distance_km` | Haversine distance between pickup and dropoff GPS coords |
| `pickup_hour_bin_enc` | Time of day bucket (night/morning/afternoon/evening), label encoded |

---

## Results

| | Linear Regression | XGBoost |
|---|---|---|
| RMSE | 282,405,528 s | **3,162 s** |
| MAE | 525,057 s | **344 s** |
| R2 | 0.3883 | **0.6532** |
| Training time | ~2 sec | ~12 sec |

---

## Why We Picked XGBoost

The R2 improvement is 0.2649 which is well above the 0.05 threshold we set.

Linear Regression performed very poorly here. The main reason is that trip duration does not have a simple linear relationship with distance — a 3km trip during evening rush hour can take 40 minutes while the same distance at 2am takes 8 minutes. Linear models cannot capture this kind of interaction between features. Even after applying log-transform on the target, the RMSE remained extremely high.

XGBoost handles this through tree splits which naturally capture non-linear patterns and feature interactions. The feature importance plot also confirms that `distance_km` is the strongest predictor (importance ~0.55) followed by `hour_of_day` (~0.20), which makes intuitive sense.

The XGBoost RMSE of 3,162 seconds (~52 minutes) is reasonable for a baseline model that uses only GPS coordinates and time features — no weather or traffic data is included. There is room for improvement in future iterations.

---

## Reproducibility

Re-running `python training/train.py` with the same `random_state=42` and the same `data/processed/features.csv` produces identical metrics every time. This was verified manually.

---

## MLflow Runs

Both runs are logged in the `eta-prediction` experiment in `mlruns/`.

| Run | RMSE | MAE | R2 |
|---|---|---|---|
| linear_regression | 282,405,528 s | 525,057 s | 0.3883 |
| xgboost | 3,162 s | 344 s | 0.6532 |

To view runs locally:
```
mlflow ui
```
Then open `http://127.0.0.1:5000` in a browser.

---

## Saved Artifacts

- `artifacts/model.pkl` — selected XGBoost model
- `artifacts/scaler.pkl` — StandardScaler fitted on training data only
- `artifacts/label_encoder.pkl` — LabelEncoder for pickup_hour_bin
- `artifacts/model_selection.json` — metrics + selection justification
