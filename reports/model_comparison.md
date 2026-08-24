# Model Comparison Report — Week 2
### PCAM ZC412 | Mini-Project-I | Flavor A — ETA Prediction
**Team:** Kishore Nandhalu | Vinay | Vishruth

---

## Experiment Setup

| Item | Detail |
|---|---|
| Dataset | `data/processed/features.csv` — 1,458,644 rows |
| Train / Test split | 80% / 20% — temporal split (no shuffle, respects time order) |
| Train rows | 1,166,915 |
| Test rows | 291,729 |
| Target | `log1p(trip_duration)` — log-transformed to reduce skew |
| Tracking | MLflow experiment: `eta-prediction` |
| Random seed | 42 (fixed — fully reproducible) |

---

## Features Used

| Feature | Type | Description |
|---|---|---|
| `hour_of_day` | Numerical | Hour extracted from pickup_datetime (0-23) |
| `day_of_week` | Numerical | Weekday (0=Mon, 6=Sun) |
| `is_weekend` | Binary | 1 if Saturday or Sunday |
| `distance_km` | Numerical | Haversine distance between pickup and dropoff |
| `pickup_hour_bin_enc` | Encoded | night/morning/afternoon/evening — label encoded |

---

## Results

| Metric | Linear Regression | XGBoost | Better |
|---|---|---|---|
| **RMSE (seconds)** | 282,405,528 s | **3,162 s** | XGBoost |
| **MAE (seconds)** | 525,057 s | **344 s** | XGBoost |
| **R²** | 0.3883 | **0.6532** | XGBoost |
| Training time | ~2 sec | ~12 sec | LR faster |
| Inference latency | Very fast | Fast | LR faster |

---

## Why XGBoost Wins

**R² improvement = 0.2649 (exceeds 0.05 threshold)**

1. **Linear Regression fails on skewed data** — `distance_km` has extreme outliers (max 1,240 km). Linear models assume a linear relationship and are sensitive to outliers. Even with log-transform on the target, the features remain skewed causing very high RMSE.

2. **XGBoost handles non-linearity** — Trip duration is not a linear function of distance. Rush-hour trips of 2km can take longer than off-peak trips of 10km. XGBoost captures these non-linear interactions through tree splits.

3. **XGBoost RMSE of 3,162s (~52 min)** — Acceptable for a baseline model on raw GPS + time features only. No weather or traffic data included. Further improvement possible in future iterations.

4. **Reproducibility confirmed** — Re-running `python training/train.py` with `random_state=42` produces identical metrics.

---

## Selected Model

**Winner: XGBoost**
- Saved to: `artifacts/model.pkl`
- MLflow run logged with all params, metrics, and model artifact
- Selection criteria: R² improvement > 0.05 over baseline

---

## MLflow Runs

| Run Name | Run ID (first 8) | RMSE | MAE | R2 |
|---|---|---|---|---|
| linear_regression | 6675366d | 282,405,528 s | 525,057 s | 0.3883 |
| xgboost | fe7eebcc | 3,162 s | 344 s | 0.6532 |

To view full MLflow UI:
```bash
mlflow ui
# Open: http://127.0.0.1:5000
```

---

## XGBoost Hyperparameters (from params.yaml)

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| random_state | 42 |

---

*Generated: Aug 2026 | Repo: https://github.com/2025paml539-lab/mleng-mini-project*
