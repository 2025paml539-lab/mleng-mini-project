# How to Run This Project and See All Outputs
### For Teammates — After Cloning from GitHub

**Repository URL:**
```
https://github.com/2025paml539-lab/mleng-mini-project
```

---

## Step 1 — Prerequisites (Install These First)

| Software | Version | Download |
|---|---|---|
| Python | 3.10 or higher | https://www.python.org/downloads/ |
| Git | Any recent | https://git-scm.com/download/win |

**During Python install:** tick the box **"Add Python to PATH"** before clicking Install.

Check they are installed — open PowerShell and run:
```powershell
python --version
git --version
pip --version
```

---

## Step 2 — Clone the Repository

```powershell
git clone https://github.com/2025paml539-lab/mleng-mini-project.git
cd mleng-mini-project
```

---

## Step 3 — Create Virtual Environment and Install Dependencies

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 4 — Download the Dataset

The raw dataset is NOT stored in Git (too large — 191 MB). Download it manually:

1. Go to: https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data
2. Create a free Kaggle account if you don't have one
3. Click "Join Competition" (free)
4. Download `train.zip`
5. Extract it — you get `train.csv`
6. Place it here:
```
mleng-mini-project\data\raw\train.csv
```

---

## Step 5 — Run Week 1 Pipeline

Run these 3 commands from the project root folder:

```powershell
python pipeline/ingest.py
python pipeline/validate.py
python pipeline/features.py
```

**What you will see:**
```
[INGEST] Loaded     : 1,458,644 rows x 11 cols
[INGEST] Nulls      : 0 — all columns complete
[INGEST] trip_duration — min: 1s  max: 3526282s  mean: 959s

[L1 SCHEMA]   PASS
[L2 RANGE]    WARN — 689 rows removed (0.047%)
[L3 STATS]    PASS — 1,457,955 rows | mean 959s
[L4 BUSINESS] PASS
[VALIDATE] All checks passed

[FEATURES] Saved: data/processed/features.csv (1,458,644 rows x 6 cols)
[FEATURES] Saved: artifacts/feature_schema.json
```

**Generate Week 1 charts:**
```powershell
python outputs/week1/generate_outputs.py
```

Charts saved to `outputs/week1/`:
- `01_validation_summary.png` — 4-level validation results
- `02_trip_duration_distribution.png` — raw and log-transformed duration
- `03_engineered_features_overview.png` — all 5 features
- `04_nyc_pickup_heatmap.png` — NYC GPS pickup density map

---

## Step 6 — Run Week 2 Training

```powershell
python training/train.py
```

**What you will see:**
```
[TRAIN] Train: 1,166,915 rows | Test: 291,729 rows
[LR]  RMSE: 282,405,528s | MAE: 525,057s | R2: 0.3883
[XGB] RMSE: 3,162s       | MAE: 344s     | R2: 0.6532
[TRAIN] Winner - XGBoost
[TRAIN] Saved -> artifacts/model.pkl
```

**Generate Week 2 charts:**
```powershell
python outputs/week2/generate_outputs.py
```

Charts saved to `outputs/week2/`:
- `01_model_comparison.png` — RMSE/MAE/R2 side by side
- `02_feature_importance.png` — XGBoost feature importance
- `03_actual_vs_predicted.png` — scatter plot and residuals
- `00_training_run_output.txt` — full training summary

**View MLflow experiment tracking UI:**
```powershell
mlflow ui --backend-store-uri mlruns --port 5000
```
Open browser: `http://127.0.0.1:5000`
Click **Experiments** → **eta-prediction** → **Training runs**

---

## Step 7 — Run Unit Tests (Verify Everything Works)

```powershell
python test_week2.py
```

Expected output:
```
Week 2 Unit Tests
[T1] Artifacts       — 4 PASS
[T2] Model prediction — 3 PASS
[T3] Model selection  — 4 PASS
[T4] MLflow runs      — 7 PASS
[T5] Data files       — 3 PASS
Results: 22 passed, 0 failed
ALL TESTS PASSED
```

---

## What Each Output File Shows

### outputs/week1/

| File | What it shows |
|---|---|
| `01_validation_summary.png` | 4-level data validation — L1 Schema, L2 Range, L3 Stats, L4 Business rules |
| `02_trip_duration_distribution.png` | Raw distribution and log-transformed target used for training |
| `03_engineered_features_overview.png` | All 5 engineered features — hour, day, weekend, distance, hour bin |
| `04_nyc_pickup_heatmap.png` | NYC map showing where most trips start |
| `00_pipeline_run_output.txt` | Full text log of ingest + validate + features run |

### outputs/week2/

| File | What it shows |
|---|---|
| `01_model_comparison.png` | RMSE, MAE, R2 comparison — Linear Regression vs XGBoost |
| `02_feature_importance.png` | Which features matter most for XGBoost (distance_km is #1) |
| `03_actual_vs_predicted.png` | Scatter plot of predictions vs actual durations |
| `04_mlflow_home_eta_experiment.png` | MLflow UI home — eta-prediction experiment visible |
| `05_mlflow_training_runs_list.png` | MLflow training runs list — all runs logged |
| `06_mlflow_training_runs_detail.png` | Run detail view |
| `07_mlflow_xgboost_metrics.png` | XGBoost metrics in MLflow: mae=344, r2=0.65, rmse=3162 |
| `08_mlflow_lr_metrics.png` | Linear Regression metrics: mae=525057, r2=0.39, rmse=282M |
| `00_training_run_output.txt` | Full training summary with both model metrics |

### outputs/week3/ — Coming (FastAPI + Docker)
### outputs/week4/ — Coming (Monitoring + Drift)

---

## Troubleshooting

**"python not found"**
Python was installed without adding to PATH. Re-install and tick "Add Python to PATH".

**"No module named pandas"**
Run `pip install -r requirements.txt` again after activating venv.

**"features.csv not found"**
Run `python pipeline/features.py` first before training.

**"train.csv not found"**
Download dataset from Kaggle and place in `data/raw/train.csv`.

**MLflow UI shows empty experiments**
Make sure you run `mlflow ui` from inside the `mleng-mini-project` folder, not from any other directory.

---

*PCAM ZC412 | Mini-Project-I | Team: Kishore Nandhalu | Vinay | Vishruth*
