"""
test_week2.py - Unit tests for Week 2
Run: python test_week2.py
"""
import os, sys, json, joblib
import pandas as pd
import numpy as np
import mlflow

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
mlflow.set_tracking_uri("mlruns")

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

print("=" * 50)
print("Week 2 Unit Tests")
print("=" * 50)

# T1 - All artifacts exist
print("\n[T1] Artifacts")
for f in ["model.pkl", "scaler.pkl", "label_encoder.pkl", "model_selection.json"]:
    check(f"artifacts/{f} exists", os.path.exists(os.path.join("artifacts", f)))

# T2 - Model loads and predicts
print("\n[T2] Model prediction")
try:
    model  = joblib.load("artifacts/model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    le     = joblib.load("artifacts/label_encoder.pkl")
    feat   = pd.read_csv("data/processed/features.csv")
    feat["pickup_hour_bin_enc"] = le.transform(feat["pickup_hour_bin"])
    cols   = ["hour_of_day", "day_of_week", "is_weekend", "distance_km", "pickup_hour_bin_enc"]
    X      = feat[cols].head(5)
    X_sc   = pd.DataFrame(scaler.transform(X), columns=cols)
    preds  = model.predict(X_sc)
    check("model.predict returns 5 values", len(preds) == 5)
    check("all predictions > 0", all(p > 0 for p in preds),
          f"got {preds.tolist()}")
    check("predictions in reasonable range (log space)",
          all(5 < p < 10 for p in preds),
          f"got {[round(float(p),2) for p in preds]}")
    print(f"         Predicted log-durations: {[round(float(p),2) for p in preds]}")
    print(f"         Predicted seconds:       {[round(float(np.expm1(p)),0) for p in preds]}")
except Exception as e:
    check("model load and predict", False, str(e))

# T3 - Model selection JSON
print("\n[T3] Model selection")
sel = json.load(open("artifacts/model_selection.json"))
check("best_model is XGBoost",      sel["best_model"] == "XGBoost")
check("xgb R2 = 0.6532",            sel["xgb_metrics"]["r2"] == 0.6532)
check("lr R2 = 0.3883",             sel["lr_metrics"]["r2"] == 0.3883)
check("r2_improvement > 0.05",      sel["r2_improvement"] > 0.05)

# T4 - MLflow runs
print("\n[T4] MLflow experiment tracking")
try:
    client   = mlflow.tracking.MlflowClient()
    all_runs = []
    for exp in client.search_experiments():
        all_runs += client.search_runs([exp.experiment_id])
    names    = [r.info.run_name for r in all_runs]
    statuses = [r.info.status   for r in all_runs]
    check("linear_regression run exists", any("linear" in n for n in names))
    check("xgboost run exists",           any("xgboost" in n for n in names))
    check("all runs FINISHED",            all(s == "FINISHED" for s in statuses),
          f"statuses: {statuses}")
    lr_run  = next((r for r in all_runs if "linear" in r.info.run_name), None)
    xgb_run = next((r for r in all_runs if "xgboost" in r.info.run_name), None)
    if lr_run:
        check("LR run has rmse metric",   "rmse" in lr_run.data.metrics)
        check("LR run has r2 metric",     "r2"   in lr_run.data.metrics)
    if xgb_run:
        check("XGB run has rmse metric",  "rmse" in xgb_run.data.metrics)
        check("XGB run has r2 metric",    "r2"   in xgb_run.data.metrics)
        check("XGB rmse < LR rmse",
              xgb_run.data.metrics["rmse"] < lr_run.data.metrics["rmse"])
except Exception as e:
    check("MLflow client", False, str(e))

# T5 - Data files
print("\n[T5] Data files")
check("features.csv exists",         os.path.exists("data/processed/features.csv"))
check("train.csv.dvc exists",        os.path.exists("data/raw/train.csv.dvc"))
check("feature_schema.json exists",  os.path.exists("artifacts/feature_schema.json"))

# Summary
print("\n" + "=" * 50)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED - Week 2 is fully working")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
