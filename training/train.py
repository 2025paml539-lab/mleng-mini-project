"""
training/train.py
Trains two models (Linear Regression + XGBoost) on engineered features.
All experiments logged to MLflow. Best model saved to artifacts/model.pkl.
Run: python training/train.py
"""

import os
import sys
import json
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
os.environ.pop("MLFLOW_TRACKING_URI", None)
mlflow.set_tracking_uri("mlruns")
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ── paths ──────────────────────────────────────────────────────────────────
FEATURES_PATH = os.path.join("data", "processed", "features.csv")
PARAMS_PATH   = os.path.join("params.yaml")
ARTIFACTS_DIR = os.path.join("artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── load params ────────────────────────────────────────────────────────────
with open(PARAMS_PATH) as f:
    params = yaml.safe_load(f)

RANDOM_STATE = params["train"]["random_state"]
TEST_SIZE    = params["train"]["test_size"]
XGB_PARAMS   = params["train"]["models"]["xgboost"]


def load_features():
    if not os.path.exists(FEATURES_PATH):
        print(f"[TRAIN] ERROR — {FEATURES_PATH} not found. Run pipeline/features.py first.")
        sys.exit(1)
    df = pd.read_csv(FEATURES_PATH)
    print(f"[TRAIN] Loaded features: {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df


def prepare_data(df):
    # Encode categorical feature
    le = LabelEncoder()
    df["pickup_hour_bin_enc"] = le.fit_transform(df["pickup_hour_bin"])
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

    feature_cols = ["hour_of_day", "day_of_week", "is_weekend",
                    "distance_km", "pickup_hour_bin_enc"]
    target_col   = "trip_duration"

    X = df[feature_cols]
    y = np.log1p(df[target_col])  # log-transform target — reduces skew

    # Temporal split — respect time order, no shuffle
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    print(f"[TRAIN] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Scale numerical features — fit on train only
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_sc  = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    print(f"[TRAIN] Scaler saved -> artifacts/scaler.pkl")

    return X_train_sc, X_test_sc, y_train, y_test, X_train, X_test


def compute_metrics(y_true, y_pred, label):
    # Back-transform from log space for interpretable metrics
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true, y_pred)
    print(f"[{label}] RMSE: {rmse:.1f}s | MAE: {mae:.1f}s | R²: {r2:.4f}")
    return {"rmse": round(rmse, 2), "mae": round(mae, 2), "r2": round(r2, 4)}


def train_linear_regression(X_train, X_test, y_train, y_test):
    print("\n[TRAIN] --- Model 1: Linear Regression ---")
    mlflow.set_experiment("eta-prediction")

    with mlflow.start_run(run_name="linear_regression") as run:
        mlflow.log_param("model_type",    "LinearRegression")
        mlflow.log_param("test_size",     TEST_SIZE)
        mlflow.log_param("random_state",  RANDOM_STATE)
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_param("features", ["hour_of_day", "day_of_week",
                                       "is_weekend", "distance_km",
                                       "pickup_hour_bin_enc"])

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, "LR")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"[TRAIN] MLflow run_id: {run_id}")
    return model, metrics, run_id


def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n[TRAIN] --- Model 2: XGBoost ---")
    mlflow.set_experiment("eta-prediction")

    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_param("model_type",       "XGBRegressor")
        mlflow.log_param("n_estimators",     XGB_PARAMS["n_estimators"])
        mlflow.log_param("max_depth",        XGB_PARAMS["max_depth"])
        mlflow.log_param("learning_rate",    XGB_PARAMS["learning_rate"])
        mlflow.log_param("random_state",     XGB_PARAMS["random_state"])
        mlflow.log_param("test_size",        TEST_SIZE)
        mlflow.log_param("target_transform", "log1p")

        model = XGBRegressor(
            n_estimators  = XGB_PARAMS["n_estimators"],
            max_depth     = XGB_PARAMS["max_depth"],
            learning_rate = XGB_PARAMS["learning_rate"],
            random_state  = XGB_PARAMS["random_state"],
            verbosity     = 0
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred, "XGB")
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        run_id = run.info.run_id
        print(f"[TRAIN] MLflow run_id: {run_id}")
    return model, metrics, run_id


def select_and_save_best(lr_model, lr_metrics, xgb_model, xgb_metrics):
    print("\n[TRAIN] --- Model Selection ---")
    lr_r2  = lr_metrics["r2"]
    xgb_r2 = xgb_metrics["r2"]
    improvement = xgb_r2 - lr_r2

    if improvement > 0.05:
        best_model  = xgb_model
        best_name   = "XGBoost"
        justification = f"XGBoost R² ({xgb_r2}) exceeds Linear Regression ({lr_r2}) by {improvement:.4f} > 0.05 threshold"
    else:
        best_model  = lr_model
        best_name   = "LinearRegression"
        justification = f"Improvement < 0.05 ({improvement:.4f}) — Linear Regression preferred (simpler, faster)"

    joblib.dump(best_model, os.path.join(ARTIFACTS_DIR, "model.pkl"))

    selection = {
        "best_model":      best_name,
        "lr_metrics":      lr_metrics,
        "xgb_metrics":     xgb_metrics,
        "r2_improvement":  round(improvement, 4),
        "justification":   justification
    }
    with open(os.path.join(ARTIFACTS_DIR, "model_selection.json"), "w") as f:
        json.dump(selection, f, indent=2)

    print(f"[TRAIN] Best model  : {best_name}")
    print(f"[TRAIN] Justification: {justification}")
    print(f"[TRAIN] Saved -> artifacts/model.pkl")
    return selection


def main():
    print("[TRAIN] Starting Week 2 training pipeline...\n")
    df = load_features()
    X_train, X_test, y_train, y_test, _, _ = prepare_data(df)

    lr_model,  lr_metrics,  lr_run_id  = train_linear_regression(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost(X_train, X_test, y_train, y_test)

    selection = select_and_save_best(lr_model, lr_metrics, xgb_model, xgb_metrics)

    print("\n[TRAIN] ======================================")
    print(f"[TRAIN] Linear Regression - RMSE: {lr_metrics['rmse']}s | MAE: {lr_metrics['mae']}s | R2: {lr_metrics['r2']}")
    print(f"[TRAIN] XGBoost           - RMSE: {xgb_metrics['rmse']}s | MAE: {xgb_metrics['mae']}s | R2: {xgb_metrics['r2']}")
    print(f"[TRAIN] Winner            - {selection['best_model']}")
    print("[TRAIN] MLflow UI -> run: mlflow ui  then open http://127.0.0.1:5000")
    print("[TRAIN] Done")


if __name__ == "__main__":
    main()
