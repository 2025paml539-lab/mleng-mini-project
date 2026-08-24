"""
outputs/week2/generate_outputs.py
Week 2 visual outputs - model comparison, feature importance, actual vs predicted.
Run from project root: python outputs/week2/generate_outputs.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

FEATURES_PATH = os.path.join("data", "processed", "features.csv")
ARTIFACTS_DIR = os.path.join("artifacts")
OUT_DIR       = os.path.join("outputs", "week2")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading artifacts...")
model  = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
le     = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "model_selection.json")) as f:
    sel = json.load(f)

lr  = sel["lr_metrics"]
xgb = sel["xgb_metrics"]

# prepare test set (same split as training)
df = pd.read_csv(FEATURES_PATH)
df["pickup_hour_bin_enc"] = le.transform(df["pickup_hour_bin"])
feature_cols = ["hour_of_day", "day_of_week", "is_weekend",
                "distance_km", "pickup_hour_bin_enc"]
X = df[feature_cols]
y = np.log1p(df["trip_duration"])
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
y_pred    = model.predict(X_test_sc)
print(f"Test set: {len(X_test):,} rows")


# Plot 1 - Model comparison bar chart
def plot_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    metrics = [
        ("RMSE (seconds)", lr["rmse"], xgb["rmse"]),
        ("MAE (seconds)",  lr["mae"],  xgb["mae"]),
        ("R2 Score",       lr["r2"],   xgb["r2"]),
    ]

    for ax, (title, lr_val, xgb_val) in zip(axes, metrics):
        bars = ax.bar(["Linear Regression", "XGBoost"],
                      [lr_val, xgb_val],
                      color=["salmon", "steelblue"],
                      width=0.4, edgecolor="gray")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, [lr_val, xgb_val]):
            label = f"{val:,.0f}" if val > 100 else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    label, ha="center", va="bottom", fontsize=9)

    axes[2].set_ylim(0, 1.0)
    plt.suptitle("Week 2 - Model Comparison: Linear Regression vs XGBoost", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_model_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 01_model_comparison.png")


# Plot 2 - XGBoost feature importance
def plot_feature_importance():
    importance = model.feature_importances_
    idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([feature_cols[i] for i in idx],
            [importance[i] for i in idx],
            color="steelblue", edgecolor="gray")
    for i, val in enumerate([importance[j] for j in idx]):
        ax.text(val + 0.002, i, f"{val:.4f}", va="center", fontsize=9)
    ax.set_title("Week 2 - XGBoost Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_feature_importance.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 02_feature_importance.png")


# Plot 3 - Actual vs predicted + residuals
def plot_actual_vs_predicted():
    y_true = np.expm1(y_test.values[:4000])
    y_pr   = np.expm1(y_pred[:4000])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_true / 60, y_pr / 60, alpha=0.2, s=5, color="teal")
    max_val = min(max(y_true.max(), y_pr.max()) / 60, 100)
    axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="Perfect fit")
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_xlabel("Actual Duration (min)")
    axes[0].set_ylabel("Predicted Duration (min)")
    axes[0].set_title("Actual vs Predicted (4k sample)")
    axes[0].legend()

    residuals = (y_pr - y_true) / 60
    axes[1].hist(residuals, bins=60, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlim(-50, 50)
    axes[1].set_xlabel("Residual (min): Predicted - Actual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.suptitle("Week 2 - XGBoost Prediction Quality", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_actual_vs_predicted.png"), dpi=130, bbox_inches="tight")
    plt.close()
    print("Saved: 03_actual_vs_predicted.png")


# Text report
def save_text_report():
    lines = [
        "WEEK 2 - TRAINING RUN SUMMARY",
        "PCAM ZC412 | Mini-Project-I | Flavor A",
        "Team: Kishore Nandhalu | Vinay | Vishruth",
        "",
        "Experiment: eta-prediction (MLflow)",
        "Train rows : 1,166,915",
        "Test rows  : 291,729",
        "Split      : temporal, no shuffle",
        "Target     : log1p(trip_duration)",
        "Seed       : 42",
        "",
        "Model 1 - Linear Regression",
        "  RMSE : {:,.2f} s".format(lr["rmse"]),
        "  MAE  : {:,.2f} s".format(lr["mae"]),
        "  R2   : {}".format(lr["r2"]),
        "  Note : High RMSE - linear model struggles with non-linear distance relationship",
        "",
        "Model 2 - XGBoost",
        "  n_estimators : 200  max_depth : 6  learning_rate : 0.1",
        "  RMSE : {:,.2f} s".format(xgb["rmse"]),
        "  MAE  : {:,.2f} s".format(xgb["mae"]),
        "  R2   : {}".format(xgb["r2"]),
        "",
        "Selected : XGBoost",
        "Reason   : R2 improvement = {} (threshold 0.05)".format(sel["r2_improvement"]),
        "Saved to : artifacts/model.pkl",
        "",
        "Artifacts saved:",
        "  artifacts/model.pkl",
        "  artifacts/scaler.pkl",
        "  artifacts/label_encoder.pkl",
        "  artifacts/model_selection.json",
    ]
    out = os.path.join(OUT_DIR, "00_training_run_output.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved: 00_training_run_output.txt")


if __name__ == "__main__":
    plot_model_comparison()
    plot_feature_importance()
    plot_actual_vs_predicted()
    save_text_report()
    print("\nAll Week 2 outputs saved to", OUT_DIR)
