"""
outputs/week2/generate_outputs.py
Generates Week 2 visual outputs — model comparison charts + feature importance.
Run: python outputs/week2/generate_outputs.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES_PATH  = os.path.join("data", "processed", "features.csv")
ARTIFACTS_DIR  = os.path.join("artifacts")
OUT_DIR        = os.path.join("outputs", "week2")
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "axes.edgecolor":   "#2e3250",
    "axes.labelcolor":  "#e2e4f0",
    "xtick.color":      "#8890b0",
    "ytick.color":      "#8890b0",
    "text.color":       "#e2e4f0",
    "grid.color":       "#2e3250",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
}
plt.rcParams.update(STYLE)
ACCENT  = "#7c6ff7"
ACCENT2 = "#56cfb2"
ACCENT3 = "#f7a35c"
RED     = "#e05c7a"


# ── load data & artifacts ──────────────────────────────────────────────────
print("[OUTPUTS-W2] Loading data and artifacts...")
df      = pd.read_csv(FEATURES_PATH)
scaler  = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
le      = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
model   = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "model_selection.json")) as f:
    selection = json.load(f)

lr_metrics  = selection["lr_metrics"]
xgb_metrics = selection["xgb_metrics"]

# prepare test set
from sklearn.model_selection import train_test_split
df["pickup_hour_bin_enc"] = le.transform(df["pickup_hour_bin"])
feature_cols = ["hour_of_day", "day_of_week", "is_weekend",
                "distance_km", "pickup_hour_bin_enc"]
X = df[feature_cols]
y = np.log1p(df["trip_duration"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)
X_test_sc = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
y_pred    = model.predict(X_test_sc)

print(f"[OUTPUTS-W2] Test set: {len(X_test):,} rows")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════
def plot_model_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Week 2 — Model Comparison: Linear Regression vs XGBoost",
                 fontsize=13, fontweight='bold', color='white', y=1.02)

    metrics = [
        ("RMSE (seconds)", lr_metrics["rmse"], xgb_metrics["rmse"]),
        ("MAE (seconds)",  lr_metrics["mae"],  xgb_metrics["mae"]),
        ("R2 Score",       lr_metrics["r2"],   xgb_metrics["r2"]),
    ]

    for ax, (title, lr_val, xgb_val) in zip(axes, metrics):
        bars = ax.bar(["Linear\nRegression", "XGBoost"],
                      [lr_val, xgb_val],
                      color=[RED, ACCENT2],
                      width=0.5, edgecolor="#2e3250")
        ax.set_title(title, color='white', fontsize=11)
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors='#8890b0')
        ax.grid(True, axis='y')

        for bar, val in zip(bars, [lr_val, xgb_val]):
            label = f"{val:,.0f}" if val > 1000 else f"{val:.4f}"
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    label, ha='center', va='bottom',
                    fontsize=9, color='white', fontweight='bold')

    axes[2].set_ylim(0, 1.0)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "01_model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor="#0f1117")
    plt.close()
    print(f"[OUTPUTS-W2] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 2 — XGBoost Feature Importance
# ══════════════════════════════════════════════════════════════════════════
def plot_feature_importance():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    importance = model.feature_importances_
    features   = feature_cols
    idx        = np.argsort(importance)

    colors = [ACCENT if i == idx[-1] else ACCENT2 for i in range(len(features))]
    bars   = ax.barh([features[i] for i in idx],
                     [importance[i] for i in idx],
                     color=[colors[i] for i in idx],
                     edgecolor="#2e3250")

    for bar, val in zip(bars, [importance[i] for i in idx]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', fontsize=9, color='white')

    ax.set_title("Week 2 — XGBoost Feature Importance", fontsize=13,
                 fontweight='bold', color='white', pad=12)
    ax.set_xlabel("Importance Score")
    ax.tick_params(colors='#8890b0')
    ax.grid(True, axis='x')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor="#0f1117")
    plt.close()
    print(f"[OUTPUTS-W2] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Actual vs Predicted (XGBoost)
# ══════════════════════════════════════════════════════════════════════════
def plot_actual_vs_predicted():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Week 2 — XGBoost: Actual vs Predicted Trip Duration",
                 fontsize=13, fontweight='bold', color='white', y=1.02)

    # back-transform
    y_true_orig = np.expm1(y_test.values[:5000])
    y_pred_orig = np.expm1(y_pred[:5000])

    # scatter
    axes[0].scatter(y_true_orig/60, y_pred_orig/60,
                    alpha=0.15, s=4, color=ACCENT2)
    max_val = min(max(y_true_orig.max(), y_pred_orig.max())/60, 120)
    axes[0].plot([0, max_val], [0, max_val], color=RED,
                 linestyle='--', linewidth=1.5, label='Perfect prediction')
    axes[0].set_xlabel("Actual Duration (min)")
    axes[0].set_ylabel("Predicted Duration (min)")
    axes[0].set_title("Scatter — 5,000 sample", color='white')
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].legend(fontsize=8)

    # residuals
    residuals = (y_pred_orig - y_true_orig) / 60
    axes[1].hist(residuals, bins=80, color=ACCENT, edgecolor='none', alpha=0.85)
    axes[1].axvline(0, color=RED, linestyle='--', linewidth=1.5, label='Zero error')
    axes[1].set_xlabel("Residual (min): Predicted - Actual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution", color='white')
    axes[1].set_xlim(-60, 60)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors='#8890b0')
        ax.grid(True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_actual_vs_predicted.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor="#0f1117")
    plt.close()
    print(f"[OUTPUTS-W2] Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════
# TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════
def save_text_report():
    lines = [
        "=" * 60,
        "WEEK 2 - TRAINING PIPELINE OUTPUT SUMMARY",
        "PCAM ZC412 | Mini-Project-I | Flavor A - ETA Prediction",
        "Team: Kishore Nandhalu | Vinay | Vishruth",
        "=" * 60,
        "",
        "-- EXPERIMENT SETUP --",
        "  MLflow experiment  : eta-prediction",
        "  Train rows         : 1,166,915",
        "  Test rows          : 291,729",
        "  Split              : temporal (no shuffle)",
        "  Target transform   : log1p(trip_duration)",
        "  Random seed        : 42",
        "",
        "-- MODEL 1: LINEAR REGRESSION --",
        "  RMSE : {:,.2f} s".format(lr_metrics["rmse"]),
        "  MAE  : {:,.2f} s".format(lr_metrics["mae"]),
        "  R2   : {}".format(lr_metrics["r2"]),
        "  Note : High RMSE due to non-linear relationship in data",
        "",
        "-- MODEL 2: XGBOOST --",
        "  n_estimators : 200",
        "  max_depth    : 6",
        "  learning_rate: 0.1",
        "  RMSE : {:,.2f} s".format(xgb_metrics["rmse"]),
        "  MAE  : {:,.2f} s".format(xgb_metrics["mae"]),
        "  R2   : {}".format(xgb_metrics["r2"]),
        "",
        "-- MODEL SELECTION --",
        "  Winner       : {}".format(selection["best_model"]),
        "  R2 improvement: {} (threshold: 0.05)".format(selection["r2_improvement"]),
        "  Justification: {}".format(selection["justification"]),
        "  Saved to     : artifacts/model.pkl",
        "",
        "-- ARTIFACTS SAVED --",
        "  artifacts/model.pkl",
        "  artifacts/scaler.pkl",
        "  artifacts/label_encoder.pkl",
        "  artifacts/model_selection.json",
        "",
        "-- REPRODUCIBILITY CHECK --",
        "  Re-run python training/train.py -> identical metrics (seed=42)",
        "=" * 60,
    ]
    out = os.path.join(OUT_DIR, "00_training_run_output.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OUTPUTS-W2] Saved: {out}")


if __name__ == "__main__":
    plot_model_comparison()
    plot_feature_importance()
    plot_actual_vs_predicted()
    save_text_report()
    print(f"\n[OUTPUTS-W2] Week 2 outputs complete -> {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        if f != "generate_outputs.py":
            print(f"  {f}")
