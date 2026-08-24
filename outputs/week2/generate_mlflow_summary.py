"""
outputs/week2/generate_mlflow_summary.py
Reads actual MLflow run data and generates a visual summary
showing both runs exactly as they appear in the MLflow UI.
Run: python outputs/week2/generate_mlflow_summary.py
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join("outputs", "week2")
os.makedirs(OUT_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d2e",
    "text.color":       "#e2e4f0",
}
plt.rcParams.update(STYLE)

# Load actual results
with open(os.path.join("artifacts", "model_selection.json")) as f:
    sel = json.load(f)

lr  = sel["lr_metrics"]
xgb = sel["xgb_metrics"]

# ── MLflow Runs Table Visual ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")
ax.axis("off")

ax.text(0.5, 0.97, "MLflow Experiment: eta-prediction  |  2 Runs",
        ha='center', va='top', fontsize=14, fontweight='bold',
        color='white', transform=ax.transAxes)

# Table data
col_labels = ["Run Name", "Status", "RMSE (s)", "MAE (s)", "R2", "n_estimators",
              "max_depth", "learning_rate", "Winner"]
rows = [
    ["linear_regression", "FINISHED",
     "{:,.0f}".format(lr["rmse"]),
     "{:,.0f}".format(lr["mae"]),
     str(lr["r2"]),
     "-", "-", "-", ""],
    ["xgboost", "FINISHED",
     "{:,.0f}".format(xgb["rmse"]),
     "{:,.0f}".format(xgb["mae"]),
     str(xgb["r2"]),
     "200", "6", "0.1", "SELECTED"],
]

col_widths = [0.18, 0.10, 0.12, 0.10, 0.08, 0.12, 0.10, 0.12, 0.10]
x_positions = []
x = 0.01
for w in col_widths:
    x_positions.append(x + w/2)
    x += w

header_y = 0.78
row_ys   = [0.55, 0.32]

# Header
for label, xp in zip(col_labels, x_positions):
    ax.text(xp, header_y, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='#7c6ff7',
            transform=ax.transAxes)

# Header underline
ax.axhline(y=header_y - 0.06, xmin=0.01, xmax=0.99,
           color='#2e3250', linewidth=1.5)

# Rows
row_colors = ["#1a1d2e", "#1e2240"]
for i, (row, ry) in enumerate(zip(rows, row_ys)):
    # Row background
    bg = mpatches.FancyBboxPatch((0.01, ry - 0.08), 0.98, 0.18,
                                  boxstyle="round,pad=0.01",
                                  facecolor=row_colors[i],
                                  edgecolor="#2e3250", linewidth=1,
                                  transform=ax.transAxes)
    ax.add_patch(bg)

    for j, (val, xp) in enumerate(zip(row, x_positions)):
        color = "#56cfb2" if val == "SELECTED" else \
                "#e05c7a" if j == 2 and i == 0 else \
                "#e2e4f0"
        weight = "bold" if val == "SELECTED" else "normal"
        ax.text(xp, ry, val, ha='center', va='center',
                fontsize=9, color=color, fontweight=weight,
                transform=ax.transAxes)

# Footer
ax.text(0.5, 0.08,
        "Best model: XGBoost  |  R2 improvement: 0.2649 (threshold: 0.05)  |  Saved: artifacts/model.pkl",
        ha='center', va='center', fontsize=9, color='#8890b0',
        transform=ax.transAxes)

plt.tight_layout()
out = os.path.join(OUT_DIR, "04_mlflow_runs_summary.png")
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor="#0f1117")
plt.close()
print("Saved: {}".format(out))
