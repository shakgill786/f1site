# backend/threshold_tuning.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    f1_score,
)

# ── 0️⃣ Paths ───────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
FEATURE_CSV = HERE / "data" / "all_race_features_last5_and_current.csv"
MODEL_PATH  = HERE / "model_f1.pkl"

# ── 1️⃣ Load your features + binary target ──────────────────────────────────
df = pd.read_csv(FEATURE_CSV, parse_dates=["raceDate"])
df = df.sort_values("raceDate")

# ── 2️⃣ Define X & y (use the exact same feature names your pipeline expects) ─
NUMERIC     = [
    "avg_finish_5",
    "podium_pct_5",
    "avg_grid_5",
    "constructor_pts_5",
    "last_year_pos",
]
CATEGORICAL = ["driverId", "circuitId"]   # <— must match your training pipeline
FEATURES    = NUMERIC + CATEGORICAL

X = df[FEATURES]
y = df["win"]  # 1 if driver won, 0 otherwise

# ── 3️⃣ Time‐aware split (80% train / 20% test) ─────────────────────────────
split_idx      = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx],  X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx],  y.iloc[split_idx:]

# ── 4️⃣ Load your calibrated pipeline ────────────────────────────────────────
pipe = joblib.load(MODEL_PATH)

# ── 5️⃣ Score probabilities on the hold-out set ─────────────────────────────
y_proba = pipe.predict_proba(X_test)[:, 1]

# ── 6️⃣ Precision–Recall vs Threshold ───────────────────────────────────────
prec, rec, th_pr = precision_recall_curve(y_test, y_proba)

# compute F1 at each threshold
f1s = [f1_score(y_test, y_proba >= t) for t in th_pr]
best_idx = np.argmax(f1s)
best_thr = th_pr[best_idx]
best_f1  = f1s[best_idx]

plt.figure(figsize=(8, 4))
plt.plot(th_pr, prec[:-1], label="Precision")
plt.plot(th_pr, rec[:-1],  label="Recall")
plt.axvline(best_thr, color="gray", linestyle="--",
            label=f"Best F1={best_f1:.2f} @ {best_thr:.2f}")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Decision Threshold")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"✅ Best F1 = {best_f1:.3f} at threshold = {best_thr:.2f}")

# ── 7️⃣ ROC Curve ───────────────────────────────────────────────────────────
fpr, tpr, th_roc = roc_curve(y_test, y_proba)
auc = np.trapz(tpr, fpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 8️⃣ Brier Score ─────────────────────────────────────────────────────────
brier = brier_score_loss(y_test, y_proba)
print(f"✅ Brier score = {brier:.4f}")
