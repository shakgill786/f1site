import os
import joblib
import pandas as pd

from sklearn.calibration      import CalibratedClassifierCV
from sklearn.metrics          import log_loss, top_k_accuracy_score

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ Paths
BASE            = os.path.dirname(__file__)
FEATURES_CSV    = os.path.join(BASE, "data", "all_race_features_last5_and_current.csv")
MODEL_IN        = os.path.join(BASE, "model_f1_multiclass.pkl")
MODEL_OUT       = os.path.join(BASE, "model_f1_multiclass_calibrated.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ Load features & true finish‐positions
df = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])
df.sort_values(["driverId", "raceDate"], inplace=True)

# build X, y (0-19 for 1st-20th)
X = df[[
    "avg_finish_5",
    "podium_pct_5",
    "avg_grid_5",
    "constructor_pts_5",
    "last_year_pos",
    "driverId",
    "circuitId",
]]
y = (df["position"] - 1).astype(int)

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ Split out the final 20% as our calibration set
split_idx = int(len(df) * 0.8)
X_cal     = X.iloc[split_idx:]
y_cal     = y.iloc[split_idx:]

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ Load your pre-trained multiclass pipeline
base_pipe = joblib.load(MODEL_IN)

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣ Calibrate with isotonic using the held-out slice
cal = CalibratedClassifierCV(
    estimator=base_pipe,
    method="isotonic",
    cv="prefit"
)
cal.fit(X_cal, y_cal)

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣ Quick eval on that same slice
y_proba = cal.predict_proba(X_cal)
print("🔢 Top-3 accuracy (calibrated):", top_k_accuracy_score(y_cal, y_proba, k=3))
print("🔢 Log-loss       (calibrated):", log_loss(y_cal, y_proba))

# ──────────────────────────────────────────────────────────────────────────────
# 7️⃣ Save calibrated multiclass model
joblib.dump(cal, MODEL_OUT)
print(f"✅ Saved calibrated multiclass model to {MODEL_OUT}")
