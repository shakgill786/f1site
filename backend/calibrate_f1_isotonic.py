# backend/calibrate_f1_isotonic.py

import os
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV

# ── 1️⃣ Paths ────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(BASE, "data", "all_race_features_last5_and_current.csv")
MODEL_IN     = os.path.join(BASE, "model_f1.pkl")             # your sigmoid‐calibrated model
MODEL_OUT    = os.path.join(BASE, "model_f1_isotonic.pkl")    # new isotonic model

# ── 2️⃣ Load hold-out data ────────────────────────────────────────────────────
df = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])
df = df.sort_values("raceDate").reset_index(drop=True)

# make binary “win” target
df["win"] = (df["position"] == 1).astype(int)

# split off final 20%
split = int(len(df) * 0.8)
hold = df.iloc[split:]

# the features your pipeline expects
FEATS = [
    "avg_finish_5",
    "podium_pct_5",
    "avg_grid_5",
    "constructor_pts_5",
    "last_year_pos",
    "driverId",
    "circuitId",
]

X_hold = hold[FEATS]
y_hold = hold["win"]

# ── 3️⃣ Load your pre-fitted (sigmoid) model ────────────────────────────────
prefit_calibrated = joblib.load(MODEL_IN)

# ── 4️⃣ Wrap in an isotonic CalibratedClassifierCV (cv="prefit") ────────────
iso_cal = CalibratedClassifierCV(
    estimator=prefit_calibrated,
    method="isotonic",
    cv="prefit"
)

# only fits the 1D isotonic map on your hold-out
iso_cal.fit(X_hold, y_hold)

# ── 5️⃣ Save the new isotonic‐calibrated model ──────────────────────────────
joblib.dump(iso_cal, MODEL_OUT)
print(f"✅ Saved isotonic‐calibrated model to {MODEL_OUT}")
