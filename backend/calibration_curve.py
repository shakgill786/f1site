import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ Paths
BASE         = os.path.dirname(__file__)
FEATURES_CSV = os.path.join(BASE, "data", "all_race_features_last5_and_current.csv")
MODEL_PATH   = os.path.join(BASE, "model_f1.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ Load data & model
df    = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])
model = joblib.load(MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ Time‐aware test split (last 20%)
split_idx = int(len(df) * 0.8)
NUMERIC   = [
    "avg_finish_5",
    "podium_pct_5",
    "avg_grid_5",
    "constructor_pts_5",
    "last_year_pos",
]
CATEGORICAL = ["driverId", "circuitId"]
FEATURES    = NUMERIC + CATEGORICAL

X_test = df.iloc[split_idx:][FEATURES]
y_test = df.iloc[split_idx:]["win"]

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ Predict “win” probabilities
probs = model.predict_proba(X_test)[:, 1]

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣ Compute calibration curve
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣ Plot
plt.figure(figsize=(6,6))
plt.plot(mean_pred, frac_pos, "s-", label="Predicted vs Actual")
plt.plot([0,1],[0,1],"--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve — Win vs Not Win")
plt.legend(loc="best")
plt.tight_layout()
plt.show()
