# backend/train_global_model_f1.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

# ── Paths ────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
CSV     = os.path.join(BASE, "data", "all_race_features_last5_and_current.csv")
OUT_RAW = os.path.join(BASE, "model_f1_raw.pkl")
OUT_CAL = os.path.join(BASE, "model_f1.pkl")

# ── Load & sort ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV, parse_dates=["raceDate"])
df.sort_values("raceDate", inplace=True)

# ── Features & target ────────────────────────────────────────────────────
NUM = ["avg_finish_5","podium_pct_5","avg_grid_5","constructor_pts_5","last_year_pos"]
CAT = ["driverId","circuitId"]
X   = df[NUM + CAT]
y   = df["win"]

# ── Pipeline ─────────────────────────────────────────────────────────────
pre = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
lr  = LogisticRegression(max_iter=1000, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
ens = VotingClassifier([("lr",lr),("xgb",xgb)], voting="soft", n_jobs=-1)
pipe = Pipeline([("prep",pre),("clf",ens)])

# ── Time-aware split (80/20) ─────────────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Fit & evaluate raw ensemble ─────────────────────────────────────────
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:,1]
print("▶ Raw Accuracy:", accuracy_score(y_test, y_pred))
print("▶ Raw ROC AUC :", roc_auc_score(y_test, y_proba))
print("▶ Raw Brier   :", brier_score_loss(y_test, y_proba))

# ── Save raw model ───────────────────────────────────────────────────────
joblib.dump(pipe, OUT_RAW)
print(f"✅ Saved raw model to {OUT_RAW}")

# ── Calibrate on test fold ───────────────────────────────────────────────
cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
cal.fit(X_test, y_test)
y_cal = cal.predict_proba(X_test)[:,1]
print("▶ Cal ROC AUC:", roc_auc_score(y_test, y_cal))
print("▶ Cal Brier  :", brier_score_loss(y_test, y_cal))

# ── Save calibrated model ───────────────────────────────────────────────
joblib.dump(cal, OUT_CAL)
print(f"✅ Saved calibrated model to {OUT_CAL}")
