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

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(__file__)
CSV_F   = os.path.join(BASE, "data", "all_races_features.csv")
MODEL_O = os.path.join(BASE, "model_f1.pkl")

# ── Load & sort ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_F, parse_dates=["raceDate"])
df.sort_values("raceDate", inplace=True)

# ── Features + target ──────────────────────────────────────────────────────
NUM = [
    "avg_finish_5","podium_pct_5","avg_grid_5",
    "constructor_pts_5","last_year_pos"
]
CAT = ["driverId","circuitId"]
X   = df[NUM + CAT]
y   = df["win"]

# ── Preprocessing ──────────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])

# ── Ensemble ───────────────────────────────────────────────────────────────
lr  = LogisticRegression(max_iter=1000, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
vc  = VotingClassifier([("lr", lr), ("xgb", xgb)], voting="soft", n_jobs=-1)

pipe = Pipeline([
    ("prep", preprocessor),
    ("clf",  vc),
])

# ── Train/test split (80/20 time‐aware) ─────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── Fit & evaluate ────────────────────────────────────────────────────────
pipe.fit(X_train, y_train)
pred  = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:,1]

print("Acc      :", accuracy_score(y_test,  pred))
print("ROC AUC  :", roc_auc_score(y_test, proba))
print("Brier    :", brier_score_loss(y_test, proba))

# ── Calibrate on test fold ─────────────────────────────────────────────────
calibrator = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
calibrator.fit(X_test, y_test)

joblib.dump(calibrator, MODEL_O)
print(f"✅ Saved calibrated F1 model to {MODEL_O}")
