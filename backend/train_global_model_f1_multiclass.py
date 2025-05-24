# backend/train_global_model_f1_multiclass.py
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

# ── 1) Paths ───────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
CSV    = os.path.join(BASE, "data", "all_race_features_2010_2024.csv")
OUT    = os.path.join(BASE, "model_f1_multiclass.pkl")

# ── 2) Load & sort ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV, parse_dates=["raceDate"])
df = df.sort_values("raceDate")

# ── 3) Features & (zero-based) target ──────────────────────────────────────
NUM = ["avg_finish_5", "podium_pct_5", "avg_grid_5", "constructor_pts_5", "last_year_pos"]
CAT = ["driverId", "circuitId"]
X   = df[NUM + CAT]

# subtract 1 so positions 1…24 → labels 0…23
y_raw = df["position"].astype(int)
y     = (y_raw - 1).values
n_classes = y.max() + 1

# ── 4) Preprocessing ───────────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),            NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
])

# ── 5) XGB multi-class model ────────────────────────────────────────────────
clf = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
)
pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

# ── 6) Time-aware split (80/20) ─────────────────────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y[:split],       y[split:]

# ── 7) Fit & evaluate ──────────────────────────────────────────────────────
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)
pred  = pipe.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))
print(
    "Log Loss :", 
     log_loss(y_test, proba, labels=list(range(n_classes)))
)

# ── 8) Save ─────────────────────────────────────────────────────────────────
joblib.dump(pipe, OUT)
print(f"✅ Saved multi-class model to {OUT}")
