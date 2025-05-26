# backend/train_global_model_f1_multiclass.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

# ── 1) Paths ────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
CSV_PATH  = os.path.join(BASE, "data", "all_race_features_last5_and_current.csv")
MODEL_OUT = os.path.join(BASE, "model_f1_multiclass.pkl")

# ── 2) Load & inspect ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, parse_dates=["raceDate"])
print("👉 Using feature file:", CSV_PATH)
print("👉 Columns available:", df.columns.tolist())

# ── 3) Time-sort & filter to positions 1–20 ─────────────────────────────────
df = df.sort_values("raceDate")
if "position" not in df.columns:
    raise RuntimeError("❌ 'position' column not found in your feature CSV.")
df = df[df["position"].between(1, 20)].copy()

# ── 4) Split out X and 0-based y ─────────────────────────────────────────────
NUMERIC     = ["avg_finish_5","podium_pct_5","avg_grid_5","constructor_pts_5","last_year_pos"]
CATEGORICAL = ["driverId","circuitId"]

X = df[NUMERIC + CATEGORICAL]
y = (df["position"] - 1).astype(int)   # map 1–20 → 0–19

# ── 5) Preprocessing & model ────────────────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
])
clf = XGBClassifier(
    objective="multi:softprob",
    num_class=20,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
)
pipeline = Pipeline([("prep", preprocessor), ("clf", clf)])

# ── 6) Time-aware split (80% train / 20% test) ───────────────────────────────
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# ── 7) Fit & evaluate ───────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_proba, labels=list(range(20)))

print(f"🚥 Test Accuracy : {acc:.4f}")
print(f"🚥 Test Log-Loss : {ll:.4f}")

# ── 8) Save your final model ─────────────────────────────────────────────────
joblib.dump(pipeline, MODEL_OUT)
print(f"✅ Saved multiclass model to {MODEL_OUT}")
