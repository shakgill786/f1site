# prepare_all_races_models.py

import os
from pathlib import Path
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# CONFIG
ROOT        = Path(__file__).parent
FEATURE_CSV = ROOT / "backend" / "data" / "all_races_with_weather_and_tyres.csv"
MODELS_DIR  = ROOT / "models"

# LOAD enriched features
df = pd.read_csv(FEATURE_CSV, parse_dates=["raceDate"])
df = df.dropna(subset=["position"])

# FEATURES & TARGET
NUM_FEATS = [
    "avg_finish_5","podium_pct_5","avg_grid_5",
    "constructor_pts_5","last_year_pos"
]
CAT_FEATS = ["circuitId","starting_tyre"]
X = df[NUM_FEATS + CAT_FEATS]
y = df["position"]

# PREPROCESSOR: impute + one‐hot
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATS),
])

# RF Pipeline
rf_pipe = Pipeline([
    ("pre", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=500,
        max_depth=3,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features=0.606451634075255,
        random_state=42
    ))
])

# XGB Pipeline
xgb_pipe = Pipeline([
    ("pre", preprocessor),
    ("model", XGBRegressor(
        n_estimators=306,
        max_depth=20,
        learning_rate=0.01,
        subsample=0.5,
        colsample_bytree=0.5,
        min_child_weight=10,
        tree_method="hist",
        eval_metric="rmse",
        random_state=42
    ))
])

# TRAIN & SAVE
rf_pipe.fit(X, y)
xgb_pipe.fit(X, y)

MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(rf_pipe,  MODELS_DIR/"rf_pipeline.joblib")
joblib.dump(xgb_pipe, MODELS_DIR/"xgb_pipeline.joblib")

print("✅ Pipelines saved to models/")
