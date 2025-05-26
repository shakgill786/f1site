# train_f1_multiclass.py
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import top_k_accuracy_score, log_loss

# 1️⃣ Load & sort features (last-5-year + current)
df = pd.read_csv("data/all_race_features_last5_and_current.csv", parse_dates=["raceDate"])
df.sort_values(["driverId","raceDate"], inplace=True)

# 2️⃣ Define X, y (zero-based classes for finish: 0→1st, …,19→20th)
FEATURES_NUM = ["avg_finish_5","podium_pct_5","avg_grid_5","constructor_pts_5","last_year_pos"]
FEATURES_CAT = ["driverId","circuitId"]
X = df[FEATURES_NUM + FEATURES_CAT]
y = (df["position"] - 1).astype(int)    # shift 1→0, 2→1, …,20→19

# 3️⃣ Build pipeline & CV
pre = ColumnTransformer([
    ("num", StandardScaler(), FEATURES_NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
])
clf = XGBClassifier(
    objective="multi:softprob",
    num_class=20,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)
pipe = Pipeline([("prep", pre), ("clf", clf)])
tscv = TimeSeriesSplit(n_splits=5)

# 4️⃣ CV metrics
proba = cross_val_score(pipe, X, y, cv=tscv,
                       scoring="neg_root_mean_squared_error",   # placeholder
                       n_jobs=-1)
# but we'll get predictions on final split for top-k & log-loss:
split = int(len(df)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
pipe.fit(X_train, y_train)
y_proba = pipe.predict_proba(X_test)

print("Top-3 accuracy:", top_k_accuracy_score(y_test, y_proba, k=3))
print("Log-loss        :", log_loss(y_test, y_proba))

# 5️⃣ Save
joblib.dump(pipe, "model_f1_multiclass.pkl")
print("✅ Saved multiclass model")
