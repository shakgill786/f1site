# backend/app_predict.py

import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ─── Load trained pipelines ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))

rf_pipeline  = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
xgb_pipeline = joblib.load(os.path.join(MODELS_DIR, "xgb_pipeline.joblib"))

# ─── Define the raw feature names we expect ────────────────────────────────────
NUM_FEATS = [
    "avg_finish_5","podium_pct_5","avg_grid_5",
    "constructor_pts_5","last_year_pos",
    "temp_avg_C","precip_mm","wind_kph","humidity_pct",
    "rain_temp","month_sin","month_cos"
]
CAT_FEATS = ["circuitId","starting_tyre"]
EXPECTED  = NUM_FEATS + CAT_FEATS

# ─── Health check ───────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return "🏎️ F1 RF & XGB Prediction API is running!", 200

# ─── Prediction endpoint ────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify(error="Input must be a JSON list of feature‐dicts"), 400

    df = pd.DataFrame(data)
    missing = [c for c in EXPECTED if c not in df.columns]
    if missing:
        return jsonify(error=f"Missing features: {missing}"), 400

    X = df[EXPECTED]
    try:
        rf_preds  = rf_pipeline.predict(X)
        xgb_preds = xgb_pipeline.predict(X)
        return jsonify({
            "rf_predictions":  [float(p) for p in rf_preds],
            "xgb_predictions": [float(p) for p in xgb_preds]
        })
    except Exception as e:
        return jsonify(error=f"Prediction failed: {e}"), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
