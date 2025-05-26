import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── 1) Load your calibrated multiclass F1 model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model_f1_multiclass_calibrated.pkl"
)
model = joblib.load(MODEL_PATH)

# ── 2) Define expected features
NUMERIC_FEATS     = [
    "avg_finish_5",
    "podium_pct_5",
    "avg_grid_5",
    "constructor_pts_5",
    "last_year_pos",
]
CATEGORICAL_FEATS = ["driverId", "circuitId"]
REQUIRED          = NUMERIC_FEATS + CATEGORICAL_FEATS

@app.route("/", methods=["GET"])
def home():
    return "🏎️ F1 Finish–Predictor API is up! POST JSON to /predict", 200

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({
            "usage": "POST JSON to /predict with keys: " + ", ".join(REQUIRED)
        }), 200

    data = request.get_json(force=True)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        return jsonify(error=f"Missing required fields: {missing}"), 400

    # ── 3) Build a single‐row DataFrame
    row = {}
    try:
        for k in NUMERIC_FEATS:
            row[k] = float(data[k])
        for k in CATEGORICAL_FEATS:
            row[k] = str(data[k])
    except Exception as e:
        return jsonify(error=f"Invalid value for feature: {e}"), 400

    X = pd.DataFrame([row])

    # ── 4) Predict full 20‐way finish distribution
    try:
        probs = model.predict_proba(X)[0]
        # map back to 1‐based finishing positions
        dist = { str(i+1): round(float(probs[i]), 4) for i in range(len(probs)) }
        return jsonify(position_probabilities=dist), 200
    except Exception as e:
        return jsonify(error=f"Prediction error: {e}"), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
