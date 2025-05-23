# app_f1.py
import os, joblib, pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model_f1.pkl"))

@app.route('/',methods=['GET'])
def home():
    return "F1 Win Predictor up", 200

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # expect keys: driverId, track, avg_finish_5, podium_pct_5, avg_grid_5, constructor_pts_5, last_year_pos
    X = pd.DataFrame([data])
    p = MODEL.predict_proba(X)[0,1]
    return jsonify(win_prob=round(float(p),3)),200

if __name__=='__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT',5000)),debug=True)
