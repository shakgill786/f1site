import os
import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Page config
st.set_page_config(page_title="🏎️ F1 Finish Predictor", layout="wide")
st.title("🏎️ F1 Next-Race Finish Predictor")

# ── Load enriched features
FEATURES_CSV = os.path.join(
    os.path.dirname(__file__),
    "data",
    "all_races_with_weather.csv"
)
df_all = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])
df_all.sort_values("raceDate", inplace=True)

# ── Sidebar: select driver & circuit
st.sidebar.header("Configure Prediction")
driver  = st.sidebar.selectbox("Driver",  sorted(df_all["driverId"].unique()))
circuit = st.sidebar.selectbox("Circuit", sorted(df_all["circuitId"].unique()))

# ── Driver‐history defaults
df_drv = df_all[df_all["driverId"] == driver]
latest = df_drv.iloc[-1] if not df_drv.empty else None

def _default(col):
    if latest is not None and pd.notna(latest[col]):
        return float(latest[col])
    return float(df_all[col].mean())

# ── Historical stats
st.sidebar.subheader("Driver Historical Stats")
avg_finish_5      = st.sidebar.slider("5-Race Avg Finish",
    float(df_all["avg_finish_5"].min()), float(df_all["avg_finish_5"].max()),
    value=_default("avg_finish_5"))
podium_pct_5      = st.sidebar.slider("5-Race Podium %",
    0.0, 1.0, value=_default("podium_pct_5"), step=0.01)
avg_grid_5        = st.sidebar.slider("5-Race Avg Grid",
    float(df_all["avg_grid_5"].min()), float(df_all["avg_grid_5"].max()),
    value=_default("avg_grid_5"))
constructor_pts_5 = st.sidebar.slider("5-Race Constructor Pts",
    float(df_all["constructor_pts_5"].min()), float(df_all["constructor_pts_5"].max()),
    value=_default("constructor_pts_5"))
last_year_pos     = st.sidebar.slider("Last Year’s Pos at This Track",
    int(df_all["last_year_pos"].min()), int(df_all["last_year_pos"].max()),
    value=int(_default("last_year_pos")))

# ── Weather & date
st.sidebar.subheader("Weather & Date Features")
# pick the next (or most recent) race at this circuit
now  = pd.Timestamp.now()
df_c = df_all[df_all["circuitId"] == circuit]
future = df_c[df_c["raceDate"] >= now]
nr = future.iloc[0] if not future.empty else df_c.iloc[-1]

temp_def   = float(nr["temp_avg_C"])
precip_def = float(nr["precip_mm"])
wind_def   = float(nr["wind_kph"])
hum_def    = float(nr["humidity_pct"])
race_month = nr["raceDate"].month

temp_avg_C   = st.sidebar.number_input("Avg Temp (°C)", value=temp_def)
precip_mm    = st.sidebar.number_input("Precipitation (mm)", value=precip_def)
wind_kph     = st.sidebar.number_input("Wind Speed (kph)", value=wind_def)
humidity_pct = st.sidebar.number_input("Humidity (%)", value=hum_def)

month_sin = np.sin(2*np.pi * race_month/12)
month_cos = np.cos(2*np.pi * race_month/12)
rain_temp = precip_mm * temp_avg_C

# ── Load pipelines
MODELS_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
rf_pipe     = joblib.load(os.path.join(MODELS_DIR, "rf_pipeline.joblib"))
xgb_pipe    = joblib.load(os.path.join(MODELS_DIR, "xgb_pipeline.joblib"))

# ── Live prediction via Flask API
if st.sidebar.button("Predict Finish"):
    payload = [{    
        "avg_finish_5":      avg_finish_5,
        "podium_pct_5":      podium_pct_5,
        "avg_grid_5":        avg_grid_5,
        "constructor_pts_5": constructor_pts_5,
        "last_year_pos":     last_year_pos,
        "temp_avg_C":        temp_avg_C,
        "precip_mm":         precip_mm,
        "wind_kph":          wind_kph,
        "humidity_pct":      humidity_pct,
        "rain_temp":         rain_temp,
        "month_sin":         month_sin,
        "month_cos":         month_cos,
        "circuitId":         circuit
    }]
    try:
        r = requests.post("http://localhost:5000/predict", json=payload, timeout=5)
        r.raise_for_status()
        out = r.json()
        st.metric("🏆 RF Predicted Finish",  f"{out['rf_predictions'][0]:.2f}")
        st.metric("🚀 XGB Predicted Finish", f"{out['xgb_predictions'][0]:.2f}")
    except Exception as e:
        st.error(f"API error: {e}")

# ── Recent‐form sparkline
st.subheader(f"{driver} — Last 10 Races: Avg Finish")
if not df_drv.empty:
    spark = df_drv.set_index("raceDate")["avg_finish_5"].tail(10)
    st.line_chart(spark)
else:
    st.info("No historical data for this driver yet.")

# ── Backtest historical performance
with st.expander("🔄 Backtest Historical Performance"):
    NUM_FEATS = [
        "avg_finish_5","podium_pct_5","avg_grid_5",
        "constructor_pts_5","last_year_pos",
        "temp_avg_C","precip_mm","wind_kph",
        "humidity_pct","rain_temp","month_sin","month_cos"
    ]
    CAT_FEATS = ["circuitId"]
    bt = df_all.copy()
    X_bt = bt[NUM_FEATS + CAT_FEATS]
    y_bt = bt["position"]

    rf_preds  = rf_pipe.predict(X_bt)
    xgb_preds = xgb_pipe.predict(X_bt)

    st.markdown("### RF Backtest Metrics")
    st.write(f"- RMSE: {mean_squared_error(y_bt, rf_preds):.2f}  "
             f"MAE: {mean_absolute_error(y_bt, rf_preds):.2f}  "
             f"R²: {r2_score(y_bt, rf_preds):.3f}")

    st.markdown("### XGB Backtest Metrics")
    st.write(f"- RMSE: {mean_squared_error(y_bt, xgb_preds):.2f}  "
             f"MAE: {mean_absolute_error(y_bt, xgb_preds):.2f}  "
             f"R²: {r2_score(y_bt, xgb_preds):.3f}")

    sample = bt[["raceDate","driverId","circuitId","position"]].copy()
    sample["RF"]  = rf_preds
    sample["XGB"] = xgb_preds
    st.dataframe(sample.set_index("raceDate"), use_container_width=True)
