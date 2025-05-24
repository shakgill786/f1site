# backend/dashboard_f1.py

import os
import pandas as pd
import joblib
import streamlit as st

# ── 1️⃣ Page config
st.set_page_config(page_title="F1 Next-Race Finish Predictor", layout="wide")

# ── 2️⃣ Load your multiclass model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_f1_multiclass.pkl")
model      = joblib.load(MODEL_PATH)

# ── 3️⃣ Load historical features
FEATURES_CSV = os.path.join(
    os.path.dirname(__file__),
    "data",
    "all_race_features_2010_2024.csv"
)
df = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])

st.title("🏎️ F1 Next-Race Finish Predictor")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Configure Prediction")

# Driver & track selectors
drivers = sorted(df["driverId"].unique())
driver  = st.sidebar.selectbox("Driver", drivers)

tracks  = sorted(df["circuitId"].unique())
track   = st.sidebar.selectbox("Circuit", tracks)

# Filter to this driver's history
df_drv = df[df["driverId"] == driver].sort_values("raceDate")
latest = df_drv.iloc[-1] if not df_drv.empty else None

# helper for defaults
def _default(col, low, high):
    if latest is not None and pd.notna(latest[col]):
        return float(latest[col])
    return float(df[col].mean())

# numeric ranges
ranges = {
    "avg_finish_5":      (df["avg_finish_5"].min(),      df["avg_finish_5"].max()),
    "podium_pct_5":      (0.0,                           1.0),
    "avg_grid_5":        (df["avg_grid_5"].min(),        df["avg_grid_5"].max()),
    "constructor_pts_5": (df["constructor_pts_5"].min(), df["constructor_pts_5"].max()),
    "last_year_pos":     (df["last_year_pos"].min(),     df["last_year_pos"].max()),
}

# sliders
avg_finish_5      = st.sidebar.slider("5-Race Avg Finish",          *ranges["avg_finish_5"],      value=_default("avg_finish_5", *ranges["avg_finish_5"]))
podium_pct_5      = st.sidebar.slider("5-Race Podium %",            0.0, 1.0,                     value=_default("podium_pct_5", 0.0, 1.0),         step=0.01)
avg_grid_5        = st.sidebar.slider("5-Race Avg Grid Slot",       *ranges["avg_grid_5"],        value=_default("avg_grid_5", *ranges["avg_grid_5"]))
constructor_pts_5 = st.sidebar.slider("5-Race Constructor Pts Avg", *ranges["constructor_pts_5"], value=_default("constructor_pts_5", *ranges["constructor_pts_5"]))
last_year_pos     = st.sidebar.slider("Last Year's Position at Track", *ranges["last_year_pos"],   value=_default("last_year_pos", *ranges["last_year_pos"]))

# ── Predict & display ─────────────────────────────────────────────────────
if st.sidebar.button("Predict Finish Distribution"):
    X = pd.DataFrame([{
        "avg_finish_5":      avg_finish_5,
        "podium_pct_5":      podium_pct_5,
        "avg_grid_5":        avg_grid_5,
        "constructor_pts_5": constructor_pts_5,
        "last_year_pos":     last_year_pos,
        "driverId":          driver,
        "circuitId":         track,
    }])
    with st.spinner("Calculating…"):
        probs = model.predict_proba(X)[0] * 100

    # Show the single most likely position
    most_likely = int(probs.argmax()) + 1
    st.metric("🏆 Most Likely Finish", f"{most_likely}")

    # Bar-chart of full distribution
    dist_df = pd.DataFrame(
        { "Probability (%)": probs },
        index=[i+1 for i in range(len(probs))]
    )
    st.bar_chart(dist_df)

# ── Recent form chart ──────────────────────────────────────────────────────
st.subheader(f"{driver} — Last 10 Races: Avg Finish")
if not df_drv.empty:
    chart = df_drv.set_index("raceDate")["avg_finish_5"].tail(10)
    st.line_chart(chart)
else:
    st.info("No historical data for this driver yet.")

st.markdown(
    """
    **Data spans 2010–2024.**  
    Adjust any slider to explore what‐if scenarios and see your favorite driver’s full finish‐position probabilities.
    """
)
