import os
import pandas as pd
import joblib
import streamlit as st

# For diagnostics plots
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

# ── 1️⃣ Page config
st.set_page_config(
    page_title="F1 Next-Race Finish Predictor",
    layout="wide",
)

# ── 2️⃣ Load your calibrated multiclass model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model_f1_multiclass_calibrated.pkl"
)
model = joblib.load(MODEL_PATH)

# ── 3️⃣ Load historical features (last 5 years + current)
FEATURES_CSV = os.path.join(
    os.path.dirname(__file__),
    "data",
    "all_race_features_last5_and_current.csv"
)
df = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])

st.title("🏎️ F1 Next-Race Finish Predictor")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Configure Prediction")

drivers = sorted(df["driverId"].unique())
driver  = st.sidebar.selectbox("Driver", drivers)

tracks = sorted(df["circuitId"].unique())
track  = st.sidebar.selectbox("Circuit", tracks)

# driver‐specific history for defaults
df_drv = df[df["driverId"] == driver].sort_values("raceDate")
latest = df_drv.iloc[-1] if not df_drv.empty else None

def _default(col):
    if latest is not None and pd.notna(latest[col]):
        return float(latest[col])
    return float(df[col].mean())

ranges = {
    "avg_finish_5":      (df["avg_finish_5"].min(),      df["avg_finish_5"].max()),
    "podium_pct_5":      (0.0,                           1.0),
    "avg_grid_5":        (df["avg_grid_5"].min(),        df["avg_grid_5"].max()),
    "constructor_pts_5": (df["constructor_pts_5"].min(), df["constructor_pts_5"].max()),
    "last_year_pos":     (df["last_year_pos"].min(),     df["last_year_pos"].max()),
}

avg_finish_5      = st.sidebar.slider(
    "5-Race Avg Finish",
    *ranges["avg_finish_5"],
    value=_default("avg_finish_5"),
)
podium_pct_5      = st.sidebar.slider(
    "5-Race Podium %",
    0.0, 1.0,
    value=_default("podium_pct_5"),
    step=0.01,
)
avg_grid_5        = st.sidebar.slider(
    "5-Race Avg Grid Slot",
    *ranges["avg_grid_5"],
    value=_default("avg_grid_5"),
)
constructor_pts_5 = st.sidebar.slider(
    "5-Race Constructor Pts Avg",
    *ranges["constructor_pts_5"],
    value=_default("constructor_pts_5"),
)
last_year_pos     = st.sidebar.slider(
    "Last Year’s Pos at This Track",
    *ranges["last_year_pos"],
    value=_default("last_year_pos"),
)

# ── Prediction ─────────────────────────────────────────────────────────────
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

    best = int(probs.argmax()) + 1
    st.metric("🏆 Most Likely Finish", f"{best}")

    dist_df = pd.DataFrame(
        {"Probability (%)": probs},
        index=[str(i) for i in range(1, len(probs) + 1)]
    )
    st.bar_chart(dist_df)

# ── Recent Form ─────────────────────────────────────────────────────────────
st.subheader(f"{driver} — Last 10 Races: Avg Finish")
if not df_drv.empty:
    chart = df_drv.set_index("raceDate")["avg_finish_5"].tail(10)
    st.line_chart(chart)
else:
    st.info("No historical data for this driver yet.")

# ── Diagnostics ────────────────────────────────────────────────────────────
with st.expander("🛠️ Model Calibration & Threshold Diagnostics", expanded=False):
    st.write("Verify calibration for **win** (finish=1) and pick a threshold:")

    # prepare binary “win” vs “not win”
    df = df.sort_values("raceDate").reset_index(drop=True)
    y = (df["position"] == 1).astype(int)
    X_bin = df[[
        "avg_finish_5","podium_pct_5","avg_grid_5",
        "constructor_pts_5","last_year_pos",
        "driverId","circuitId"
    ]]
    split = int(len(df) * 0.8)
    X_test, y_test = X_bin.iloc[split:], y.iloc[split:]
    # probability of finishing first (class 0 of multiclass)
    proba_win = model.predict_proba(X_test)[:, 0]

    # 1) Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, proba_win, n_bins=10)
    fig1, ax1 = plt.subplots()
    ax1.plot(mean_pred, frac_pos, "s-", label="Observed")
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of wins")
    ax1.set_title("Calibration Curve (Win vs Not Win)")
    ax1.legend()
    st.pyplot(fig1)

    # 2) Precision–Recall vs Threshold
    prec, rec, thresh = precision_recall_curve(y_test, proba_win)
    fig2, ax2 = plt.subplots()
    ax2.plot(thresh, prec[:-1], label="Precision")
    ax2.plot(thresh, rec[:-1],  label="Recall")
    ax2.set_xlabel("Probability threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision & Recall vs Threshold")
    ax2.legend()
    st.pyplot(fig2)

# ── Footer ─────────────────────────────────────────────────────────────────
years = df["raceDate"].dt.year.unique()
min_year, max_year = years.min(), years.max()
st.markdown(
    f"**Data spans {min_year}–{max_year}.**  \n"
    "Adjust sliders or threshold above to explore what-ifs."
)
