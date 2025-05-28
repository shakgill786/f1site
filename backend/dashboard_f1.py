# backend/dashboard_f1.py

import os
import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

# ── 1️⃣ Page config
st.set_page_config(page_title="F1 Next-Race Finish Predictor", layout="wide")

# ── 2️⃣ Load your calibrated multiclass model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_f1_multiclass_calibrated.pkl")
model = joblib.load(MODEL_PATH)

# ── 3️⃣ Load historical features (last 5 years + current)
FEATURES_CSV = os.path.join(
    os.path.dirname(__file__), "data", "all_race_features_last5_and_current.csv"
)
df_all = pd.read_csv(FEATURES_CSV, parse_dates=["raceDate"])
df_all.sort_values("raceDate", inplace=True)

st.title("🏎️ F1 Next-Race Finish Predictor")

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Configure Prediction")

drivers = sorted(df_all["driverId"].unique())
driver  = st.sidebar.selectbox("Driver", drivers)

tracks  = sorted(df_all["circuitId"].unique())
track   = st.sidebar.selectbox("Circuit", tracks)

# driver-specific history for defaults
df_drv = df_all[df_all["driverId"] == driver].copy()
latest = df_drv.iloc[-1] if not df_drv.empty else None

def _default(col):
    if latest is not None and pd.notna(latest[col]):
        return float(latest[col])
    return float(df_all[col].mean())

ranges = {
    "avg_finish_5":      (df_all["avg_finish_5"].min(),      df_all["avg_finish_5"].max()),
    "podium_pct_5":      (0.0,                              1.0),
    "avg_grid_5":        (df_all["avg_grid_5"].min(),        df_all["avg_grid_5"].max()),
    "constructor_pts_5": (df_all["constructor_pts_5"].min(), df_all["constructor_pts_5"].max()),
    "last_year_pos":     (df_all["last_year_pos"].min(),     df_all["last_year_pos"].max()),
}

avg_finish_5      = st.sidebar.slider("5-Race Avg Finish", *ranges["avg_finish_5"], value=_default("avg_finish_5"))
podium_pct_5      = st.sidebar.slider("5-Race Podium %",    0.0, 1.0, value=_default("podium_pct_5"), step=0.01)
avg_grid_5        = st.sidebar.slider("5-Race Avg Grid Slot", *ranges["avg_grid_5"], value=_default("avg_grid_5"))
constructor_pts_5 = st.sidebar.slider("5-Race Constructor Pts Avg", *ranges["constructor_pts_5"], value=_default("constructor_pts_5"))
last_year_pos     = st.sidebar.slider("Last Year’s Pos at This Track", *ranges["last_year_pos"], value=_default("last_year_pos"))

# ── Predict & display distribution ─────────────────────────────────────────
if st.sidebar.button("Predict Finish Distribution"):
    X_pred = pd.DataFrame([{
        "avg_finish_5":      avg_finish_5,
        "podium_pct_5":      podium_pct_5,
        "avg_grid_5":        avg_grid_5,
        "constructor_pts_5": constructor_pts_5,
        "last_year_pos":     last_year_pos,
        "driverId":          driver,
        "circuitId":         track,
    }])
    with st.spinner("Calculating…"):
        probs = model.predict_proba(X_pred)[0] * 100

    best = int(probs.argmax()) + 1
    st.metric("🏆 Most Likely Finish", f"{best}")

    dist_df = pd.DataFrame(
        {"Probability (%)": probs},
        index=[str(i) for i in range(1, len(probs)+1)]
    )
    st.bar_chart(dist_df)

# ── Recent Form Sparkline ──────────────────────────────────────────────────
st.subheader(f"{driver} — Last 10 Races: Avg Finish")
if not df_drv.empty:
    spark = df_drv.set_index("raceDate")["avg_finish_5"].tail(10)
    st.line_chart(spark)
else:
    st.info("No historical data for this driver yet.")

# ── Diagnostics: Calibration & Threshold ───────────────────────────────────
with st.expander("🛠️ Model Calibration & Threshold Diagnostics"):
    st.write("Verify **win** calibration (finish=1) and choose a cutoff")

    # prepare hold-out
    df_all = df_all.reset_index(drop=True)
    y_bin = (df_all["position"] == 1).astype(int)
    FEATURES = [
        "avg_finish_5","podium_pct_5","avg_grid_5",
        "constructor_pts_5","last_year_pos",
        "driverId","circuitId"
    ]
    split = int(len(df_all)*0.8)
    X_test, y_test = df_all[FEATURES].iloc[split:], y_bin.iloc[split:]
    proba_win = model.predict_proba(X_test)[:, 0]  # class=0 ⇒ win

    # 1) Calibration curve (5 bins)
    frac_pos, mean_pred = calibration_curve(y_test, proba_win, n_bins=5)
    fig1, ax1 = plt.subplots()
    ax1.plot(mean_pred, frac_pos, "s-", label="Observed")
    ax1.plot([0,1],[0,1], "--", color="gray", label="Ideal")
    ax1.set_xlabel("Mean pred. probability")
    ax1.set_ylabel("Fraction of wins")
    ax1.set_title("Calibration Curve")
    ax1.legend()
    st.pyplot(fig1)

    # 2) Precision–Recall vs Threshold
    prec, rec, th = precision_recall_curve(y_test, proba_win)
    fig2, ax2 = plt.subplots()
    ax2.plot(th, prec[:-1], label="Precision")
    ax2.plot(th, rec[:-1],  label="Recall")
    ax2.set_xlabel("Probability threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision & Recall vs Threshold")

    cutoff = st.slider("Select win-prob cutoff", 0.0, 1.0, 0.5, 0.01)
    ax2.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff={cutoff:.2f}")
    ax2.legend()
    st.pyplot(fig2)

    # metrics at cutoff
    y_pred_thr = (proba_win >= cutoff).astype(int)
    p = precision_score(y_test, y_pred_thr)
    r = recall_score   (y_test, y_pred_thr)
    f1 = f1_score      (y_test, y_pred_thr)
    st.markdown(
        f"**At cutoff = {cutoff:.2f}:**  \n"
        f"- Precision = {p:.2f}  \n"
        f"- Recall    = {r:.2f}  \n"
        f"- F₁ Score  = {f1:.2f}"
    )

# ── Backtest Past Races (Manual Input) ──────────────────────────────────────
with st.expander("🔄 Backtest Past Races (Manual Input)"):
    st.write("Enter `actual_position` for each current-season race")

    this_year = datetime.date.today().year
    df_season = df_all[df_all["season"] == this_year].copy()
    if df_season.empty:
        st.info("No records for the current season yet.")
    else:
        editor = df_season[[
            "raceDate","driverId","circuitId",
            "avg_finish_5","podium_pct_5",
            "avg_grid_5","constructor_pts_5","last_year_pos"
        ]].reset_index(drop=True)
        editor["actual_position"] = pd.NA

        try:
            edited = st.data_editor(editor, num_rows="dynamic", use_container_width=True)
        except AttributeError:
            edited = st.experimental_data_editor(editor, num_rows="dynamic", use_container_width=True)

        if st.button("Show Backtest Predictions"):
            bt = edited.copy()
            bt["actual_position"] = pd.to_numeric(bt["actual_position"], errors="coerce")
            bt = bt.dropna(subset=["actual_position"])
            if bt.empty:
                st.warning("No valid entries.")
            else:
                bt["actual_position"] = bt["actual_position"].astype(int)
                X_bt = bt[[
                    "avg_finish_5","podium_pct_5","avg_grid_5",
                    "constructor_pts_5","last_year_pos","driverId","circuitId"
                ]]
                probs_bt = model.predict_proba(X_bt)
                bt["predicted_position"] = np.argmax(probs_bt, axis=1) + 1
                bt["predicted_prob"]     = np.round(probs_bt.max(axis=1)*100, 1)

                acc = (bt["predicted_position"] == bt["actual_position"]).mean()
                st.write(f"**Backtest Accuracy:** {acc:.2%} on {len(bt)} races")

                st.dataframe(
                    bt[[
                        "raceDate","driverId","circuitId",
                        "actual_position","predicted_position","predicted_prob"
                    ]],
                    use_container_width=True
                )

# ── Footer ─────────────────────────────────────────────────────────────────
years = df_all["raceDate"].dt.year.unique()
st.markdown(f"**Data spans {years.min()}–{years.max()}**  \nAdjust sliders/threshold to explore what-ifs.")
