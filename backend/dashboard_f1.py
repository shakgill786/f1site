# dashboard_f1.py
import os, streamlit as st, pandas as pd, joblib

st.set_page_config("F1 Win Predictor",layout="wide")
MODEL    = joblib.load(os.path.join(os.path.dirname(__file__),"model_f1.pkl"))
FEAT_CSV = os.path.join(os.path.dirname(__file__),"data","all_races_features.csv")
df       = pd.read_csv(FEAT_CSV, parse_dates=['raceDate'])

drivers = sorted(df['driverId'].unique())
tracks  = sorted(df['track'].unique())

st.title("🏎️ F1 Next-Race Win Predictor")
drv = st.sidebar.selectbox("Driver", drivers)
trk = st.sidebar.selectbox("Track",   tracks)

# pick the driver’s last row
last = df[df.driverId==drv].sort_values('raceDate').iloc[-1]
st.sidebar.header("Driver’s recent 5-race stats")
avg_finish = st.sidebar.slider("Avg finish",  1.0,20.0, float(last['avg_finish_5']))
pod_pct     = st.sidebar.slider("Podium %",   0.0,1.0,    float(last['podium_pct_5']),step=0.01)
avg_grid    = st.sidebar.slider("Quali grid", 1.0,20.0, float(last['avg_grid_5']))
cnstr_pts   = st.sidebar.number_input("Constructor pts (5r)", 0,100, int(last['constructor_pts_5']))
last_yr     = st.sidebar.number_input("Last yr finish", 1,20, int(last.get('last_year_pos',10)))

if st.sidebar.button("Predict Winner"):
    X = pd.DataFrame([{
      'driverId':drv,'track':trk,
      'avg_finish_5':avg_finish,
      'podium_pct_5':pod_pct,
      'avg_grid_5':avg_grid,
      'constructor_pts_5':cnstr_pts,
      'last_year_pos':last_yr
    }])
    p = MODEL.predict_proba(X)[0,1]*100
    st.success(f"🏁 Win Probability: {p:.1f}%")

# show mini–form chart
recent = df[df.driverId==drv].set_index('raceDate')[['win']].tail(10)
st.line_chart(recent.rolling(5).mean(),height=200)
