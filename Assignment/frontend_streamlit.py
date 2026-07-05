"""
frontend_streamlit.py

Run:
  pip install streamlit
  streamlit run frontend_streamlit.py

This Streamlit app loads the outputs of backend.py (or the CSV) and provides
interactive visualizations, Digital Twin scenario sliders, and SHAP images.
"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import streamlit as st  # type: ignore
else:
    import sys
    import subprocess
    import importlib
    try:
        import streamlit as st
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        st = importlib.import_module("streamlit")
import pandas as pd
import matplotlib.pyplot as plt
import sys
import subprocess
import importlib
try:
    joblib = importlib.import_module("joblib")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
    joblib = importlib.import_module("joblib")
import os



OUT_DIR = 'outputs'
DATA_PATH = "powerconsumption.csv"

st.set_page_config(layout='wide', page_title='Digital Twin + CNN-LSTM Forecasting')

st.title('Digital Twin Assisted Forecasting — Dashboard')

# Upload or use default
uploaded = st.file_uploader('Upload power consumption CSV (optional)', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded, low_memory=False)
else:
    df = pd.read_csv(DATA_PATH, low_memory=False)

# quick parsing
# try to detect datetime column
possible_dt = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
if possible_dt:
    dtcol = possible_dt[0]
else:
    dtcol = df.columns[0]

df[dtcol] = pd.to_datetime(df[dtcol], errors='coerce')
df = df.sort_values(dtcol).reset_index(drop=True)

st.sidebar.header('Controls')
show_n = st.sidebar.slider('Show first N rows', 100, 5000, 1000)
st.sidebar.markdown('Digital Twin temperature delta (simulate hotter/colder scenario)')
temp_delta = st.sidebar.slider('Temp delta (°C)', -5.0, 10.0, 0.0)

st.subheader('Raw data preview')
st.write(df.head(show_n))

# Basic plots
st.subheader('Time Series — Zones')
fig, ax = plt.subplots(figsize=(12,4))
# try to find power columns by heuristics
power_cols = [c for c in df.columns if 'power' in c.lower() or 'zone' in c.lower()]
if len(power_cols) == 0:
    st.warning('No power columns found in uploaded CSV — check column names')
else:
    for c in power_cols[:3]:
        ax.plot(df[dtcol], pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce'), label=c)
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Power')
    ax.legend()
    st.pyplot(fig)

# Temperature vs Zone1 scatter
temp_cols = [c for c in df.columns if 'temp' in c.lower()]
if temp_cols and power_cols:
    fig2, ax2 = plt.subplots()
    ax2.scatter(pd.to_numeric(df[temp_cols[0]].astype(float), errors='coerce'), pd.to_numeric(df[power_cols[0]].astype(str).str.replace(',',''), errors='coerce'), s=2)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel(power_cols[0])
    ax2.set_title('Temperature vs ' + power_cols[0])
    st.pyplot(fig2)

# Show DT-simulated image if exists
st.subheader('Digital Twin Simulation')
img_path = os.path.join(OUT_DIR, 'dt_vs_original_zone1.png')
if os.path.exists(img_path):
    st.image(img_path, caption='Digital Twin vs Original (sample)')
else:
    st.info('Run backend.py to produce Digital Twin simulation images')

# SHAP
st.subheader('SHAP Explanation (RandomForest)')
shap_img = os.path.join(OUT_DIR, 'shap_summary.png')
if os.path.exists(shap_img):
    st.image(shap_img, caption='SHAP Summary (feature importance)')
else:
    st.info('Run backend.py to generate SHAP plots (requires shap package)')

# Forecast
st.subheader('Forecasts (sample)')
forecast_csv = os.path.join(OUT_DIR, 'forecast.csv')
if os.path.exists(forecast_csv):
    fdf = pd.read_csv(forecast_csv, index_col=0, parse_dates=True)
    st.write('Forecast sample (first 10 rows)')
    st.write(fdf.head(10))
    if st.button('Plot Forecast vs Actual (first horizon)'):
        fig3, ax3 = plt.subplots(figsize=(12,4))
        ax3.plot(fdf.index[:200], fdf['act_h1'][:200], label='Actual')
        ax3.plot(fdf.index[:200], fdf['pred_h1'][:200], label='Predicted')
        ax3.legend()
        ax3.set_title('Forecast vs Actual (first horizon)')
        st.pyplot(fig3)
else:
    st.info('Run backend.py to generate forecasts (outputs/forecast.csv)')

st.sidebar.markdown('---')
st.sidebar.write('Backend outputs are read from the `outputs/` folder. Run `python backend.py` to generate them.')

st.write('---')
st.markdown('Developed: Digital Twin assisted CNN+LSTM forecasting — example dashboard')
