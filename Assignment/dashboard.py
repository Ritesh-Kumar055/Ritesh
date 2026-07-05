try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

import pandas as pd
import os
import sys

if st is None:
    sys.stderr.write(
        "Error: Streamlit is not installed or could not be imported.\n"
        "Install it with: pip install streamlit\n"
        "Then run the app with: streamlit run dashboard.py\n"
    )
    sys.exit(1)

st.set_page_config(page_title="Digital Twin + CNN-LSTM Load Forecasting", layout="wide")

OUTPUT_DIR = "outputs"

st.title("⚡ Digital Twin-assisted CNN–LSTM Load Forecasting System")
st.write("Real-Time Dashboard — All model outputs, visuals, and forecasts.")

# ---------- Utility ----------
def show_image(name, caption):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_column_width=True)
    else:
        st.warning(f"{name} not found in outputs/")

def load_csv(name):
    path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

# ---------- Section 1: Data Overview ----------
st.header("📊 Data Snapshot")
df_snapshot = load_csv("data_snapshot.csv")
if df_snapshot is not None:
    st.write(df_snapshot.head())
else:
    st.warning("data_snapshot.csv missing — run backend.py")

show_image("timeseries_zones.png", "Power Consumption — Zone 1, 2, 3")

show_image("temp_vs_zone1.png", "Temperature vs Load (Zone 1)")

# ---------- Section 2: Digital Twin Simulation ----------
st.header("🏠 Digital Twin HVAC Model Output")
st.write("Comparison between original load and Digital Twin simulated load (first 24 hours).")
show_image("dt_vs_original_zone1.png", "Digital Twin vs Original Load")

dt_sim_sample = load_csv("dt_sim_snapshot.csv")
if dt_sim_sample is not None:
    st.write(dt_sim_sample.head())

# ---------- Section 3: Anomaly Detection ----------
st.header("🚨 Anomaly Detection (Isolation Forest)")
an = load_csv("anomalies.csv")
if an is not None:
    st.write(f"Detected anomalies: **{len(an)}**")
    st.write(an.head())

show_image("anomalies_zone1.png", "Detected Load Anomalies")

# ---------- Section 4: RandomForest SHAP Explanation ----------
st.header("🔍 Model Explainability (SHAP — RandomForest)")
show_image("shap_summary.png", "SHAP Summary — Feature Importance")

st.info("If shap_summary.png is missing, ensure you installed SHAP and ran backend.py")

# ---------- Section 5: CNN–LSTM Training Diagnostics ----------
st.header("📉 CNN–LSTM Training Loss")
show_image("training_loss.png", "Training vs Validation Loss")

# ---------- Section 6: Forecasting Results ----------
st.header("📈 Forecast vs Actual (First Horizon)")
show_image("forecast_vs_actual_h1_fixed.png", "Forecast vs Actual — H1")

preds = load_csv("forecast_fixed.csv")
if preds is not None:
    st.subheader("Prediction Table (sample)")
    st.write(preds.head(20))

    st.download_button(
        "⬇ Download Full Forecast CSV",
        data=preds.to_csv().encode('utf-8'),
        file_name="forecast_fixed.csv",
        mime="text/csv"
    )

# ---------- Footer ----------
st.markdown("---")
st.write("Developed: Digital Twin-assisted CNN+LSTM Short-Term Load Forecasting")
st.write("Made with ❤️ using Streamlit")
