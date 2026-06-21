"""
app.py
======
Streamlit front-end for the Smart Energy Consumption Prediction project.

Run with:  streamlit run app.py

WHAT THIS FILE DOES
--------------------
Wraps the trained model (loaded via src/predict.py's EnergyPredictor) in
a simple, recruiter-friendly web UI: pick a time of day and a handful of
environmental conditions, hit "Predict," and see the estimated appliance
energy draw along with a confidence range and a small context chart.
"""

import sys
from datetime import datetime, time as dtime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from predict import DEFAULT_SENSOR_VALUES, EnergyPredictor  # noqa: E402

st.set_page_config(
    page_title="Smart Energy Consumption Predictor",
    page_icon="⚡",
    layout="centered",
)

# ---------------------------------------------------------------------
# Styling — a few small CSS tweaks so this doesn't look like a default
# Streamlit demo. Kept intentionally minimal.
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stMetric { background-color: #f8fafc; border-radius: 10px; padding: 1rem; }
        .block-container { padding-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> EnergyPredictor:
    return EnergyPredictor()


st.title("⚡ Smart Energy Consumption Predictor")
st.caption(
    "Predicts household appliance energy use (Wh per 10-minute interval) from "
    "indoor/outdoor conditions and time of day, trained on the UCI Appliances "
    "Energy Prediction dataset."
)

try:
    predictor = load_predictor()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python src/train.py` from the project root first to generate models/model.pkl.")
    st.stop()

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Time")
    selected_date = st.date_input("Date", value=datetime.now().date())
    selected_time = st.time_input("Time", value=dtime(hour=18, minute=30))

with col2:
    st.subheader("Weather")
    indoor_temp = st.slider("Indoor temperature (°C)", 14.0, 30.0, 21.0, 0.1)
    indoor_humidity = st.slider("Indoor relative humidity (%)", 20.0, 65.0, 40.0, 0.5)
    outdoor_temp = st.slider("Outdoor temperature (°C)", -5.0, 30.0, 12.0, 0.1)
    outdoor_humidity = st.slider("Outdoor relative humidity (%)", 20.0, 100.0, 75.0, 1.0)

with st.expander("Advanced sensor inputs (optional)"):
    lights = st.slider("Light energy use (Wh)", 0.0, 70.0, 0.0, 1.0)
    windspeed = st.slider("Wind speed", 0.0, 15.0, 4.0, 0.5)
    visibility = st.slider("Visibility", 1.0, 70.0, 38.0, 1.0)
    press = st.slider("Pressure (mm Hg)", 720.0, 775.0, 755.0, 0.5)

st.divider()

if st.button("🔮 Predict Energy Consumption", type="primary", use_container_width=True):
    timestamp = datetime.combine(selected_date, selected_time)

    sensor_overrides = {
        "T1": indoor_temp, "T2": indoor_temp, "T3": indoor_temp,
        "RH_1": indoor_humidity, "RH_2": indoor_humidity, "RH_3": indoor_humidity,
        "T_out": outdoor_temp, "RH_out": outdoor_humidity,
        "lights": lights, "Windspeed": windspeed, "Visibility": visibility,
        "Press_mm_hg": press,
    }

    result = predictor.predict_with_confidence(timestamp, **sensor_overrides)

    st.subheader("Prediction")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Usage", f"{result['prediction_wh']} Wh")
    m2.metric("Lower bound (95%)", f"{result['lower_bound_wh']} Wh")
    m3.metric("Upper bound (95%)", f"{result['upper_bound_wh']} Wh")

    st.caption(f"Model: **{result['model_used']}** · {result['confidence_note']}")

    # Small context chart: where does this prediction sit relative to a
    # typical day's usage curve? Static reference curve based on the
    # EDA-derived hourly pattern, just to give visual context.
    hours = list(range(24))
    reference_curve = [
        53, 51, 49, 48, 49, 52, 58, 79, 106, 113, 125, 133,
        124, 125, 108, 106, 120, 162, 190, 144, 127, 97, 69, 57,
    ]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(hours, reference_curve, linestyle=":", color="#94a3b8", label="Typical daily pattern")
    ax.scatter([timestamp.hour], [result["prediction_wh"]], color="#2563eb", s=120,
               zorder=5, label="Your prediction")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Wh")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Your prediction vs. typical hourly pattern")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
with st.expander("ℹ️ About this model"):
    st.markdown(
        """
        Trained on the **UCI Appliances Energy Prediction** dataset (Candanedo et al., 2017):
        19,735 readings taken every 10 minutes over 4.5 months in a low-energy house in Belgium.

        Four models were compared (Linear Regression, Random Forest, Gradient Boosting, XGBoost)
        on an 80/20 random split. The best model by RMSE was selected automatically — see
        `outputs/reports/model_comparison.csv` for the full comparison table.

        **Note on interpretation:** this dataset is genuinely noisy (R² ≈ 0.55-0.58 even for the
        best model) — appliance use depends heavily on human behavior that isn't captured by
        temperature/humidity sensors alone, so treat predictions as a directional estimate, not
        an exact reading.
        """
    )
