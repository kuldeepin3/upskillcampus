import os
from datetime import date

import gdown
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="centered"
)

# -----------------------------
# Download & Load Model (Cached & Memory-Mapped)
# -----------------------------
MODEL_PATH = "traffic_prediction_model.pkl"

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        FILE_ID = "16gHaTNPQfkmmU5-XvGR8_UmgGNl5uodd"
        URL = f"https://drive.google.com/uc?id={FILE_ID}"
        # Download the model
        gdown.download(URL, path, quiet=False)
    
    # Load using memory mapping to prevent Out-Of-Memory (OOM) on Streamlit Cloud
    try:
        loaded_model = joblib.load(path, mmap_mode="r")
    except Exception:
        loaded_model = joblib.load(path)
    return loaded_model

with st.spinner("Downloading and loading trained model... Please wait."):
    model = load_model(MODEL_PATH)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.big-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1f77b4;
}
.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}
.result-box{
    background-color:#d4edda;
    padding:20px;
    border-radius:12px;
    text-align:center;
    border-radius:12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<p class="big-title">🚦 Smart City Traffic Forecasting</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Predict hourly traffic using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
selected_date = st.date_input(
    "📅 Select Date",
    value=date.today()
)

selected_hour = st.selectbox(
    "🕒 Select Time",
    options=range(24),
    format_func=lambda x: f"{(x % 12) or 12}:00 {'AM' if x < 12 else 'PM'}"
)

junction = st.selectbox(
    "🚦 Select Junction",
    [1, 2, 3, 4]
)

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(date_value, hour, junction):

    df = pd.DataFrame({
        "DateTime": [pd.Timestamp(date_value)],
        "Junction": [junction]
    })

    df["Year"] = df["DateTime"].dt.year
    df["Month"] = df["DateTime"].dt.month
    df["Day"] = df["DateTime"].dt.day
    df["Hour"] = hour

    df["DayOfWeek"] = df["DateTime"].dt.day_name()

    df = pd.get_dummies(
        df,
        columns=["DayOfWeek"],
        drop_first=True
    )

    df.drop(columns=["DateTime"], inplace=True)

    required_columns = [
        "Junction",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek_Monday",
        "DayOfWeek_Saturday",
        "DayOfWeek_Sunday",
        "DayOfWeek_Thursday",
        "DayOfWeek_Tuesday",
        "DayOfWeek_Wednesday"
    ]

    df = df.reindex(
        columns=required_columns,
        fill_value=0
    )

    return df

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚗 Predict Traffic"):

    input_data = preprocess(
        selected_date,
        selected_hour,
        junction
    )

    prediction = model.predict(input_data)[0]

    if prediction < 40:
        status = "🟢 Low Traffic"
    elif prediction < 100:
        status = "🟡 Moderate Traffic"
    else:
        status = "🔴 Heavy Traffic"

    st.markdown(
        f"""
        <div class="result-box">
            <h2>Estimated Vehicles</h2>
            <h1>{int(prediction)}</h1>
            <h3>{status}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Model", "Random Forest")

    with col2:
        st.metric("R² Score", "96.6%")

st.divider()

st.caption("Developed by Kuldeep Prajapati")