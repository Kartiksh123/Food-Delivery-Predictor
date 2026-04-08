import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🍕 Delivery Predictor", layout="centered")

# ---------------- CUSTOM CSS (ANIMATIONS + UI) ----------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

/* Title Animation */
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
    animation: fadeIn 2s ease-in-out;
}

/* Fade animation */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

/* Button Styling */
.stButton>button {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
}

/* Result Card */
.result {
    text-align: center;
    font-size: 24px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<div class='title'>🍕 Food Delivery Time Predictor</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:white;'>AI powered delivery estimation</p>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open('best_linear_regression_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# ---------------- INPUT UI ----------------
st.markdown("<div class='glass'>", unsafe_allow_html=True)

st.subheader("📦 Delivery Details")

col1, col2 = st.columns(2)

with col1:
    order_id = st.number_input("Order ID", 0, 10000, 500)
    distance = st.slider("Distance (km)", 0.5, 20.0, 10.0)
    prep_time = st.slider("Prep Time (min)", 5, 30, 15)

with col2:
    courier_exp = st.slider("Courier Experience", 0.0, 10.0, 5.0)
    weather = st.selectbox("Weather", ['Clear', 'Foggy', 'Rainy', 'Sunny', 'Windy'])
    traffic = st.selectbox("Traffic", ['High', 'Low', 'Medium'])

time_of_day = st.selectbox("Time of Day", ['Afternoon', 'Evening', 'Morning', 'Night'])
vehicle = st.selectbox("Vehicle", ['Bike', 'Car', 'Scooter'])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Delivery Time"):

    with st.spinner("Analyzing data..."):
        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        weather_map = {'Clear':0,'Foggy':1,'Rainy':2,'Sunny':3,'Windy':4}
        traffic_map = {'High':0,'Low':1,'Medium':2}
        time_map = {'Afternoon':0,'Evening':1,'Morning':2,'Night':3}
        vehicle_map = {'Bike':0,'Car':1,'Scooter':2}

        input_data = {
            'Order_ID': order_id,
            'Distance_km': distance,
            'Preparation_Time_min': prep_time,
            'Courier_Experience_yrs': courier_exp
        }

        for i in range(5):
            input_data[f'Weather_{i}'] = 1 if i == weather_map[weather] else 0

        for i in range(3):
            input_data[f'Traffic_Level_{i}'] = 1 if i == traffic_map[traffic] else 0

        for i in range(4):
            input_data[f'Time_of_Day_{i}'] = 1 if i == time_map[time_of_day] else 0

        for i in range(3):
            input_data[f'Vehicle_Type_{i}'] = 1 if i == vehicle_map[vehicle] else 0

        feature_columns = list(input_data.keys())
        input_df = pd.DataFrame([input_data], columns=feature_columns)

        prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class='glass result'>
        ⏱️ Estimated Time: <b>{prediction:.2f} minutes</b>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

# ---------------- FOOTER ----------------
st.markdown("""
<p style='text-align:center;color:white;margin-top:30px;'>
Made with ❤️ by Kartik Sharma
</p>
""", unsafe_allow_html=True)