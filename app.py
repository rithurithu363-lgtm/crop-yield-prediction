import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")  # we'll generate .pkl next

st.title("🌾 Crop Yield Prediction App")

rain = st.number_input("Rainfall")
temp = st.number_input("Temperature")
fert = st.number_input("Fertilizer Usage")
ph = st.number_input("Soil pH", format="%.2f")
area = st.number_input("Area")

if st.button("Predict Yield"):
    prediction = model.predict([[rain, temp, fert, ph, area]])
    st.success(f"Predicted Yield: {prediction[0]:.2f}")
