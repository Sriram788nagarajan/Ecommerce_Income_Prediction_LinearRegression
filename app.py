import streamlit as st
import joblib
import numpy as np

# Load your trained model (.pkl)
model = joblib.load("ecommerce_price_model.pkl")

st.title("E-commerce Price Predictor")

f1 = st.number_input("Avg. Session Length", value=0.0, step=0.1)
f2 = st.number_input("Time on App", value=0.0, step=0.1)
f3 = st.number_input("Time on Website", value=0.0, step=0.1)
f4 = st.number_input("Length of Membership", value=0.0, step=0.1)

if st.button("Predict"):
    X = np.array([[f1, f2, f3, f4]])
    y = model.predict(X)[0]
    st.success(f"Prediction: {y:.2f}")
