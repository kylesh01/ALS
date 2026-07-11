import streamlit as st
import pandas as pd
import joblib

st.title("🧬 ALS Prediction Tool")

# Load saved model
try:
    model = joblib.load("als_model.pkl")
    st.success("Model loaded!")
except:
    st.error("Upload als_model.pkl")
    model = None

if model:
    age = st.number_input("Age", 0, 120, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    jitter = st.number_input("Jitter", value=0.0)
    shimmer = st.number_input("Shimmer", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame({
            'Age': [age],
            'Sex': [0 if sex == "Female" else 1],
            'J1_a': [jitter],
            'S1_a': [shimmer]
        })
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
        st.write("**Prediction:**", "ALS Present" if pred == 1 else "No ALS")
        st.write("**Probability:**", f"{proba:.1%}")
