import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="ALS Prediction Tool", layout="centered")

st.title("🧬 ALS Prediction Tool (Voice Analysis)")

with st.sidebar:
    st.header("Dataset (Optional)")
    uploaded_csv = st.file_uploader("Upload CSV if needed", type="csv")
    if uploaded_csv:
        with open("dataset.csv", "wb") as f:
            f.write(uploaded_csv.getbuffer())
        st.success("Dataset uploaded")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("als_model.pkl")
        st.success("✅ Loaded trained model")
        return model
    except:
        st.error("❌ Model file 'als_model.pkl' not found. Upload it to GitHub.")
        return None

model = load_model()

if model:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
    with col2:
        jitter = st.number_input("Jitter (J1_a)", value=0.0, format="%.4f")
        shimmer = st.number_input("Shimmer (S1_a)", value=0.0, format="%.4f")
    
    if st.button("Predict", type="primary"):
        # Match exact feature names from training
        input_data = {
            'Sex': 0 if sex == "Female" else 1,
            'Age': float(age),
            'J1_a': jitter,
            'S1_a': shimmer
        }
        
        input_df = pd.DataFrame([input_data])
        
        try:
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0, 1]
            
            if pred == 1:
                st.error(f"**ALS Likely Present** (Probability: {proba:.1%})")
            else:
                st.success(f"**ALS Likely Not Present** (Probability: {proba:.1%})")
        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Check that model was trained with columns: Sex, Age, J1_a, S1_a")
else:
    st.info("Upload als_model.pkl to your GitHub repo and redeploy.")
