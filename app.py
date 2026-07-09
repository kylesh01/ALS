import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

st.set_page_config(page_title="ALS Prediction Tool", layout="centered")

st.title("🧬 ALS Prediction Tool (Voice Analysis)")

# Dataset upload
with st.sidebar:
    st.header("Dataset")
    uploaded_csv = st.file_uploader("Upload CSV dataset", type="csv")
    if uploaded_csv:
        with open("dataset.csv", "wb") as f:
            f.write(uploaded_csv.getbuffer())
        st.success("✅ Dataset ready!")

# Load model (same as before)
def load_model():
    if not os.path.exists("dataset.csv"):
        return None, None
    df = pd.read_csv("dataset.csv")
    df.dropna(axis=1, how='all', inplace=True)
    if 'ID' in df.columns: df.drop('ID', axis=1, inplace=True)
    if 'Sex' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
    df = df.dropna()
    
    target = 'Diagnosis (ALS)'
    X = df.drop(columns=[target] + [c for c in df.columns if 'Diagnosis (ALS).1' in str(c)])
    y = df[target]
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, features = load_model()

# Main interface
if model:
    st.success("Model loaded successfully!")
    
    # Manual inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
    with col2:
        jitter = st.number_input("Jitter", value=0.0, format="%.4f")
        shimmer = st.number_input("Shimmer", value=0.0, format="%.4f")
    
    # New Audio Upload Section
    st.subheader("🎤 Upload Voice Recording (Sustained 'A' vowel)")
    audio_file = st.file_uploader("Upload MP3 or WAV file", type=["mp3", "wav"])
    
    if audio_file:
        # Save temporarily
        with open("temp_audio", "wb") as f:
            f.write(audio_file.getbuffer())
        
        try:
            # Install parselmouth if needed (Streamlit handles it via requirements)
            import parselmouth
            from parselmouth.praat import call
            
            sound = parselmouth.Sound("temp_audio")
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call(sound, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            jitter = jitter_local
            shimmer = shimmer_local
            
            st.success(f"✅ Extracted → Jitter: {jitter:.4f} | Shimmer: {shimmer:.4f}")
            
        except Exception as e:
            st.error(f"Audio processing error: {e}. Using manual values instead.")
    
    # Predict button
    if st.button("Predict", type="primary"):
        input_data = {col: 0.0 for col in features}
        input_data['Age'] = float(age)
        input_data['Sex'] = 0 if sex == "Female" else 1
        
        for col in features:
            if any(x in col.lower() for x in ['jitter', 'j1','j3','j5']):
                input_data[col] = jitter
            if any(x in col.lower() for x in ['shimmer', 's1','s3','s5']):
                input_data[col] = shimmer
        
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
        
        if pred == 1:
            st.error(f"**ALS Likely Present** (Probability: {proba:.1%})")
        else:
            st.success(f"**ALS Likely Not Present** (Probability: {proba:.1%})")
else:
    st.info("Upload your dataset first.")
