import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="ALS Prediction Tool", layout="centered")

st.title("🧬 ALS Prediction Tool (Voice Analysis)")

with st.sidebar:
    st.header("Upload Dataset")
    uploaded_csv = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        
        # Clean data
        df = df.dropna()
        if 'Sex' in df.columns and df['Sex'].dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df['Sex'] = le.fit_transform(df['Sex'])
        
        st.success("Dataset loaded and cleaned!")

        features = ['Age', 'Sex', 'J1_a', 'S1_a']
        X = df[features]
        y = df['Diagnosis (ALS)']

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        st.success("Model trained successfully!")

        # Inputs
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 0, 120, 50)
            sex = st.selectbox("Sex", ["Female", "Male"])
        with col2:
            jitter = st.number_input("Jitter (J1_a)", value=0.0, format="%.4f")
            shimmer = st.number_input("Shimmer (S1_a)", value=0.0, format="%.4f")

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame({
                'Age': [float(age)],
                'Sex': [0 if sex == "Female" else 1],
                'J1_a': [jitter],
                'S1_a': [shimmer]
            })
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0, 1]
            
            if pred == 1:
                st.error(f"**ALS Likely Present** (Probability: {proba:.1%})")
            else:
                st.success(f"**ALS Likely Not Present** (Probability: {proba:.1%})")
