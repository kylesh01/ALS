import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("🧬 ALS Prediction Tool")

uploaded_csv = st.file_uploader("Upload small CSV", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    df = df.dropna()
    
    # Simple encoding
    df['Sex'] = df['Sex'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
    
    X = df[['Age', 'Sex', 'J1_a', 'S1_a']]
    y = df['Diagnosis (ALS)']

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    st.success("Model ready!")

    age = st.number_input("Age", value=50)
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
