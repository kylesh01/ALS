import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="ALS Voice Prediction", layout="wide")

st.title("🧬 ALS Prediction Tool (Voice Analysis)")
st.write("Enter details or use voice features extracted from audio.")

# Upload dataset in sidebar
with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload your CSV (polygence cvs dataset...)", type="csv")
    if uploaded_file:
        with open("dataset.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Dataset ready!")

# Load & train model
@st.cache_resource
def get_model():
    if not os.path.exists("dataset.csv"):
        st.error("Please upload the dataset first.")
        return None, None
    df = pd.read_csv("dataset.csv")
    # Your preprocessing
    df.dropna(axis=1, how='all', inplace=True)
    if 'ID' in df.columns: df.drop('ID', axis=1, inplace=True)
    if 'Sex' in df.columns:
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
    df = df.dropna()
    
    target = 'Diagnosis (ALS)'
    if target not in df.columns:
        st.error("Target column not found!")
        return None, None
    X = df.drop(columns=[target] + [c for c in df.columns if 'Diagnosis (ALS).1' in c])
    y = df[target]
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist()

model, features = get_model()

# Main form
if model:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 0, 120, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
    with col2:
        jitter = st.number_input("Jitter (J1_a or similar)", value=0.0, format="%.4f")
        shimmer = st.number_input("Shimmer (S1_a or similar)", value=0.0, format="%.4f")
    
    if st.button("🚀 Predict", type="primary"):
        input_data = {col: 0.0 for col in features}
        input_data['Age'] = float(age)
        input_data['Sex'] = 0 if sex == "Female" else 1
        # Try to map jitter/shimmer (adjust column names if needed)
        for col in features:
            if 'jitter' in col.lower() or 'j1' in col.lower():
                input_data[col] = jitter
            if 'shimmer' in col.lower() or 's1' in col.lower():
                input_data[col] = shimmer
        
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0, 1]
        
        if pred == 1:
            st.error(f"**ALS Likely Present** (Probability: {proba:.1%})")
        else:
            st.success(f"**ALS Likely Not Present** (Probability: {proba:.1%})")

st.info("Audio upload for automatic jitter/shimmer coming in next version if you want it.")
