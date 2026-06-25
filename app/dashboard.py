import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from transformers import pipeline as llm_pipeline

st.set_page_config(page_title="Network IDS Dashboard", layout="wide")
st.title("Network Intrusion Detection Dashboard")
st.write("ML-based DDoS detection with Explainable AI and LLM alert summarisation")

@st.cache_resource
def load_models():
    xgb_model = joblib.load("../models/xgboost.pkl")
    rf_model = joblib.load("../models/random_forest.pkl")
    return xgb_model, rf_model

@st.cache_resource
def load_llm():
    return llm_pipeline("text-generation", model="distilgpt2")

xgb_model, rf_model = load_models()
generator = load_llm()
st.success("Models loaded successfully!")

X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")
df = pd.read_csv("../data/cleaned_data.csv")
feature_names = df.drop('Label', axis=1).columns.tolist()

st.write(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")

st.divider()
st.header("Test a Network Traffic Sample")

sample_num = st.number_input("Enter sample index (0 to 44252)",
                              min_value=0, max_value=44252, value=0, step=1)

if st.button("Analyze Sample"):
    sample = X_test[sample_num].reshape(1, -1)
    actual = y_test[sample_num]

    pred = xgb_model.predict(sample)[0]
    confidence = round(xgb_model.predict_proba(sample)[0][pred] * 100, 2)
    label = "DDoS" if pred == 1 else "Benign"
    actual_label = "DDoS" if actual == 1 else "Benign"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual Label", actual_label)
    with col2:
        st.metric("Predicted Label", label)
    with col3:
        st.metric("Confidence", f"{confidence}%")

    if label == "DDoS":
        st.error(f"DDoS Attack Detected!!! Confidence: {confidence}%")
    else:
        st.success(f"Benign Traffic. Confidence: {confidence}%")

    st.divider()
    st.subheader("SHAP Explanation")

    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(sample)
    top_idx = np.argmax(np.abs(shap_vals[0]))
    top_feature = feature_names[top_idx]
    top_value = round(float(np.array(shap_vals[0][top_idx]).flatten()[0]), 4)

    st.write(f"**Top contributing feature:** `{top_feature}`")
    st.write(f"**SHAP value:** {top_value}")

    shap_flat = np.abs(shap_vals[0]).flatten()
    top5_idx = np.argsort(shap_flat)[-5:][::-1]
    top5_features = [feature_names[int(i) % len(feature_names)] for i in top5_idx]
    top5_values = [float(shap_vals[0].flatten()[int(i)]) for i in top5_idx]

    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.tick_params(labelsize=7)
    ax.barh(top5_features[::-1], top5_values[::-1], color='steelblue')
    ax.set_xlabel("SHAP Value")
    ax.set_title("Top 5 Features")
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.subheader("LLM Alert Summary")

    with st.spinner("Generating summary..."):
        prompt = f"Security Alert: The IDS detected a {label} attack with {confidence}% confidence. The main indicator was {top_feature}. Security team should"
        result = generator(prompt, max_new_tokens=40, do_sample=True,
                          temperature=0.7, repetition_penalty=2.0)
        summary = result[0]['generated_text']

    st.info(summary)