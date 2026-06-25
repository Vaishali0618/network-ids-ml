import numpy as np
import pandas as pd
import joblib
import shap
from transformers import pipeline

# load data
X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")
df = pd.read_csv("../data/cleaned_data.csv")
feature_names = df.drop('Label', axis=1).columns.tolist()

# load both models
xgb_model = joblib.load("../models/xgboost.pkl")
rf_model = joblib.load("../models/random_forest.pkl")
print("models loaded!")

# load llm
print("loading llm...")
generator = pipeline("text-generation", model="distilgpt2")

def get_summary(sample, label, confidence, top_feature):
    prompt = f"Security Alert: The IDS detected a {label} attack with {confidence}% confidence. The main indicator was {top_feature}. Security team should"
    result = generator(prompt, max_new_tokens=40, do_sample=True,
                       temperature=0.7, repetition_penalty=2.0)
    return result[0]['generated_text']

# test on benign sample
sample1 = X_test[0].reshape(1, -1)
pred1 = xgb_model.predict(sample1)[0]
conf1 = round(xgb_model.predict_proba(sample1)[0][pred1] * 100, 2)
label1 = "DDoS" if pred1 == 1 else "Benign"
shap_vals1 = rf_model.predict(sample1)
explainer = shap.TreeExplainer(rf_model)
sv1 = explainer.shap_values(sample1)
top_feature1 = feature_names[np.argmax(np.abs(sv1[0]))]

print(f"\nBenign Sample — Predicted: {label1} | Confidence: {conf1}%")
print(f"Top feature: {top_feature1}")
print(get_summary(sample1, label1, conf1, top_feature1))

# test on ddos sample
ddos_idx = np.where(y_test == 1)[0][0]
sample2 = X_test[ddos_idx].reshape(1, -1)
pred2 = xgb_model.predict(sample2)[0]
conf2 = round(xgb_model.predict_proba(sample2)[0][pred2] * 100, 2)
label2 = "DDoS" if pred2 == 1 else "Benign"
sv2 = explainer.shap_values(sample2)
top_feature2 = feature_names[np.argmax(np.abs(sv2[0]))]

print(f"\nDDoS Sample — Predicted: {label2} | Confidence: {conf2}%")
print(f"Top feature: {top_feature2}")
print(get_summary(sample2, label2, conf2, top_feature2))