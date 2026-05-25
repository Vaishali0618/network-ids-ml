import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

# Load test data
X_test = np.load("../data/X_test.npy")
df = pd.read_csv("../data/cleaned_data.csv")
feature_names = df.drop('Label', axis=1).columns.tolist()

# Load Random Forest model
rf_model = joblib.load("../models/random_forest.pkl")

print("Calculating SHAP values... (may take 2-3 minutes)")

# Use a sample of 500 rows for speed
X_sample = X_test[:500]

# SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated!")

# Summary plot
shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.title("SHAP Feature Importance — Random Forest")
plt.tight_layout()
plt.savefig("../reports/shap_summary.png", dpi=150, bbox_inches='tight')
plt.close()

print("SHAP plot saved to reports/shap_summary.png")

# Bar chart — cleaner view of top 10 features
shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.title("Top 10 Most Important Features — Random Forest")
plt.tight_layout()
plt.savefig("../reports/shap_bar.png", dpi=150, bbox_inches='tight')
plt.close()

print("SHAP bar chart saved to reports/shap_bar.png")