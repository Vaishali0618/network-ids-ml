import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load saved splits
X_train = np.load("../data/X_train.npy")
X_test = np.load("../data/X_test.npy")
y_train = np.load("../data/y_train.npy")
y_test = np.load("../data/y_test.npy")

print("Data loaded!")
print("Training XGBoost... (may take 2-3 minutes)")

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

print("Training complete!")

# Predict
y_pred_xgb = xgb_model.predict(X_test)

# Results
print("\n--- XGBoost Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_xgb) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['Benign', 'DDoS']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))

# Save model
joblib.dump(xgb_model, "../models/xgboost.pkl")
print("\nXGBoost model saved!")

from sklearn.svm import SVC
from sklearn.utils import resample

print("\nTraining SVM... (may take 3-5 minutes)")

# SVM on a sample (50,000 rows) — standard practice for large datasets
X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=50000, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_sample, y_train_sample)

print("rf model finished")

# Predict
y_pred_svm = svm_model.predict(X_test)

# Results
print("\n--- SVM Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred_svm) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Benign', 'DDoS']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Save model
joblib.dump(svm_model, "../models/svm.pkl")
print("\nSVM model saved!")