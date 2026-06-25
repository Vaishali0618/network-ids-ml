import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load saved splits
X_train = np.load("../data/X_train.npy")
X_test = np.load("../data/X_test.npy")
y_train = np.load("../data/y_train.npy")
y_test = np.load("../data/y_test.npy")

print("Data loaded successfully!")
print("Training Random Forest...")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Training complete!")

# Predict
y_pred = rf_model.predict(X_test)

# Results
print("\n Random Forest Results :-")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'DDoS']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save the model
joblib.dump(rf_model, "../models/random_forest.pkl")
print("\nModel saved to models/random_forest.pkl")