import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# load my data
X_test = np.load("../data/X_test.npy")
y_test = np.load("../data/y_test.npy")

# load all 3 models
model_rf = joblib.load("../models/random_forest.pkl")
model_xgb = joblib.load("../models/xgboost.pkl")
model_svm = joblib.load("../models/svm.pkl")

# get predictions from each
pred_rf = model_rf.predict(X_test)
pred_xgb = model_xgb.predict(X_test)
pred_svm = model_svm.predict(X_test)

# plot confusion matrices for all 3
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

models = [("Random Forest", pred_rf), ("XGBoost", pred_xgb), ("SVM", pred_svm)]

for ax, (name, preds) in zip(axes, models):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign','DDoS'],
                yticklabels=['Benign','DDoS'])
    ax.set_title(name)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

plt.tight_layout()
plt.savefig("../reports/confusion_matrices.png", dpi=150)
plt.close()
print("confusion matrix chart saved!")

# accuracy comparison bar chart
names = ["Random Forest", "XGBoost", "SVM"]
scores = [
    accuracy_score(y_test, pred_rf) * 100,
    accuracy_score(y_test, pred_xgb) * 100,
    accuracy_score(y_test, pred_svm) * 100
]

plt.figure(figsize=(7, 4))
bars = plt.bar(names, scores, color=['steelblue', 'orange', 'green'])
plt.ylim(98, 100.5)
plt.ylabel("accuracy %")
plt.title("model accuracy comparison")
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{score:.2f}%", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("../reports/accuracy_comparison.png", dpi=150)
plt.close()
print("accuracy chart saved!")