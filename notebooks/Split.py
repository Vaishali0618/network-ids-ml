import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load cleaned data
df = pd.read_csv("../data/cleaned_data.csv")

# Separate features (X) and label (y)
X = df.drop('Label', axis=1)
y = df['Label']

print("Features shape:", X.shape)
print("Label shape:", y.shape)
print("Label distribution:\n", y.value_counts())

# Scale features (bring all columns to same range)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaling done!")
print("Sample mean (should be 0):", X_scaled.mean().round(4))
print("Sample std (should be 1):", X_scaled.std().round(4))

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

import numpy as np

# Save all splits
np.save("../data/X_train.npy", X_train)
np.save("../data/X_test.npy", X_test)
np.save("../data/y_train.npy", y_train)
np.save("../data/y_test.npy", y_test)

print("\nAll splits saved to data folder!")
print("Files: X_train.npy, X_test.npy, y_train.npy, y_test.npy")