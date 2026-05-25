import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet("../data/DDoS-Friday-no-metadata.parquet")

print("Shape before cleaning:", df.shape)

# Replace infinite values with NaN, then drop those rows
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print("Shape after dropping inf/NaN:", df.shape)

# Encode labels: DDoS = 1, Benign = 0
df['Label'] = df['Label'].map({'DDoS': 1, 'Benign': 0})
print("\nLabel encoding done:")
print(df['Label'].value_counts())

# Drop zero-variance columns (columns where all values are the same)
before = df.shape[1]
df = df.loc[:, df.nunique() > 1]
after = df.shape[1]
print(f"\nColumns dropped (zero variance): {before - after}")
print(f"Remaining columns: {after}")

# Save cleaned data
df.to_csv("../data/cleaned_data.csv", index=False)
print("\nClean data saved to data/cleaned_data.csv")
print("Final shape:", df.shape)
