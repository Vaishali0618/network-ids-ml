import pandas as pd

df = pd.read_parquet("../data/DDoS-Friday-no-metadata.parquet")

# Basic info
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nLabel counts:\n", df['Label'].value_counts())
print("\nMissing values:", df.isnull().sum().sum())
print("\nColumn types:\n", df.dtypes.value_counts())