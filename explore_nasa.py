import pandas as pd

# Define the column names in the NASA FD001 dataset

columns = [
"engine_id", "cycle",
"setting_1", "setting_2", "setting_3"
]

# Add the 21 sensor column names

columns += [f"sensor_{i}" for i in range(1, 22)]

# Read the NASA training dataset

file_path = "data/NASA_CMAPSS/train_FD001.txt"

df = pd.read_csv(
file_path,
sep=r"\s+",
header=None,
names=columns
)

# Display the first five rows

print("First five rows:")
print(df.head())

# Display the dataset dimensions

print("\nDataset dimensions:")
print(df.shape)

# Display column information

print("\nDataset information:")
print(df.info())

# Show how many different engines are in the dataset

print("\nNumber of unique engines:")
print(df["engine_id"].nunique())

# Show the first few engine IDs

print("\nFirst 10 unique engine IDs:")
print(df["engine_id"].unique()[:10])

# Find the final recorded cycle for each engine

print("\nFinal recorded cycle for the first 10 engines:")
print(df.groupby("engine_id")["cycle"].max().head(10))
