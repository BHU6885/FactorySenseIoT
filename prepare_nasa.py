"""
FactorySenseIoT - Milestone 1
File: prepare_nasa.py
Purpose: Load raw NASA C-MAPSS FD001 training data, assign standard column names,
         calculate Remaining Useful Life (RUL) target, perform quality checks,
         and export a clean, labeled CSV for model training.
"""

import os
import pandas as pd
import numpy as np

# 1. Define dataset paths
RAW_DATA_PATH = os.path.join("data", "NASA_CMAPSS", "train_FD001.txt")
OUTPUT_DATA_PATH = os.path.join("data", "NASA_CMAPSS", "prepared_train_FD001.csv")

# 2. Define standard C-MAPSS column names (26 columns total)
INDEX_COLUMNS = ["unit_id", "time_cycle"]
SETTING_COLUMNS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def prepare_nasa_training_data():
    print("=" * 65)
    print(" FactorySenseIoT: NASA C-MAPSS FD001 Data Preparation")
    print("=" * 65)

    # Verify raw file exists
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find '{RAW_DATA_PATH}'. Please ensure the file is placed correctly."
        )

    # 3. Load whitespace-separated data
    print(f"\n[1/5] Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)
    print(f"      Rows loaded: {df.shape[0]:,}")
    print(f"      Columns loaded: {df.shape[1]}")

    # 4. Check for missing values
    null_counts = df.isnull().sum().sum()
    print(f"\n[2/5] Missing value check: {null_counts} null values found.")

    # 5. Calculate RUL for each engine unit
    # For run-to-failure data: RUL = (max observed cycle for that unit) - (current cycle)
    print("\n[3/5] Calculating Remaining Useful Life (RUL)...")
    max_cycles = df.groupby("unit_id")["time_cycle"].transform("max")
    df["RUL"] = max_cycles - df["time_cycle"]

    num_engines = df["unit_id"].nunique()
    min_engine_life = max_cycles.min()
    max_engine_life = max_cycles.max()
    avg_engine_life = df.groupby("unit_id")["time_cycle"].max().mean()

    print(f"      Total unique engines: {num_engines}")
    print(f"      Shortest engine life: {min_engine_life} cycles")
    print(f"      Longest engine life:  {max_engine_life} cycles")
    print(f"      Average engine life:  {avg_engine_life:.1f} cycles")
    print(f"      Target variable 'RUL' created (range: {df['RUL'].min()} to {df['RUL'].max()} cycles).")

    # 6. Quality Check: Detect constant/zero-variance sensor columns
    # In FD001, some sensors have no change because operating conditions are constant (sea level)
    print("\n[4/5] Checking sensor variance...")
    std_series = df[SENSOR_COLUMNS].std()
    constant_sensors = std_series[std_series == 0].index.tolist()
    if constant_sensors:
        print(f"      Note for feature selection: Sensors with zero variance: {constant_sensors}")
    else:
        print("      All sensors have non-zero variance.")

    # 7. Save prepared dataset
    print(f"\n[5/5] Saving prepared dataset to: {OUTPUT_DATA_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print("      File successfully saved!")

    # Display preview sample
    print("\nSample Preview (First 5 rows of unit 1):")
    preview_cols = ["unit_id", "time_cycle", "sensor_2", "sensor_3", "sensor_4", "RUL"]
    print(df[preview_cols].head())

    print("\n" + "=" * 65)
    print(" SUCCESS: NASA training data prepared and ready for model training.")
    print("=" * 65)


if __name__ == "__main__":
    prepare_nasa_training_data()