"""
FactorySenseIoT - Milestone 2
File: train_rul_model.py
Purpose: Train, evaluate, and save the NASA Turbofan Remaining Useful Life (RUL)
         regression model using Linear Regression (baseline) and Random Forest Regressor.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_PATH = os.path.join("data", "NASA_CMAPSS", "prepared_train_FD001.csv")
MODELS_DIR = "models"
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "rul_regressor.joblib")
SCALER_OUTPUT_PATH = os.path.join(MODELS_DIR, "rul_scaler.joblib")
METADATA_OUTPUT_PATH = os.path.join(MODELS_DIR, "rul_metadata.json")


def train_and_evaluate_rul():
    print("=" * 68)
    print(" FactorySenseIoT: NASA Turbofan RUL Regression Model Training")
    print("=" * 68)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Prepared data not found at '{DATA_PATH}'. Run prepare_nasa.py first."
        )

    # 1. Load prepared data
    print(f"\n[1/6] Loading prepared dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      Total records: {len(df):,}")

    # 2. Select Features (drop zero-variance sensors and constant settings)
    # Target is RUL
    # Identifiers: unit_id, time_cycle (not physical sensor attributes)
    zero_variance_cols = ["op_setting_3", "sensor_1", "sensor_10", "sensor_18", "sensor_19"]
    ignore_cols = ["unit_id", "time_cycle", "RUL"] + zero_variance_cols
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    print(f"\n[2/6] Selected {len(feature_cols)} sensor/operating features for training:")
    print(f"      {feature_cols}")

    # 3. Engine-Level Train/Validation Split (Group Split)
    # Engines 1-80: Training set (~80%)
    # Engines 81-100: Validation set (~20%)
    print("\n[3/6] Performing Engine-Level Group Split (to prevent temporal leakage)...")
    train_mask = df["unit_id"] <= 80
    test_mask = df["unit_id"] > 80

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, "RUL"]

    X_val = df.loc[test_mask, feature_cols]
    y_val = df.loc[test_mask, "RUL"]

    print(f"      Training samples:   {len(X_train):,} rows (from 80 engines)")
    print(f"      Validation samples: {len(X_val):,} rows (from 20 engines)")

    # 4. Feature Scaling (StandardScaler)
    # Fit scaler ONLY on training data to prevent data leakage
    print("\n[4/6] Standardizing features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 5. Train and Compare Models
    print("\n[5/6] Training and comparing regression models...")

    # --- Model A: Linear Regression (Baseline) ---
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_val_scaled)

    mae_lr = mean_absolute_error(y_val, y_pred_lr)
    rmse_lr = np.sqrt(mean_squared_error(y_val, y_pred_lr))
    r2_lr = r2_score(y_val, y_pred_lr)

    print("\n      --- [Baseline] Linear Regression ---")
    print(f"      MAE:  {mae_lr:.2f} cycles")
    print(f"      RMSE: {rmse_lr:.2f} cycles")
    print(f"      R2:   {r2_lr:.4f}")

    # --- Model B: Random Forest Regressor ---
    print("\n      --- Training Random Forest Regressor (100 estimators)... ---")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)  # Random Forest works directly on unscaled or scaled features
    y_pred_rf = rf.predict(X_val)

    mae_rf = mean_absolute_error(y_val, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_val, y_pred_rf))
    r2_rf = r2_score(y_val, y_pred_rf)

    print(f"      MAE:  {mae_rf:.2f} cycles")
    print(f"      RMSE: {rmse_rf:.2f} cycles")
    print(f"      R2:   {r2_rf:.4f}")

    # Select the superior model
    selected_model_name = "RandomForestRegressor" if rmse_rf < rmse_lr else "LinearRegression"
    selected_model = rf if selected_model_name == "RandomForestRegressor" else lr
    print(f"\n      Selected Model: {selected_model_name} (achieved lowest RMSE).")

    # 6. Save Model Artifacts
    print(f"\n[6/6] Saving model artifacts to '{MODELS_DIR}' directory...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(selected_model, MODEL_OUTPUT_PATH)
    joblib.dump(scaler, SCALER_OUTPUT_PATH)

    metadata = {
        "model_type": selected_model_name,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "validation_metrics": {
            "linear_regression": {"mae": round(mae_lr, 2), "rmse": round(rmse_lr, 2), "r2": round(r2_lr, 4)},
            "random_forest": {"mae": round(mae_rf, 2), "rmse": round(rmse_rf, 2), "r2": round(r2_rf, 4)},
        },
        "target": "RUL (Remaining Useful Life in cycles)",
    }

    with open(METADATA_OUTPUT_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"      Saved model:    {MODEL_OUTPUT_PATH}")
    print(f"      Saved scaler:   {SCALER_OUTPUT_PATH}")
    print(f"      Saved metadata: {METADATA_OUTPUT_PATH}")

    # Show top 5 most important degradation sensors
    if hasattr(selected_model, "feature_importances_"):
        print("\nTop 5 Most Informative Sensors (Feature Importance):")
        importances = pd.Series(selected_model.feature_importances_, index=feature_cols)
        top5 = importances.sort_values(ascending=False).head(5)
        for rank, (sensor, imp) in enumerate(top5.items(), start=1):
            print(f"      {rank}. {sensor}: {imp * 100:.2f}% importance")

    print("\n" + "=" * 68)
    print(" SUCCESS: RUL Regression Model trained, evaluated, and persisted.")
    print("=" * 68)


if __name__ == "__main__":
    train_and_evaluate_rul()