# FactorySenseIoT
# AI4I Industrial Machine Failure Classification

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "AI4I" / "ai4i2020.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("FactorySenseIoT: Machine Failure Classification")
print("=" * 60)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

print("\n[1/6] Loading AI4I dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nAvailable columns:")
print(df.columns.tolist())

# Remove accidental spaces from column names
df.columns = df.columns.str.strip()

# --------------------------------------------------
# 3. Prepare features and target
# --------------------------------------------------

print("\n[2/6] Preparing features and target...")

target_column = "Machine failure"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' not found. "
        "Check the CSV column names."
    )

# These columns are excluded:
# UDI and Product ID are identifiers.
# TWF, HDF, PWF, OSF and RNF describe failure modes
# and must not be used to predict the overall failure
# because they can reveal information about the target.

columns_to_exclude = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X = df.drop(
    columns=columns_to_exclude,
    errors="ignore"
)

y = df[target_column].astype(int)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("Input features:", X.columns.tolist())
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

print("\nFailure class distribution:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 4. Split the dataset
# --------------------------------------------------

print("\n[3/6] Splitting training and testing data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 5. Build and train the model
# --------------------------------------------------

print("\n[4/6] Training Random Forest classifier...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ]
)

model.fit(X_train, y_train)

print("Model training completed.")

# --------------------------------------------------
# 6. Evaluate and save the model
# --------------------------------------------------

print("\n[5/6] Evaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No Failure", "Failure"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

# ROC-AUC measures how well the model separates the classes.
if len(set(y_test)) == 2:
    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    print(f"\nROC-AUC: {auc:.4f}")
else:
    auc = None
    print("\nROC-AUC unavailable: only one class in test data.")

print("\n[6/6] Saving trained model...")

model_path = MODEL_DIR / "failure_classifier.joblib"
metadata_path = MODEL_DIR / "failure_metadata.json"

joblib.dump(model, model_path)

metadata = {
    "model_name": "Random Forest Classifier",
    "dataset": "AI4I 2020 Predictive Maintenance",
    "target": target_column,
    "features": X.columns.tolist(),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "accuracy": float(accuracy),
    "roc_auc": float(auc) if auc is not None else None,
    "random_state": 42,
    "test_size": 0.20
}

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nSaved model:", model_path)
print("Saved metadata:", metadata_path)

print("\n" + "=" * 60)
print("SUCCESS: Failure classification model trained!")
print("=" * 60)