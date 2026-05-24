import pandas as pd
import numpy as np
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import os

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = "heart_preprocessing/heart_preprocessing.csv"
TARGET_COL = "target"
EXPERIMENT_NAME = "Heart-Disease-Classification"
MLFLOW_URI = "mlruns"          # local
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable autolog
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Autolog already handles metrics, but log extra ones for clarity
    mlflow.log_metric("test_accuracy",  accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall",    recall_score(y_test, y_pred))
    mlflow.log_metric("test_f1",        f1_score(y_test, y_pred))

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

# Log model ke MLflow
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model"
)

print("Model successfully logged to MLflow!")

if __name__ == "__main__":
    main()
