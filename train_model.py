"""
train_model.py
Trains an XGBoost classifier to predict student mental health risk level
(Low / Moderate / High) from lifestyle features, evaluates accuracy, logs
the run (params, metrics, model artifact) to MLflow, and saves the model +
label encoder for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

FEATURES = [
    "sleep_hours", "study_hours", "stress_level", "social_activity",
    "screen_time", "exercise_freq", "academic_pressure", "financial_stress",
]

MLFLOW_EXPERIMENT_NAME = "student-mental-health-predictor"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = pd.read_csv("student_mental_health_data.csv")

    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["risk_label"])  # High/Low/Moderate -> 0/1/2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = dict(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )

    with mlflow.start_run(run_name="xgboost_mental_health_risk"):
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        report = classification_report(y_test, preds, target_names=le.classes_)

        print(f"Test Accuracy: {acc * 100:.1f}%")
        print(report)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.xgboost.log_model(model, artifact_path="model")

        joblib.dump(model, "model.pkl")
        joblib.dump(le, "label_encoder.pkl")
        joblib.dump(FEATURES, "features.pkl")
        mlflow.log_artifact("label_encoder.pkl")
        mlflow.log_artifact("features.pkl")

        print("Saved model.pkl, label_encoder.pkl, features.pkl")
        print(f"MLflow run logged under experiment '{MLFLOW_EXPERIMENT_NAME}'.")
        print("Run `mlflow ui` to view metrics, params, and artifacts in the browser.")


if __name__ == "__main__":
    main()
