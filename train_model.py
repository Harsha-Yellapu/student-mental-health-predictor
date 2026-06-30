"""
train_model.py
Trains an XGBoost classifier to predict student mental health risk level
(Low / Moderate / High) from lifestyle features, evaluates accuracy, and
saves the model + label encoder + SHAP explainer for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

FEATURES = [
    "sleep_hours", "study_hours", "stress_level", "social_activity",
    "screen_time", "exercise_freq", "academic_pressure", "financial_stress",
]


def main():
    df = pd.read_csv("student_mental_health_data.csv")

    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df["risk_label"])  # High/Low/Moderate -> 0/1/2

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, preds, target_names=le.classes_))

    joblib.dump(model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(FEATURES, "features.pkl")
    print("Saved model.pkl, label_encoder.pkl, features.pkl")


if __name__ == "__main__":
    main()
