"""
generate_data.py
Generates a realistic synthetic dataset of student lifestyle metrics and a
derived mental health risk label (Low / Moderate / High).

In a real-world version of this project, this script would be replaced by
an anonymized survey dataset. The synthetic generator here uses domain
knowledge (sleep, stress, study load, social activity, screen time) to
build a label so the project is fully reproducible end-to-end.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000


def generate():
    sleep_hours = np.clip(np.random.normal(6.5, 1.5, N), 2, 11)
    study_hours = np.clip(np.random.normal(5, 2.2, N), 0, 14)
    stress_level = np.clip(np.random.normal(5.5, 2.2, N), 1, 10)  # self-reported 1-10
    social_activity = np.clip(np.random.normal(5, 2.5, N), 0, 10)  # 1-10 scale
    screen_time = np.clip(np.random.normal(6, 2.5, N), 0, 16)  # hours/day
    exercise_freq = np.clip(np.random.poisson(2.5, N), 0, 7)  # days/week
    academic_pressure = np.clip(np.random.normal(6, 2, N), 1, 10)
    financial_stress = np.clip(np.random.normal(4.5, 2.5, N), 1, 10)

    # Composite risk score built from domain-informed weighted factors
    risk_score = (
        (10 - sleep_hours) * 1.3
        + stress_level * 1.6
        + academic_pressure * 1.1
        + financial_stress * 0.8
        + screen_time * 0.5
        - social_activity * 0.9
        - exercise_freq * 0.8
        + np.random.normal(0, 1.2, N)  # noise
    )

    # Bucket into 3 classes using quantile-like thresholds
    low_thresh = np.percentile(risk_score, 45)
    high_thresh = np.percentile(risk_score, 80)

    labels = np.where(
        risk_score < low_thresh, "Low",
        np.where(risk_score < high_thresh, "Moderate", "High")
    )

    df = pd.DataFrame({
        "sleep_hours": sleep_hours.round(1),
        "study_hours": study_hours.round(1),
        "stress_level": stress_level.round(1),
        "social_activity": social_activity.round(1),
        "screen_time": screen_time.round(1),
        "exercise_freq": exercise_freq,
        "academic_pressure": academic_pressure.round(1),
        "financial_stress": financial_stress.round(1),
        "risk_label": labels,
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("student_mental_health_data.csv", index=False)
    print(f"Generated {len(df)} rows")
    print(df["risk_label"].value_counts())
