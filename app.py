"""
app.py
Streamlit web app for the Student Mental Health Predictor.
Takes lifestyle inputs, predicts risk level (Low/Moderate/High) using a
trained XGBoost model, and explains the prediction with SHAP values.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
    features = joblib.load("features.pkl")
    explainer = shap.TreeExplainer(model)
    return model, le, features, explainer

model, le, features, explainer = load_artifacts()

st.title("🧠 Student Mental Health Risk Predictor")
st.write(
    "Estimate your mental health risk level based on daily lifestyle habits. "
    "This tool is for awareness purposes only and is **not a substitute for "
    "professional medical advice**."
)

st.subheader("Enter your daily habits")

col1, col2 = st.columns(2)
with col1:
    sleep_hours = st.slider("Sleep hours (per night)", 2.0, 11.0, 6.5, 0.5)
    study_hours = st.slider("Study hours (per day)", 0.0, 14.0, 5.0, 0.5)
    stress_level = st.slider("Stress level (1-10)", 1, 10, 5)
    social_activity = st.slider("Social activity (1-10)", 0, 10, 5)
with col2:
    screen_time = st.slider("Screen time (hours/day)", 0.0, 16.0, 6.0, 0.5)
    exercise_freq = st.slider("Exercise frequency (days/week)", 0, 7, 2)
    academic_pressure = st.slider("Academic pressure (1-10)", 1, 10, 6)
    financial_stress = st.slider("Financial stress (1-10)", 1, 10, 4)

if st.button("Predict Risk Level", type="primary"):
    input_df = pd.DataFrame([[
        sleep_hours, study_hours, stress_level, social_activity,
        screen_time, exercise_freq, academic_pressure, financial_stress
    ]], columns=features)

    pred_idx = model.predict(input_df)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(input_df)[0]

    color = {"Low": "green", "Moderate": "orange", "High": "red"}[pred_label]
    st.markdown(f"### Predicted Risk Level: :{color}[{pred_label}]")

    prob_df = pd.DataFrame({"Risk Level": le.classes_, "Probability": probs})
    st.bar_chart(prob_df.set_index("Risk Level"))

    # Personalized recommendations
    st.subheader("Recommendations")
    tips = []
    if sleep_hours < 6:
        tips.append("Try to increase sleep to at least 7 hours — sleep is one of the strongest predictors of mental wellbeing.")
    if stress_level >= 7:
        tips.append("Your stress level is high. Consider short breaks, breathing exercises, or talking to a counselor.")
    if social_activity <= 3:
        tips.append("Low social engagement detected. Spending time with friends or peers can meaningfully reduce risk.")
    if exercise_freq <= 1:
        tips.append("Try to fit in light exercise 2-3 times a week — even short walks help.")
    if screen_time >= 9:
        tips.append("High screen time. Reducing non-essential screen use, especially before sleep, may help.")
    if not tips:
        tips.append("Your habits look balanced — keep it up!")
    for t in tips:
        st.write(f"- {t}")

    # SHAP explanation
    st.subheader("Why this prediction? (SHAP explanation)")
    shap_values = explainer.shap_values(input_df)
    class_idx = pred_idx
    if isinstance(shap_values, list):
        sv = shap_values[class_idx][0]
    else:
        sv = shap_values[0, :, class_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(np.abs(sv))[::-1]
    feat_names = [features[i] for i in order]
    vals = [sv[i] for i in order]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]
    ax.barh(feat_names[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value (impact on predicted risk)")
    ax.set_title(f"Feature contributions for predicted class: {pred_label}")
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward higher risk; green bars push it toward lower risk."
    )

st.divider()
st.caption(
    "⚠️ This is a portfolio project for demonstration purposes. "
    "If you or someone you know is struggling, please reach out to a mental health professional."
)
