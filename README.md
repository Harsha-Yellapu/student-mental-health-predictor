# 🧠 Student Mental Health Predictor

An ML-powered web app that predicts a student's mental health risk level
(**Low / Moderate / High**) from daily lifestyle inputs — sleep, stress,
study load, social activity, screen time, and more — and explains *why*
using SHAP, with personalized recommendations.

## Problem

Over 35% of university students experience depression or anxiety symptoms
but go undetected due to a lack of accessible, early screening tools.
Most students never get a simple, judgment-free check-in on their mental
wellbeing patterns.

## Solution

This project trains an **XGBoost classifier** on lifestyle features (sleep
hours, study hours, stress level, social activity, screen time, exercise
frequency, academic pressure, financial stress) to predict risk level, and
uses **SHAP** to make the prediction interpretable — showing exactly which
habits are driving the result. The model is served through an interactive
**Streamlit** app that gives instant predictions and tailored suggestions.

## Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `Pandas` · `Streamlit`

## Project Structure

```
student-mental-health-predictor/
├── generate_data.py      # Synthetic dataset generator (domain-informed)
├── train_model.py        # Trains and evaluates the XGBoost model
├── app.py                 # Streamlit web application
├── requirements.txt
└── README.md
```

## How It Works

1. **Data**: `generate_data.py` builds a domain-informed synthetic dataset
   of 3,000 student profiles using realistic distributions for sleep,
   stress, study load, etc., with a risk label derived from a weighted
   combination of known mental-health risk factors plus noise.
   > Note: in production this would be replaced by an anonymized,
   > consented survey dataset — the generator exists so this project is
   > fully reproducible without needing access to private student data.
2. **Model**: `train_model.py` trains an XGBoost multi-class classifier,
   achieving **~84% test accuracy** (Low/Moderate/High).
3. **Explainability**: SHAP TreeExplainer breaks down each prediction into
   per-feature contributions, shown as a bar chart in the app.
4. **App**: `app.py` is a Streamlit interface where a user adjusts sliders
   for their habits and instantly sees their predicted risk, probability
   breakdown, SHAP explanation, and personalized tips.

## Run It Locally

```bash
git clone https://github.com/Harsha-Yellapu/student-mental-health-predictor.git
cd student-mental-health-predictor
pip install -r requirements.txt

python generate_data.py     # creates student_mental_health_data.csv
python train_model.py       # trains model.pkl, label_encoder.pkl, features.pkl
streamlit run app.py        # launches the web app
```

## Disclaimer

This is a portfolio/educational project. It is **not** a diagnostic or
clinical tool. If you or someone you know is struggling, please reach out
to a licensed mental health professional.

## Author

**Yellapu Harsha** — [LinkedIn](https://www.linkedin.com/in/harshayellapu120/) · [GitHub](https://github.com/Harsha-Yellapu)
