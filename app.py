"""Streamlit demo app for the heart disease risk prediction system.

This app provides a simple web interface where users can input
clinical variables and obtain a risk prediction from the trained model.
If the model file is missing the app will display a friendly error
message instructing the user to run the training script first.
"""

import streamlit as st
from src.inference import load_model, predict_single


def main() -> None:
    st.set_page_config(page_title="Medical Risk Prediction System", layout="centered")
    st.title("Medical Risk Prediction System")
    st.write(
        "This educational app predicts the risk of heart disease based on a set of clinical features."
    )
    st.warning(
        "**Disclaimer:** This model is for educational purposes only and must not be used for real medical diagnoses."
    )

    # Load the model once at the start
    model = load_model()
    if model is None:
        st.error("Model not found. Please run `python train_model.py` to train the model before using the app.")
        return

    st.sidebar.header("Input Features")
    # Create input widgets for common features.  Defaults are arbitrary examples.
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=55)
    sex = st.sidebar.selectbox("Sex", options={0: "Female", 1: "Male"}, format_func=lambda x: {0: "Female", 1: "Male"}[x])
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        options={0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
        format_func=lambda x: {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "Asymptomatic",
        }[x],
    )
    trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
    chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options={0: "False", 1: "True"}, format_func=lambda x: {0: "False", 1: "True"}[x])
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        options={0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        }[x],
    )
    thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", options={0: "No", 1: "Yes"}, format_func=lambda x: {0: "No", 1: "Yes"}[x])
    oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        options={0: "Upsloping", 1: "Flat", 2: "Downsloping"},
        format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x],
    )
    ca = st.sidebar.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=3, value=0, step=1)
    thal = st.sidebar.selectbox(
        "Thalassemia",
        options={1: "Normal", 2: "Fixed defect", 3: "Reversible defect"},
        format_func=lambda x: {1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}[x],
    )

    if st.button("Predict Risk"):
        input_data = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }
        pred_class, prob, interpretation = predict_single(input_data)
        if pred_class is None:
            st.error(interpretation)
        else:
            st.success(f"Prediction: {interpretation}")
            if prob is not None:
                st.info(f"Estimated risk probability: {prob:.2f}")
            st.write(
                "\nThis result should not be used for medical decisions. Always consult a healthcare professional."
            )


if __name__ == "__main__":
    main()