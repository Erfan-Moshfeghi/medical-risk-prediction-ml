"""Simple command‑line prediction script.

This script loads the trained model and makes a prediction for a single
example defined within the script.  It is intended as a quick way to
verify that the inference pipeline works end‑to‑end.  For real usage,
adapt the `sample_input` dictionary to match your dataset's feature names.
"""

from src.inference import load_model, predict_single


def main() -> None:
    # Define a sample input.  Update the keys to match the column names in
    # your dataset.  Values provided here are placeholders.
    sample_input = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }

    pred_class, prob, interpretation = predict_single(sample_input)
    if pred_class is None:
        print(interpretation)
    else:
        print(f"Predicted class: {pred_class}")
        if prob is not None:
            print(f"Probability of heart disease: {prob:.2f}")
        print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()