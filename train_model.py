"""Training script for the heart disease risk prediction project.

Running this script will load the dataset, preprocess it, train several
candidate models, evaluate them and save the best performing model to
disk.  If the dataset is missing, a clear message is printed and the
script exits gracefully.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import load_dataset, detect_target_column, validate_dataset
from src.preprocessing import build_preprocessor
from src.model_training import train_and_select_model
from src.evaluation import evaluate_model, save_metrics, plot_confusion_matrix, plot_model_comparison


def main() -> None:
    # Check whether the raw dataset exists
    data_path = config.RAW_DATA_PATH
    if not data_path.exists():
        print(
            "Dataset not found. Please place heart_disease.csv in data/raw/ and run python train_model.py again."
        )
        return

    # Load dataset
    try:
        df = load_dataset(data_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load dataset: {exc}")
        return

    try:
        validate_dataset(df)
    except Exception as exc:  # noqa: BLE001
        print(f"Dataset validation error: {exc}")
        return

    # Detect target column
    try:
        target_col = detect_target_column(df)
    except Exception as exc:
        print(f"Target column detection error: {exc}")
        return

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Train models and select the best
    best_model, results = train_and_select_model(
        X_train, y_train, X_test, y_test, preprocessor
    )

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save best model
    joblib.dump(best_model, config.MODEL_PATH)

    # Evaluate best model on the test set
    metrics = evaluate_model(best_model, X_test, y_test)

    # Save metrics to JSON
    save_metrics(metrics, config.METRICS_PATH)

    # Plot confusion matrix and model comparison
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(best_model, X_test, y_test, config.FIGURES_DIR / "confusion_matrix.png")
    plot_model_comparison(results, config.FIGURES_DIR / "model_comparison.png")

    # Print summary of results
    print("Training complete. Best model saved to", config.MODEL_PATH)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"  {k}: N/A")


if __name__ == "__main__":
    main()