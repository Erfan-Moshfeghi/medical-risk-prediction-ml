"""Inference utilities for the heart disease risk prediction system.

This module loads a persisted model and provides functions to make
predictions on new data.  It gracefully handles missing model files and
provides a simple interpretation of the result.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import pandas as pd
import joblib

from . import config


def load_model() -> Optional[Any]:
    """Load the trained model from disk.

    Returns
    -------
    model : object or None
        The deserialised model pipeline, or `None` if the model file is
        missing.
    """
    model_path = config.MODEL_PATH
    if not model_path.exists():
        return None
    try:
        model = joblib.load(model_path)
    except Exception:
        return None
    return model


def predict_single(input_data: Dict[str, Any] | pd.DataFrame) -> Tuple[Any, Optional[float], str]:
    """Predict the heart disease risk for a single observation.

    Parameters
    ----------
    input_data : dict or pd.DataFrame
        A dictionary mapping feature names to values, or a one‑row
        DataFrame.  All expected feature columns must be present.  Extra
        columns will be ignored by the model's preprocessing pipeline.

    Returns
    -------
    Tuple[predicted_class, probability, interpretation]
        A tuple containing the predicted class label (0 or 1), the
        probability of class 1 if available, and a short interpretation
        string (e.g. "High risk" or "Low risk").  If the model is
        missing, returns `(None, None, message)`.
    """
    model = load_model()
    if model is None:
        return None, None, (
            "Model not found. Please run python train_model.py to train the model before making predictions."
        )

    # Convert input to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be a dictionary or a pandas DataFrame")

    # Ensure DataFrame has correct columns; missing columns will be handled
    # by the preprocessing pipeline's imputation/encoding
    try:
        pred_class = model.predict(df)[0]
        # Attempt to get probability of the positive class
        prob: Optional[float]
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(df)[0][1])
        else:
            # Some models provide a decision function; transform to probability via sigmoid
            import numpy as np
            scores = model.decision_function(df)[0]
            prob = 1 / (1 + np.exp(-scores))
    except Exception as exc:
        return None, None, f"Failed to make prediction: {exc}"

    interpretation = "High risk of heart disease" if pred_class == 1 else "Low risk of heart disease"
    return pred_class, prob, interpretation