"""Tests for the inference module."""

import os

from src import config
from src.inference import load_model, predict_single


def test_load_model_when_missing(tmp_path):
    """load_model should return None when the model file is missing."""
    # Temporarily override MODEL_PATH to a path that does not exist
    original_model_path = config.MODEL_PATH
    try:
        config.MODEL_PATH = tmp_path / "nonexistent.joblib"
        model = load_model()
        assert model is None
    finally:
        config.MODEL_PATH = original_model_path


def test_predict_single_handles_missing_model(tmp_path):
    """predict_single should gracefully handle missing models and return a message."""
    # Override model path to ensure model is missing
    original_model_path = config.MODEL_PATH
    try:
        config.MODEL_PATH = tmp_path / "nonexistent.joblib"
        pred_class, prob, message = predict_single({"age": 50})
        assert pred_class is None
        assert prob is None
        assert "run python train_model.py" in message
    finally:
        config.MODEL_PATH = original_model_path