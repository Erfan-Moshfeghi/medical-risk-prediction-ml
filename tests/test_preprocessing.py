"""Tests for the preprocessing module."""

import pandas as pd

from src.preprocessing import build_preprocessor


def test_preprocessor_fits_and_transforms():
    """The preprocessing pipeline should fit and transform a simple DataFrame."""
    # Create a synthetic DataFrame with numeric and categorical columns
    df = pd.DataFrame(
        {
            "age": [60, 45, 50],
            "sex": [1, 0, 1],
            "cp": [0, 2, 1],
            "chol": [200.0, None, 180.0],
            "fbs": [1, 0, 1],
        }
    )
    preprocessor = build_preprocessor(df)
    # Fit and transform should not raise
    transformed = preprocessor.fit_transform(df)
    # Check that the output has the expected number of rows
    assert transformed.shape[0] == df.shape[0]