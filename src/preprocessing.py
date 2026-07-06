"""Preprocessing utilities for the heart disease dataset.

The preprocessing pipeline handles missing values and encodes both numeric
and categorical features.  It uses scikit‑learn's `ColumnTransformer` to
apply different transformations to different column types.  Numeric
columns are imputed with the median and standardised, while categorical
columns are imputed with the most frequent value and one‑hot encoded.
"""

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return a list of numeric column names from the DataFrame."""
    numeric_types = ["int16", "int32", "int64", "float16", "float32", "float64"]
    return [c for c in df.columns if df[c].dtype.name in numeric_types]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for the given feature set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame without the target column.

    Returns
    -------
    ColumnTransformer
        A transformer that applies appropriate preprocessing to numeric and
        categorical columns.
    """
    numeric_cols = _get_numeric_columns(X)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Numeric pipeline: median imputation + standardisation
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: most frequent imputation + one‑hot encoding
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine pipelines using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor