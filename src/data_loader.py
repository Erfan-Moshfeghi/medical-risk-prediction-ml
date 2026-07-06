"""Utilities for loading and validating the dataset.

This module encapsulates the logic of reading the CSV file, detecting the
target column and performing basic validation checks.  Keeping this logic
separate makes it easier to test and reuse.
"""

from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to read CSV file {path}: {exc}") from exc
    return df


def detect_target_column(df: pd.DataFrame) -> str:
    """Detect the target column in the DataFrame.

    The function looks for commonly used target column names: `target`,
    `HeartDisease`, `heart_disease`, `output` or `disease`.  The search is
    case insensitive.  If multiple candidate columns are found, the first
    match in the predefined order is returned.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    str
        Name of the detected target column.

    Raises
    ------
    ValueError
        If no suitable target column is found.
    """
    candidates = [
        "target",
        "HeartDisease",
        "heart_disease",
        "output",
        "disease",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    raise ValueError(
        "Could not detect a target column. Expected one of: " + ", ".join(candidates)
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Perform basic validation checks on the dataset.

    Ensures that the dataset is non‑empty and has at least two columns
    (one feature and one target).  Additional validation logic can be
    implemented here as needed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Raises
    ------
    ValueError
        If the dataset fails basic validation checks.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")