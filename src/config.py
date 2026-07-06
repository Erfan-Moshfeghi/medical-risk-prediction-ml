"""Configuration module for paths used throughout the project.

All file system paths are defined relative to the project root using
`pathlib.Path`.  This centralises path management and avoids hardcoding
absolute locations elsewhere in the code.  When modifying the project
structure, update these constants accordingly.
"""

from pathlib import Path

# Determine the project root as the parent of the src directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Default raw dataset path
RAW_DATA_PATH = DATA_RAW_DIR / "heart_disease.csv"

# Model directory and file names
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Reports and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"