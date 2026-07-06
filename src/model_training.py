"""Model training and selection utilities.

This module defines candidate models and provides a function to train
multiple models, evaluate them and select the best one based on
F1‑score (with ROC‑AUC as a secondary criterion).  All models are
classical scikit‑learn estimators with reasonable default parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score


def get_candidate_models() -> List[Tuple[str, Any]]:
    """Return a list of candidate model names and instances.

    The models chosen here are simple and fast to train.  Hyperparameter
    tuning is intentionally omitted to keep the project lightweight.

    Returns
    -------
    List[Tuple[str, Any]]
        A list of (name, estimator) tuples.
    """
    models: List[Tuple[str, Any]] = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        (
            "Random Forest",
            RandomForestClassifier(n_estimators=100, random_state=42),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
        ),
    ]
    return models


def train_and_select_model(
    X_train,
    y_train,
    X_test,
    y_test,
    preprocessor,
) -> Tuple[Pipeline, Dict[str, Dict[str, float]]]:
    """Train candidate models and select the best based on F1‑score.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    preprocessor : ColumnTransformer
        Preprocessing transformer to apply to the data before model fitting.

    Returns
    -------
    best_pipeline : Pipeline
        The fitted pipeline with the best performing model.
    results : Dict[str, Dict[str, float]]
        A dictionary mapping model names to their evaluation metrics
        (F1‑score and ROC‑AUC).  Useful for plotting comparisons.
    """
    candidate_models = get_candidate_models()
    results: Dict[str, Dict[str, float]] = {}
    best_name: str = ""
    best_f1: float = -1.0
    best_auc: float = -1.0
    best_pipeline: Pipeline | None = None

    for name, model in candidate_models:
        # Build a pipeline that first preprocesses the data then fits the model
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        # Fit on training data
        pipeline.fit(X_train, y_train)
        # Predict labels and probabilities on test data
        y_pred = pipeline.predict(X_test)
        # F1‑score (binary case uses "binary" average)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # ROC‑AUC: only computed if the model provides probabilities or decision function
        try:
            if hasattr(pipeline, "predict_proba"):
                y_scores = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_scores = pipeline.decision_function(X_test)
            auc = roc_auc_score(y_test, y_scores)
        except Exception:
            auc = float("nan")
        results[name] = {"f1": f1, "roc_auc": auc}
        # Select best model primarily by F1, then by ROC‑AUC if F1 is equal
        # NaN AUC values are treated as lower than any number
        current_auc = auc if auc == auc else -1.0  # NaN check
        if (f1 > best_f1) or (f1 == best_f1 and current_auc > best_auc):
            best_f1 = f1
            best_auc = current_auc
            best_name = name
            best_pipeline = pipeline

    assert best_pipeline is not None, "No model was trained."
    return best_pipeline, results