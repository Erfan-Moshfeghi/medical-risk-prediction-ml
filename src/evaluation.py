"""Evaluation and plotting utilities.

This module defines functions for computing evaluation metrics, saving
them to disk and generating basic plots such as a confusion matrix and
a bar chart comparing model performance.
"""

from pathlib import Path
from typing import Dict, Any

import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> Dict[str, Any]:
    """Compute standard evaluation metrics for a binary classifier.

    Parameters
    ----------
    model : fitted estimator
        The trained pipeline or model.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing accuracy, precision, recall, F1‑score and
        ROC‑AUC (if available).
    """
    y_pred = model.predict(X_test)
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    # Compute ROC‑AUC when possible
    try:
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.decision_function(X_test)
        metrics["roc_auc"] = roc_auc_score(y_test, y_scores)
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save evaluation metrics to a JSON file.

    Parameters
    ----------
    metrics : dict
        Dictionary of evaluation metrics.
    path : Path
        File path where metrics should be saved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def plot_confusion_matrix(model, X_test, y_test, output_path: Path) -> None:
    """Generate and save a confusion matrix plot.

    Parameters
    ----------
    model : fitted estimator
        The trained model or pipeline.
    X_test : array-like
        Test features.
    y_test : array-like
        True labels.
    output_path : Path
        Destination path for the PNG file.
    """
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_model_comparison(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """Plot a comparison of F1‑scores for different models.

    Parameters
    ----------
    results : dict
        Dictionary mapping model names to a dictionary of metrics.  Each
        inner dictionary must contain a key `f1` with the F1‑score.
    output_path : Path
        Destination path for the PNG file.
    """
    model_names = list(results.keys())
    f1_scores = [results[name]["f1"] for name in model_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=model_names, y=f1_scores, ax=ax)
    ax.set_ylabel("F1‑score")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison (F1‑score)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)