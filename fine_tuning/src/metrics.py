"""
Evaluation metrics for ICD chapter classification.

Includes accuracy, macro-F1, weighted-F1, MCC, and confusion matrix utilities.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of all possible labels (for consistent ordering).
        class_names: Optional mapping from labels to human-readable names.
    
    Returns:
        Dictionary of metrics.
    """
    
    # Handle empty predictions
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} true vs {len(y_pred)} pred")
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    
    return metrics


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of all possible labels (for consistent ordering).
    
    Returns:
        Confusion matrix as numpy array.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return cm


def get_classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    class_names: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of all possible labels.
        class_names: Optional mapping from labels to human-readable names.
    
    Returns:
        Classification report as string.
    """
    # Create target names for report
    target_names = None
    if class_names and labels:
        target_names = [class_names.get(label, label) for label in labels]
    
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    
    return report


def compute_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of all possible labels.
    
    Returns:
        Dictionary mapping each class to its metrics.
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    if labels is None:
        labels = sorted(set(y_true))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    
    per_class = {}
    for i, label in enumerate(labels):
        per_class[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    
    return per_class


def analyze_errors(
    y_true: List[str],
    y_pred: List[str],
    examples: List[Dict[str, Any]],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Analyze misclassified examples.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        examples: Original examples (for context).
        top_n: Number of errors to return.
    
    Returns:
        List of error analysis dictionaries.
    """
    errors = []
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            error = {
                "index": i,
                "true_label": true,
                "pred_label": pred,
                "example": examples[i] if i < len(examples) else None,
            }
            errors.append(error)
    
    # Return top N errors
    return errors[:top_n]
