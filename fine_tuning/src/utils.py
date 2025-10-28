"""
Utility functions for config loading, logging, and W&B integration.
"""

import os
import logging
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def verify_environment_variables() -> None:
    """
    Verify required environment variables are set.
    """
    logger = logging.getLogger(__name__)
    
    # Check for required API keys
    if not os.environ.get("MISTRAL_API_KEY"):
        logger.error("MISTRAL_API_KEY not found in environment!")
        logger.error("Please set it: export MISTRAL_API_KEY=your-key")
        raise ValueError("MISTRAL_API_KEY environment variable is required")
    
    logger.info("MISTRAL_API_KEY found in environment")
    
    # W&B is optional, just log if missing
    if not os.environ.get("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not found (W&B will prompt for login if enabled)")


def setup_logging(level: int = logging.INFO) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to config.yaml.
    
    Returns:
        Configuration dictionary.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading config from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Config loaded: {config.get('project_name', 'unknown')}")
    return config


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting random seed: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)


def save_predictions(
    predictions: list,
    output_path: str,
) -> None:
    """
    Save predictions to a file.
    
    Args:
        predictions: List of predictions.
        output_path: Output file path.
    """
    logger = logging.getLogger(__name__)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(str(pred) + "\n")
    
    logger.info(f"Predictions saved to: {output_path}")


def get_absolute_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Convert relative path to absolute path.
    
    Args:
        path: File path (relative or absolute).
        base_dir: Base directory for relative paths.
    
    Returns:
        Absolute path as string.
    """
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        return str(path_obj)
    
    if base_dir:
        return str(Path(base_dir) / path_obj)
    
    return str(path_obj.resolve())


def load_class_names(config: Dict[str, Any], base_dir: Path) -> Dict[str, str]:
    """
    Load chapter metadata for class names if configured.
    
    Args:
        config: Configuration dictionary.
        base_dir: Base directory for resolving paths.
    
    Returns:
        Dictionary mapping chapter codes to names, or empty dict if not configured.
    """
    from datasets import load_chapter_metadata
    
    if not config.get("class_names"):
        return {}
    
    metadata_path = get_absolute_path(config["class_names"], str(base_dir))
    return load_chapter_metadata(metadata_path)


def get_sorted_labels(y_true: List[str], y_pred: List[str]) -> List[str]:
    """
    Get labels sorted by support (most common first).
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
    
    Returns:
        Sorted list of unique labels.
    """
    label_counts = Counter(y_true)
    sorted_labels = [label for label, _ in label_counts.most_common()]
    
    # Add any labels that appear only in predictions
    for label in sorted(set(y_pred) - set(y_true)):
        sorted_labels.append(label)
    
    return sorted_labels


def save_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    output_dir: Path,
) -> None:
    """
    Save confusion matrix as both CSV and PNG heatmap.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_dir: Directory to save results.
    """
    logger = logging.getLogger(__name__)
    
    sorted_labels = get_sorted_labels(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=sorted_labels)
    
    # Save as CSV
    cm_df = pd.DataFrame(cm, index=sorted_labels, columns=sorted_labels)
    cm_csv_file = output_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv_file)
    logger.info(f"Confusion matrix (CSV) saved to: {cm_csv_file}")
    
    # Save as PNG heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=sorted_labels,
        yticklabels=sorted_labels,
        cbar_kws={'label': 'Count'},
        square=True,
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    
    cm_png_file = output_dir / "confusion_matrix.png"
    plt.savefig(cm_png_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix (PNG) saved to: {cm_png_file}")


def save_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    per_class_metrics: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
    base_dir: Path,
    output_dir: Path,
) -> None:
    """
    Save per-class metrics as CSV.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        per_class_metrics: Per-class metrics dictionary.
        config: Configuration dictionary.
        base_dir: Base directory for resolving paths.
        output_dir: Directory to save results.
    """
    logger = logging.getLogger(__name__)
    
    all_labels = sorted(set(y_true) | set(y_pred))
    class_names_dict = load_class_names(config, base_dir)
    
    per_class_data = []
    for label in all_labels:
        metrics = per_class_metrics.get(label, {})
        class_name = class_names_dict.get(label, label) if class_names_dict else label
        per_class_data.append({
            "chapter_code": label,
            "chapter_name": class_name,
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "f1": metrics.get("f1", 0.0),
            "support": metrics.get("support", 0),
        })
    
    per_class_df = pd.DataFrame(per_class_data)
    per_class_csv_file = output_dir / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_file, index=False)
    logger.info(f"Per-class metrics saved to: {per_class_csv_file}")


def create_output_directory(config: Dict[str, Any], base_dir: Path, model_id: str) -> Path:
    """
    Create timestamped output directory for evaluation results.
    
    Args:
        config: Configuration dictionary.
        base_dir: Base directory for resolving paths.
        model_id: Model ID being evaluated.
    
    Returns:
        Path to created output directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_id.split("/")[-1].replace(":", "_")
    output_dir_name = f"{timestamp}_{model_name}"
    
    base_output_dir = Path(config.get("output_dir", "outputs"))
    if not base_output_dir.is_absolute():
        base_output_dir = base_dir / base_output_dir
    
    output_dir = base_output_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def extract_user_messages(data: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """
    Extract user messages (without assistant responses) for inference.
    
    Args:
        data: List of examples in chat format.
    
    Returns:
        List of message sequences (user messages only).
    """
    messages_list = []
    
    for example in data:
        messages = example.get("messages", [])
        
        # Keep only user and system messages (remove assistant)
        user_messages = [
            msg for msg in messages
            if msg.get("role") in ["system", "user"]
        ]
        
        messages_list.append(user_messages)
    
    return messages_list