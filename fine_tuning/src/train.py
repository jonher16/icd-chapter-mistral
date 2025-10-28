"""
Training script for fine-tuning Mistral models via API.

This script orchestrates:
1. Data preparation and upload
2. Fine-tuning job creation and monitoring
3. W&B logging of training metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

from mistral_api import MistralClient
from datasets import load_jsonl, validate_chat_format, get_dataset_stats
from utils import (
    setup_logging,
    verify_environment_variables,
    load_config,
    set_seed,
    get_absolute_path,
)

logger = logging.getLogger(__name__)


def prepare_data(config: Dict[str, Any], base_dir: str) -> tuple[str, str]:
    """
    Load and validate training and validation data.
    
    Args:
        config: Configuration dictionary.
        base_dir: Base directory for relative paths.
    
    Returns:
        Tuple of (train_path, val_path).
    """
    logger.info("=== Preparing Data ===")
    
    train_path = get_absolute_path(config["train_path"], base_dir)
    val_path = get_absolute_path(config["val_path"], base_dir)
    
    train_data = load_jsonl(train_path)
    validate_chat_format(train_data, config.get("text_key", "messages"))
    
    val_data = load_jsonl(val_path)
    validate_chat_format(val_data, config.get("text_key", "messages"))
    
    train_stats = get_dataset_stats(train_data)
    val_stats = get_dataset_stats(val_data)
    
    logger.info(f"Train: {train_stats['num_examples']} examples, "
                f"{train_stats['estimated_tokens']:,} tokens")
    logger.info(f"Val: {val_stats['num_examples']} examples, "
                f"{val_stats['estimated_tokens']:,} tokens")
    

def run_training(config: Dict[str, Any], train_path: str, val_path: str) -> Dict[str, Any]:
    """
    Run fine-tuning via Mistral API with WandB integration.
    
    Args:
        config: Configuration dictionary.
        train_path: Path to training JSONL.
        val_path: Path to validation JSONL.
    
    Returns:
        Job status dictionary.
    """
    logger.info("=== Starting Fine-Tuning ===")
    
    client = MistralClient(wandb_config=config.get("wandb"))
    
    logger.info("Uploading training data...")
    train_file_id = client.upload_file(train_path, purpose="fine-tune")
    
    logger.info("Uploading validation data...")
    val_file_id = client.upload_file(val_path, purpose="fine-tune")
    
    train_config = config["train"]
    hyperparameters = {
        "epochs": train_config.get("epochs"),
        "training_steps": train_config.get("train_steps"),
        "learning_rate": train_config.get("learning_rate", 1e-5),
        "weight_decay": train_config.get("weight_decay", 0.1),
        "warmup_fraction": train_config.get("warmup_fraction", 0.05),
        "gradient_clip_norm": train_config.get("gradient_clip_norm"),
    }
    
    logger.info(f"Creating fine-tuning job (auto_start disabled)...")
    
    client.create_fine_tuning_job(
        training_file_id=train_file_id,
        validation_file_id=val_file_id,
        model=config["model"],
        hyperparameters=hyperparameters,
        suffix=config.get("run_name"),
    )

def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info("=== Mistral Fine-Tuning ===")
    
    # Get base directory (experiment root)
    base_dir = Path(__file__).parent.parent
    
    verify_environment_variables()
    
    config_path = base_dir / args.config
    
    config = load_config(str(config_path))

    set_seed(config.get("seed", 42))
    
    try:
        train_path, val_path = prepare_data(config, str(base_dir))
        run_training(config, train_path, val_path)
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
