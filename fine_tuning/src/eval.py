"""
Evaluation script for fine-tuned Mistral models.

This script:
1. Loads the fine-tuned model
2. Runs inference on test set
3. Computes all evaluation metrics
4. Saves results to disk and displays in terminal
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

from mistral_api import MistralClient
from datasets import load_jsonl, extract_labels_from_messages
from metrics import (
    compute_metrics,
    compute_confusion_matrix,
    get_classification_report,
    compute_per_class_metrics,
    analyze_errors,
)
from utils import (
    setup_logging,
    verify_environment_variables,
    load_config,
    get_absolute_path,
    save_predictions,
    load_class_names,
    save_confusion_matrix,
    save_per_class_metrics,
    create_output_directory,
    extract_user_messages,
)

logger = logging.getLogger(__name__)

def run_evaluation(
    config: Dict[str, Any],
    model_id: str,
    test_path: str,
) -> Dict[str, Any]:
    """
    Run evaluation on test set.
    
    Args:
        config: Configuration dictionary.
        model_id: Fine-tuned model ID.
        test_path: Path to test JSONL.
    
    Returns:
        Evaluation results dictionary.
    """
    logger.info("=== Running Evaluation ===")
    
    test_data = load_jsonl(test_path)
    
    logger.info("Extracting labels from test data messages...")
    y_true = extract_labels_from_messages(test_data)
    logger.info(f"Extracted {len(y_true)} labels")
    
    messages_list = extract_user_messages(test_data)
    
    client = MistralClient()

    logger.info("Running inference on test set with JSON output format...")
    responses = client.inference(
        model=model_id,
        messages_list=messages_list,
        max_tokens=config.get("max_tokens", 512),
        temperature=0.0,
        top_p=1.0,
        response_format={"type": "json_object"}
    )
    
    # Parse predictions from JSON responses
    y_pred = [json.loads(resp)["chapter"] for resp in responses]
    
    # Get unique labels for consistent ordering
    all_labels = sorted(set(y_true) | set(y_pred))
    
    logger.info("Computing metrics...")
    metrics = compute_metrics(y_true, y_pred, labels=all_labels)
    
    cm = compute_confusion_matrix(y_true, y_pred, labels=all_labels)
    
    base_dir = Path(__file__).parent.parent
    class_names_dict = load_class_names(config, base_dir)
    
    report = get_classification_report(
        y_true, 
        y_pred, 
        labels=all_labels, 
        class_names=class_names_dict if class_names_dict else None
    )
    
    per_class_metrics = compute_per_class_metrics(y_true, y_pred, labels=all_labels)
    
    errors = analyze_errors(y_true, y_pred, test_data, top_n=20)
    
    results = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_metrics": per_class_metrics,
        "errors": errors,
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "responses": responses,
        },
    }
    
    return results

def save_results_to_disk(
    results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save evaluation results to disk (metrics, confusion matrix, per-class metrics).
    
    Args:
        results: Evaluation results dictionary.
        config: Configuration dictionary.
        output_dir: Directory to save results.
    """
    logger.info(f"Saving results to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_true = results["predictions"]["y_true"]
    y_pred = results["predictions"]["y_pred"]
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    logger.info(f"Metrics saved to: {metrics_file}")
    
    save_confusion_matrix(y_true, y_pred, output_dir)
    
    base_dir = Path(__file__).parent.parent
    save_per_class_metrics(y_true, y_pred, results["per_class_metrics"], config, base_dir, output_dir)
    
    report_file = output_dir / "classification_report.txt"
    with open(report_file, "w") as f:
        f.write(results["classification_report"])
    logger.info(f"Classification report saved to: {report_file}")
    
    predictions_file = output_dir / "predictions.txt"
    save_predictions(results["predictions"]["y_pred"], str(predictions_file))
    logger.info(f"Predictions saved to: {predictions_file}")
    
    logger.info(f"\nAll results saved successfully to: {output_dir}")


def print_evaluation_summary(results: Dict[str, Any], model_id: str) -> None:
    """
    Print evaluation summary to terminal.
    
    Args:
        results: Evaluation results dictionary.
        config: Configuration dictionary.
        model_id: Model ID evaluated.
    """
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {model_id}")
    logger.info(f"Test Examples: {len(results['predictions']['y_true'])}")
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:    {results['metrics']['accuracy']:.4f}")
    logger.info(f"  Macro-F1:    {results['metrics']['macro_f1']:.4f}")
    logger.info(f"  Weighted-F1: {results['metrics']['weighted_f1']:.4f}")
    logger.info(f"  MCC:         {results['metrics']['mcc']:.4f}")


def main():
    """Main evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Mistral model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--model-id", type=str, required=True, help="Fine-tuned model ID to evaluate")
    parser.add_argument("--test-path", type=str, default=None, help="Path to test data (overrides config)")
    args = parser.parse_args()
    

    setup_logging()
    logger.info("=== Mistral Model Evaluation ===")
    verify_environment_variables()
    

    base_dir = Path(__file__).parent.parent
    config = load_config(str(base_dir / args.config))
    

    output_dir = create_output_directory(config, base_dir, args.model_id)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        
        test_path = args.test_path or config["test_path"]
        test_path = get_absolute_path(test_path, str(base_dir))
        
        results = run_evaluation(config, args.model_id, test_path)
        save_results_to_disk(results, config, output_dir)
        print_evaluation_summary(results, args.model_id)
        
        return 0
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
