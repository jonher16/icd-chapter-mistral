"""
Dataset utilities for loading and preprocessing JSONL data.

This module handles loading training, validation, and test data
in JSONL format for fine-tuning and evaluation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
    
    Returns:
        List of dictionaries, one per line.
    """
    logger.info(f"Loading JSONL from: {file_path}")
    
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {line_num}: {e}")
                raise
    
    logger.info(f"Loaded {len(data)} examples from {file_path}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save.
        file_path: Output file path.
    """
    logger.info(f"Saving {len(data)} examples to: {file_path}")
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved to {file_path}")


def extract_labels_from_messages(
    data: List[Dict[str, Any]],
    messages_key: str = "messages",
) -> List[str]:
    """
    Extract labels from assistant messages in chat format.
    
    For ICD chapter prediction, the label is in the assistant's response.
    
    Args:
        data: List of examples in chat format.
        messages_key: Key containing the messages list.
    
    Returns:
        List of extracted labels.
    """
    labels = []
    
    for example in data:
        messages = example.get(messages_key, [])
        
        # Find the assistant message (should be the last one)
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
        
        if assistant_msg:
            import json
            
            assistant_msg = assistant_msg.strip()
            
            # Try to parse as JSON first
            try:
                if assistant_msg.startswith("{"):
                    data = json.loads(assistant_msg)
                    if "chapter" in data:
                        labels.append(data["chapter"])
                        continue
            except:
                pass
            
            # Fallback to text parsing
            label = assistant_msg
            if ":" in label:
                label = label.split(":")[-1].strip()
            labels.append(label)
        else:
            logger.warning(f"No assistant message found in example")
            labels.append("")
    
    return labels


def validate_chat_format(data: List[Dict[str, Any]], messages_key: str = "messages") -> bool:
    """
    Validate that data is in correct chat completion format.
    
    Args:
        data: List of examples to validate.
        messages_key: Key containing the messages list.
    
    Returns:
        True if valid, raises ValueError otherwise.
    """
    logger.info(f"Validating {len(data)} examples")
    
    for i, example in enumerate(data):
        if messages_key not in example:
            raise ValueError(f"Example {i} missing '{messages_key}' key")
        
        messages = example[messages_key]
        if not isinstance(messages, list):
            raise ValueError(f"Example {i}: messages must be a list")
        
        if len(messages) < 2:
            raise ValueError(f"Example {i}: must have at least 2 messages (user + assistant)")
        
        # Check roles
        roles = [msg.get("role") for msg in messages]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError(f"Example {i}: must have both 'user' and 'assistant' roles")
        
        # Validate message structure
        for j, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Example {i}, message {j}: must have 'role' and 'content'")
    
    logger.info("Validation passed!")
    return True


def count_tokens_estimate(
    data: List[Dict[str, Any]],
    messages_key: str = "messages",
    avg_chars_per_token: float = 4.0,
) -> int:
    """
    Estimate total tokens in dataset.
    
    This is a rough estimate using character count.
    
    Args:
        data: List of examples.
        messages_key: Key containing messages.
        avg_chars_per_token: Average characters per token (default: 4.0).
    
    Returns:
        Estimated total tokens.
    """
    total_chars = 0
    
    for example in data:
        messages = example.get(messages_key, [])
        for msg in messages:
            content = msg.get("content", "")
            total_chars += len(content)
    
    estimated_tokens = int(total_chars / avg_chars_per_token)
    logger.info(f"Estimated tokens: {estimated_tokens:,}")
    
    return estimated_tokens


def get_dataset_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the dataset.
    
    Args:
        data: List of examples.
    
    Returns:
        Dictionary with dataset statistics.
    """
    labels = extract_labels_from_messages(data)
    
    # Count label distribution
    from collections import Counter
    label_counts = Counter(labels)
    
    stats = {
        "num_examples": len(data),
        "num_unique_labels": len(label_counts),
        "label_distribution": dict(label_counts),
        "estimated_tokens": count_tokens_estimate(data),
    }
    
    logger.info(f"Dataset stats: {stats['num_examples']} examples, "
                f"{stats['num_unique_labels']} unique labels")
    
    return stats


def load_chapter_metadata(metadata_path: str) -> Dict[str, str]:
    """
    Load ICD-10 chapter metadata (code ranges to chapter names).
    
    Args:
        metadata_path: Path to chapter_metadata.json.
    
    Returns:
        Dictionary mapping chapter codes to names.
    """
    logger.info(f"Loading chapter metadata from: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    # Create a mapping from chapter code to description
    chapter_map = {}
    if "chapters" in metadata:
        for code, info in metadata["chapters"].items():
            description = info.get("description", code)
            chapter_map[code] = description
    
    logger.info(f"Loaded {len(chapter_map)} chapters")
    return chapter_map
