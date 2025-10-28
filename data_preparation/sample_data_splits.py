#!/usr/bin/env python3

"""
Creates train/val/test splits in JSONL format for ICD Chapter prediction fine-tuning.

Natural distribution with focused cleaning
- Uses merged_icd_notes.csv.gz (5 essential sections only)
- Aggressive length filtering: [5th, 90th] percentile (removes very long notes)
- Drops rare chapters: P (perinatal), U (special purposes), VWXY (external causes)
- Train: Capped-proportional sampling (15% cap, 1% floor) to prevent extreme imbalance
- Val/Test: Natural distribution (respects real-world class frequencies)
- Split by subject_id to prevent data leakage

Key differences from code-level prediction:
- NO top-K filtering (use all codes)
- NO "OTHER" token
- Focus: Predict which ICD-10 chapter the diagnosis belongs to
- Simpler task: ~15 chapters instead of 500+ codes

Output format:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<discharge_note>"},
    {"role": "assistant", "content": "{\"chapter\":\"I\"}"}
  ]
}

Note: Each note will have exactly 1 chapter (PRIMARY diagnosis only).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Set, Tuple, Dict

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default paths
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "mimic_data/csv/merged_icd_notes_focused.csv.gz")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "mimic_data/jsonl")
DEFAULT_SYSTEM_PROMPT_FILE = os.path.join(PROJECT_ROOT, "data_preparation/jsonl/system_prompt")

# Chapters to drop (too rare or not clinically relevant for this task)
CHAPTERS_TO_DROP = {'P', 'U', 'VWXY'}

# ICD-10 Chapter mapping (using chapter names without special characters)
CHAPTER_MAP = {
    'A': 'AB', 'B': 'AB',              # Infectious/parasitic
    'C': 'CD', 'D': 'CD',              # Neoplasms/blood
    'E': 'E',                           # Endocrine
    'F': 'F',                           # Mental
    'G': 'G',                           # Nervous
    'H': 'H',                           # Eye/ear
    'I': 'I',                           # Circulatory
    'J': 'J',                           # Respiratory
    'K': 'K',                           # Digestive
    'L': 'LM', 'M': 'LM',              # Skin/musculoskeletal
    'N': 'N',                           # Genitourinary
    'O': 'O',                           # Pregnancy (rare in discharge data)
    'P': 'P',                           # Perinatal (DROPPED - too rare)
    'Q': 'Q',                           # Congenital (rare)
    'R': 'R',                           # Symptoms/abnormal findings
    'S': 'ST', 'T': 'ST',              # Injury/poisoning
    'U': 'U',                           # Special purposes (DROPPED - too rare)
    'V': 'VWXY', 'W': 'VWXY',          # External causes (DROPPED - too rare)
    'X': 'VWXY', 'Y': 'VWXY',          # External causes (DROPPED - too rare)
    'Z': 'Z',                           # Status/follow-up/devices
}

CHAPTER_DESCRIPTIONS = {
    'AB': 'Infectious and parasitic diseases',
    'CD': 'Neoplasms and diseases of the blood',
    'E': 'Endocrine, nutritional and metabolic diseases',
    'F': 'Mental, behavioral and neurodevelopmental disorders',
    'G': 'Diseases of the nervous system',
    'H': 'Diseases of the eye/ear and adnexa',
    'I': 'Diseases of the circulatory system',
    'J': 'Diseases of the respiratory system',
    'K': 'Diseases of the digestive system',
    'LM': 'Diseases of the skin and musculoskeletal system',
    'N': 'Diseases of the genitourinary system',
    'O': 'Pregnancy, childbirth and the puerperium',
    'P': 'Certain conditions originating in the perinatal period (DROPPED)',
    'Q': 'Congenital malformations and chromosomal abnormalities',
    'R': 'Symptoms, signs and abnormal findings',
    'ST': 'Injury, poisoning and external consequences',
    'U': 'Codes for special purposes (DROPPED)',
    'VWXY': 'External causes of morbidity and mortality (DROPPED)',
    'Z': 'Factors influencing health status and contact with health services',
}


def load_system_prompt(filepath: str) -> str:
    """Load system prompt from external file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"System prompt file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def normalize_icd_code(code: str) -> str:
    """Remove dots from ICD codes (e.g., 'I10.0' -> 'I100')."""
    return code.replace('.', '').replace(' ', '')


def get_code_cluster(code: str) -> str:
    """Map an ICD code to its chapter."""
    first_char = code[0].upper()
    return CHAPTER_MAP.get(first_char, 'UNKNOWN')


def analyze_cluster_distribution(df: pd.DataFrame, title: str = "Chapter Distribution Analysis") -> Dict:
    """
    Analyze chapter distribution in the dataset.
    
    Args:
        df: DataFrame with 'chapter' column
        title: Title for the analysis output
    
    Returns:
        Dictionary with chapter statistics
    """
    chapter_counts = df['chapter'].value_counts()
    
    print(f"\n{title}:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Unique chapters: {len(chapter_counts)}")
    print(f"\n   Chapter breakdown:")
    
    stats_dict = {}
    for chapter in sorted(chapter_counts.index):
        count = chapter_counts[chapter]
        pct = count / len(df) * 100
        desc = CHAPTER_DESCRIPTIONS.get(chapter, 'Unknown')
        dropped_marker = " [WILL BE DROPPED]" if chapter in CHAPTERS_TO_DROP else ""
        print(f"     {chapter:10s} ({desc[:40]:40s}): {count:6,} ({pct:5.2f}%){dropped_marker}")
        stats_dict[chapter] = {
            'count': int(count),
            'percentage': float(pct),
            'description': desc
        }
    
    return stats_dict


def filter_by_length_percentile(df: pd.DataFrame, p_low: float = 5.0, p_high: float = 95.0) -> pd.DataFrame:
    """
    Filter notes to keep only those with lengths in the [p_low, p_high] percentile range.
    
    Args:
        df: DataFrame with 'cleaned_text' column
        p_low: Lower percentile (default: 5.0)
        p_high: Upper percentile (default: 95.0)
    
    Returns:
        Filtered DataFrame
    """
    # Calculate lengths, handling missing values
    df = df.copy()
    df['text_length'] = df['cleaned_text'].fillna('').astype(str).str.len()
    
    # Remove rows with zero or very small lengths
    df = df[df['text_length'] > 0].copy()
    
    if len(df) == 0:
        print(f"\nWarning: No valid text found after length calculation")
        return df
    
    low_threshold = np.percentile(df['text_length'], p_low)
    high_threshold = np.percentile(df['text_length'], p_high)
    
    print(f"\nFiltering by text length percentiles [{p_low}, {p_high}]:")
    print(f"   Original samples: {len(df):,}")
    print(f"   Length range: [{low_threshold:.0f}, {high_threshold:.0f}] characters")
    
    filtered_df = df[(df['text_length'] >= low_threshold) & (df['text_length'] <= high_threshold)].copy()
    
    print(f"   Filtered samples: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}% retained)")
    print(f"   Mean length: {filtered_df['text_length'].mean():.0f} characters")
    print(f"   Median length: {filtered_df['text_length'].median():.0f} characters")
    
    return filtered_df


def deduplicate_notes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact and near-duplicate notes by subject_id and note hash.
    
    Args:
        df: DataFrame with 'subject_id' and 'cleaned_text' columns
    
    Returns:
        Deduplicated DataFrame
    """
    print(f"\nDeduplicating notes:")
    print(f"   Original samples: {len(df):,}")
    
    # Create normalized hash
    df['note_hash'] = df['cleaned_text'].fillna('').str.lower().str.replace(r'\s+', ' ', regex=True).apply(hash)
    
    # Remove duplicates by subject and hash
    df_dedup = df.drop_duplicates(subset=['subject_id', 'note_hash']).copy()
    
    n_removed = len(df) - len(df_dedup)
    print(f"   Duplicates removed: {n_removed:,} ({n_removed/len(df)*100:.1f}%)")
    print(f"   Remaining samples: {len(df_dedup):,}")
    
    return df_dedup


def apply_capped_proportional_sampling(df: pd.DataFrame, 
                                      max_samples: int,
                                      cap_percentage: float = 15.0,
                                      floor_percentage: float = 2,
                                      seed: int = 42) -> pd.DataFrame:
    """
    Apply capped-proportional sampling to prevent extreme class imbalance in training.
    
    Strategy:
    - Compute natural chapter frequencies
    - Apply cap (default: 15%) to prevent any chapter from dominating
    - Apply floor (default: 1.2%) to ensure minimum representation (~12 samples for N=1000)
    - Redistribute excess from capped chapters to under-represented chapters
    - Use largest-remainder method to hit exact target sample count
    - Sample according to adjusted proportions
    
    Args:
        df: DataFrame with 'chapter' column
        max_samples: Maximum number of samples to select
        cap_percentage: Maximum percentage any chapter can occupy (default: 15%)
        floor_percentage: Minimum percentage each chapter should have (default: 1.2%)
        seed: Random seed
    
    Returns:
        DataFrame with capped-proportional samples (exactly max_samples)
    """
    np.random.seed(seed)
    
    print(f"Random seed: {seed}")
    
    print(f"\nApplying capped-proportional sampling (cap={cap_percentage}%, floor={floor_percentage}%):")
    
    # Calculate natural distribution
    chapter_counts = df['chapter'].value_counts()
    total_samples = len(df)
    
    print(f"   Total available samples: {total_samples:,}")
    print(f"   Target samples: {max_samples:,}")
    print(f"\n   Natural distribution:")
    
    natural_props = {}
    for chapter in sorted(chapter_counts.index):
        prop = chapter_counts[chapter] / total_samples * 100
        natural_props[chapter] = prop
        print(f"     {chapter:10s}: {chapter_counts[chapter]:6,} ({prop:5.2f}%)")
    
    # Apply caps and floors
    adjusted_props = {}
    cap_pct = cap_percentage
    floor_pct = floor_percentage
    
    print(f"\n   Applying caps and floors:")
    
    # First pass: apply cap
    total_excess = 0
    capped_chapters = []
    uncapped_chapters = []
    
    for chapter, prop in natural_props.items():
        if prop > cap_pct:
            adjusted_props[chapter] = cap_pct
            excess = prop - cap_pct
            total_excess += excess
            capped_chapters.append(chapter)
            print(f"     {chapter:10s}: {prop:5.2f}% → {cap_pct:5.2f}% (capped, excess={excess:.2f}%)")
        else:
            adjusted_props[chapter] = prop
            uncapped_chapters.append(chapter)
    
    # Second pass: apply floor and redistribute excess
    if total_excess > 0 and len(uncapped_chapters) > 0:
        print(f"\n   Redistributing {total_excess:.2f}% excess to uncapped chapters:")
        
        # Calculate how much to add to each uncapped chapter
        redistribution = total_excess / len(uncapped_chapters)
        
        for chapter in uncapped_chapters:
            old_prop = adjusted_props[chapter]
            adjusted_props[chapter] = max(floor_pct, old_prop + redistribution)
            print(f"     {chapter:10s}: {old_prop:5.2f}% → {adjusted_props[chapter]:5.2f}%")
    
    # Apply floor to all
    for chapter in adjusted_props:
        if adjusted_props[chapter] < floor_pct:
            print(f"     {chapter:10s}: {adjusted_props[chapter]:5.2f}% → {floor_pct:5.2f}% (floor applied)")
            adjusted_props[chapter] = floor_pct
    
    # Normalize to sum to 100%
    total_prop = sum(adjusted_props.values())
    for chapter in adjusted_props:
        adjusted_props[chapter] = adjusted_props[chapter] / total_prop * 100
    
    print(f"Final adjusted distribution (using largest-remainder method):")
    
    # Use largest-remainder method to allocate exact N samples
    # Step 1: Compute ideal float counts
    ideal_counts = {}
    for chapter in adjusted_props:
        ideal_counts[chapter] = max_samples * adjusted_props[chapter] / 100
    
    # Step 2: Integer parts
    integer_counts = {ch: int(ideal_counts[ch]) for ch in adjusted_props}
    
    # Step 3: Remainders
    remainders = {ch: ideal_counts[ch] - integer_counts[ch] for ch in adjusted_props}
    
    # Step 4: Distribute leftover to chapters with largest remainders
    allocated = sum(integer_counts.values())
    leftover = max_samples - allocated
    
    # Sort chapters by remainder (descending)
    sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(leftover):
        chapter = sorted_by_remainder[i][0]
        integer_counts[chapter] += 1
    
    # Verify we hit exact target
    assert sum(integer_counts.values()) == max_samples, f"Largest-remainder failed: {sum(integer_counts.values())} != {max_samples}"
    
    # Sample according to computed counts
    sampled_dfs = []
    
    for chapter in sorted(integer_counts.keys()):
        target_count = integer_counts[chapter]
        prop = adjusted_props[chapter]
        
        chapter_df = df[df['chapter'] == chapter]
        available = len(chapter_df)
        
        # Sample (with replacement if needed)
        if target_count > available:
            sampled = chapter_df.sample(n=target_count, replace=True, random_state=seed + hash(chapter) % 1000)
            print(f"     {chapter:10s}: {prop:5.2f}% → {target_count:5,} samples (oversampled from {available:,})")
        else:
            sampled = chapter_df.sample(n=target_count, replace=False, random_state=seed + hash(chapter) % 1000)
            print(f"     {chapter:10s}: {prop:5.2f}% → {target_count:5,} samples")
        
        sampled_dfs.append(sampled)
    
    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n   Total sampled: {len(result_df):,} (target: {max_samples:,}) - EXACT HIT!")
    
    # Verify final distribution
    final_counts = result_df['chapter'].value_counts()
    print(f"\n   Verification of final distribution:")
    for chapter in sorted(final_counts.index):
        count = final_counts[chapter]
        pct = count / len(result_df) * 100
        natural_pct = natural_props.get(chapter, 0)
        print(f"     {chapter:10s}: {count:5,} ({pct:5.2f}%) [natural: {natural_pct:5.2f}%]")
    
    return result_df


def split_by_subject_stratified(df: pd.DataFrame, 
                                train_ratio: float, 
                                val_ratio: float,
                                seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by subject_id with chapter stratification to prevent data leakage
    and maintain chapter distribution across splits.
    
    Args:
        df: DataFrame with 'subject_id' and 'chapter' columns
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(seed)
    
    print(f"\nSplitting by subject with chapter stratification:")
    
    # Group by subject and get their primary chapter (most common chapter for that subject)
    subject_chapters = df.groupby('subject_id')['chapter'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    
    print(f"   Total unique subjects: {len(subject_chapters):,}")
    print(f"   Total admissions: {len(df):,}")
    print(f"   Avg admissions per subject: {len(df)/len(subject_chapters):.2f}")
    
    # Split subjects by chapter to maintain distribution
    train_subjects = set()
    val_subjects = set()
    test_subjects = set()
    
    print(f"\n   Stratifying by chapter:")
    for chapter in sorted(subject_chapters.unique()):
        chapter_subjects = subject_chapters[subject_chapters == chapter].index.tolist()
        n_chapter = len(chapter_subjects)
        
        # Shuffle subjects in this chapter
        np.random.shuffle(chapter_subjects)
        
        # Split proportionally
        n_train = int(n_chapter * train_ratio)
        n_val = int(n_chapter * val_ratio)
        
        train_subjects.update(chapter_subjects[:n_train])
        val_subjects.update(chapter_subjects[n_train:n_train + n_val])
        test_subjects.update(chapter_subjects[n_train + n_val:])
        
        print(f"     {chapter:10s}: {n_chapter:5,} subjects → train={n_train:4,}, val={int(n_chapter * val_ratio):4,}, test={n_chapter - n_train - n_val:4,}")
    
    # Split dataframe by subject
    train_df = df[df['subject_id'].isin(train_subjects)].reset_index(drop=True)
    val_df = df[df['subject_id'].isin(val_subjects)].reset_index(drop=True)
    test_df = df[df['subject_id'].isin(test_subjects)].reset_index(drop=True)
    
    print(f"\n   Final split (natural distribution maintained):")
    print(f"   Train: {len(train_subjects):,} subjects → {len(train_df):,} admissions ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val:   {len(val_subjects):,} subjects → {len(val_df):,} admissions ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:  {len(test_subjects):,} subjects → {len(test_df):,} admissions ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no leakage
    assert len(train_subjects & val_subjects) == 0, "Train-Val subject overlap!"
    assert len(train_subjects & test_subjects) == 0, "Train-Test subject overlap!"
    assert len(val_subjects & test_subjects) == 0, "Val-Test subject overlap!"
    print(f"   No subject overlap between splits")
    
    return train_df, val_df, test_df


def sample_natural_distribution(df: pd.DataFrame, 
                                max_samples: int,
                                split_name: str,
                                seed: int = 42) -> pd.DataFrame:
    """
    Sample from natural distribution without artificial balancing.
    Uses stratified sampling with largest-remainder method to maintain exact proportions.
    
    Args:
        df: DataFrame with 'chapter' column
        max_samples: Maximum number of samples
        split_name: Name of split (for logging)
        seed: Random seed
    
    Returns:
        Sampled DataFrame maintaining natural distribution (exactly max_samples)
    """
    np.random.seed(seed)
    
    if max_samples >= len(df):
        print(f"\n{split_name.upper()}: Using all {len(df):,} samples (natural distribution)")
        print(f"   Random seed: {seed}")
        return df
    
    print(f"\n{split_name.upper()}: Sampling {max_samples:,} from {len(df):,} (pure proportional, natural distribution)")
    print(f"   Random seed: {seed}")
    
    # Calculate natural proportions
    chapter_counts = df['chapter'].value_counts()
    
    print(f"\n   Original distribution:")
    for chapter in sorted(chapter_counts.index):
        pct = chapter_counts[chapter] / len(df) * 100
        print(f"     {chapter:10s}: {chapter_counts[chapter]:6,} ({pct:5.2f}%)")
    
    # Use largest-remainder method for exact proportional allocation
    # Step 1: Compute ideal float counts based on natural proportions
    ideal_counts = {}
    for chapter in chapter_counts.index:
        natural_prop = chapter_counts[chapter] / len(df)
        ideal_counts[chapter] = max_samples * natural_prop
    
    # Step 2: Integer parts
    integer_counts = {ch: int(ideal_counts[ch]) for ch in ideal_counts}
    
    # Step 3: Remainders
    remainders = {ch: ideal_counts[ch] - integer_counts[ch] for ch in ideal_counts}
    
    # Step 4: Distribute leftover to chapters with largest remainders
    allocated = sum(integer_counts.values())
    leftover = max_samples - allocated
    
    # Sort chapters by remainder (descending)
    sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(leftover):
        chapter = sorted_by_remainder[i][0]
        integer_counts[chapter] += 1
    
    # Verify we hit exact target
    assert sum(integer_counts.values()) == max_samples, f"Largest-remainder failed: {sum(integer_counts.values())} != {max_samples}"
    
    # Stratified sampling by chapter
    sampled_dfs = []
    
    print(f"\n   Sampled distribution (stratified with largest-remainder):")
    for chapter in sorted(integer_counts.keys()):
        n_samples = integer_counts[chapter]
        
        if n_samples == 0:
            continue
            
        chapter_df = df[df['chapter'] == chapter]
        available = len(chapter_df)
        
        # Sample without replacement (or with if needed)
        if n_samples > available:
            sampled = chapter_df.sample(n=n_samples, replace=True, random_state=seed + hash(chapter) % 1000)
        else:
            sampled = chapter_df.sample(n=n_samples, replace=False, random_state=seed + hash(chapter) % 1000)
        
        sampled_dfs.append(sampled)
        
        pct = n_samples / max_samples * 100
        orig_pct = chapter_counts[chapter] / len(df) * 100
        print(f"     {chapter:10s}: {n_samples:6,} ({pct:5.2f}%) [natural: {orig_pct:5.2f}%]")
    
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    print(f"\n   Total sampled: {len(sampled_df):,} (target: {max_samples:,})")
    
    return sampled_df


def create_jsonl_entry(cleaned_text: str, chapter: str, system_prompt: str) -> dict:
    """
    Create a single JSONL entry with chapter only (no ICD code).
    
    Args:
        cleaned_text: Cleaned discharge note
        chapter: Single chapter label (e.g., "I")
        system_prompt: System prompt text
    
    Returns:
        Dictionary in the required format with single chapter
    """
    # Create assistant response with single chapter only
    assistant_content = json.dumps({"chapter": chapter})
    
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": cleaned_text
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }


def save_cluster_metadata(chapter_stats: Dict, output_dir: str, dropped_chapters: Set[str], 
                         cap_pct: float = 15.0, floor_pct: float = 1.2):
    """
    Save chapter distribution metadata for reference.
    
    Args:
        chapter_stats: Dictionary with chapter statistics
        output_dir: Output directory
        dropped_chapters: Set of dropped chapter names
        cap_pct: Cap percentage used for training
        floor_pct: Floor percentage used for training
    """
    metadata = {
        "task": "chapter_prediction_only",
        "scope": "PRIMARY diagnosis chapter (seq_num=1)",
        "approach": "Natural distribution with capped-proportional train sampling",
        "dropped_chapters": list(dropped_chapters),
        "num_chapters": len(chapter_stats),
        "chapters": chapter_stats,
        "chapter_descriptions": {k: v for k, v in CHAPTER_DESCRIPTIONS.items() if k not in dropped_chapters},
        "sampling_strategy": {
            "train": f"Capped-proportional ({cap_pct}% cap, {floor_pct}% floor) with largest-remainder",
            "val": "Pure proportional (natural distribution) with largest-remainder",
            "test": "Pure proportional (natural distribution) with largest-remainder"
        },
        "random_seed": 42,
        "split_method": "Subject-wise (no data leakage)",
        "rounding_method": "Largest-remainder (exact sample counts)"
    }
    
    metadata_path = os.path.join(output_dir, "chapter_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   Saved chapter metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                        help="Path to merged CSV file")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for JSONL files")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT_FILE,
                        help="Path to system prompt text file")
    parser.add_argument("--train-ratio", type=float, default=0.80,
                        help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10,
                        help="Test set ratio")
    parser.add_argument("--max-train-samples", type=int, default=2000,
                        help="Maximum number of training samples")
    parser.add_argument("--max-val-samples", type=int, default=250,
                        help="Maximum number of validation samples")
    parser.add_argument("--max-test-samples", type=int, default=250,
                        help="Maximum number of test samples")
    parser.add_argument("--cap-percentage", type=float, default=15.0,
                        help="Cap percentage for train sampling")
    parser.add_argument("--floor-percentage", type=float, default=2,
                        help="Floor percentage for train sampling")
    parser.add_argument("--percentile-low", type=float, default=5.0,
                        help="Lower percentile for text length filtering")
    parser.add_argument("--percentile-high", type=float, default=90.0,
                        help="Upper percentile for text length filtering")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"Error: Split ratios must sum to 1.0 (got {total_ratio})")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Creating Chapter-Only Splits")
    print("=" * 80)
    print(f"\nRandom seed: {args.seed} (reproducible sampling)")
    print(f"\nStrategy:")
    print(f"   • Drops rare chapters: {', '.join(sorted(CHAPTERS_TO_DROP))}")
    print(f"   • Train: Capped-proportional ({args.cap_percentage}% cap, {args.floor_percentage}% floor)")
    print(f"   • Val/Test: Pure proportional (no caps/floors, maintains natural distribution)")
    print(f"   • Split by subject_id (prevents data leakage)")
    print(f"   • Largest-remainder method (ensures exact sample counts)")
    
    # Load system prompt
    print("\nLoading system prompt...")
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"   System prompt loaded ({len(system_prompt)} characters)")
    
    # Load dataset
    print(f"\nLoading dataset from: {args.input}")
    df = pd.read_csv(args.input, compression='gzip')
    print(f"   Loaded {len(df):,} records")
    
    # Deduplicate notes
    df = deduplicate_notes(df)
    
    # Parse ICD codes
    print("\nParsing ICD codes...")
    import ast
    df['icd_codes_parsed'] = df['icd_codes'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    
    # Verify we only have PRIMARY diagnosis (seq_num=1)
    print("   Verifying PRIMARY-only diagnoses...")
    df['seq_nums_parsed'] = df['seq_nums'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    
    all_seq_nums = []
    for seq_list in df['seq_nums_parsed']:
        all_seq_nums.extend(seq_list)
    unique_seq = set(all_seq_nums)
    if unique_seq != {1}:
        print(f"    WARNING: Found unexpected seq_nums: {unique_seq}. Expected only [1]")
    else:
        print("     All rows contain PRIMARY diagnosis only (seq_num=1)")
    
    # Verify all notes have exactly 1 code
    code_counts = df['icd_codes_parsed'].apply(len)
    if (code_counts != 1).any():
        multi_code = (code_counts > 1).sum()
        print(f"    WARNING: {multi_code} rows have multiple codes (expected exactly 1)")
    else:
        print(f"   All {len(df):,} rows have exactly 1 code (PRIMARY only)")
    
    # Map codes to chapters
    print(f"\nMapping codes to chapters...")
    df['single_code'] = df['icd_codes_parsed'].apply(lambda codes: codes[0] if len(codes) > 0 else None)
    df['chapter'] = df['single_code'].apply(get_code_cluster)
    
    # Remove any UNKNOWN chapters
    unknown_count = (df['chapter'] == 'UNKNOWN').sum()
    if unknown_count > 0:
        print(f"   Removing {unknown_count:,} records with UNKNOWN chapter")
        df = df[df['chapter'] != 'UNKNOWN'].copy()
    
    # DROP rare chapters
    print(f"\nDropping rare chapters: {', '.join(sorted(CHAPTERS_TO_DROP))}")
    df_before_drop = df.copy()
    df = df[~df['chapter'].isin(CHAPTERS_TO_DROP)].copy()
    
    dropped_count = len(df_before_drop) - len(df)
    print(f"   Dropped {dropped_count:,} samples ({dropped_count/len(df_before_drop)*100:.2f}%)")
    print(f"   Remaining: {len(df):,} samples")
    
    # Analyze chapter distribution AFTER dropping
    chapter_stats_after = analyze_cluster_distribution(df, "Chapter Distribution (AFTER dropping rare chapters)")
    
    # Apply percentile filtering
    df = filter_by_length_percentile(df, args.percentile_low, args.percentile_high)
    
    # Split by subject with chapter stratification
    print(f"\nSplitting dataset by subject:")
    print(f"   Train: {args.train_ratio*100:.0f}%")
    print(f"   Val:   {args.val_ratio*100:.0f}%")
    print(f"   Test:  {args.test_ratio*100:.0f}%")
    
    train_df, val_df, test_df = split_by_subject_stratified(
        df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    # Apply sampling strategies
    print(f"\n" + "="*80)
    print("APPLYING SAMPLING STRATEGIES")
    print("="*80)
    
    # TRAIN: Capped-proportional sampling
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = apply_capped_proportional_sampling(
            train_df, 
            max_samples=args.max_train_samples,
            cap_percentage=args.cap_percentage,
            floor_percentage=args.floor_percentage,
            seed=args.seed
        )
    else:
        print(f"\nTRAIN: Using all {len(train_df):,} samples (no max limit)")
        analyze_cluster_distribution(train_df, "Train Distribution (natural)")
    
    # VAL: Natural distribution sampling
    if args.max_val_samples and args.max_val_samples < len(val_df):
        val_df = sample_natural_distribution(
            val_df,
            max_samples=args.max_val_samples,
            split_name='val',
            seed=args.seed + 1
        )
    else:
        print(f"\nVAL: Using all {len(val_df):,} samples (natural distribution)")
        analyze_cluster_distribution(val_df, "Val Distribution (natural)")
    
    # TEST: Natural distribution sampling
    if args.max_test_samples and args.max_test_samples < len(test_df):
        test_df = sample_natural_distribution(
            test_df,
            max_samples=args.max_test_samples,
            split_name='test',
            seed=args.seed + 2
        )
    else:
        print(f"\nTEST: Using all {len(test_df):,} samples (natural distribution)")
        analyze_cluster_distribution(test_df, "Test Distribution (natural)")
    
    # Create JSONL files
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    print(f"\n" + "="*80)
    print(f"Writing JSONL files...")
    for split_name, split_df in splits.items():
        output_path = os.path.join(args.output_dir, f"{split_name}.jsonl")

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in split_df.iterrows():
                entry = create_jsonl_entry(
                    cleaned_text=row['cleaned_text'],
                    chapter=row['chapter'],
                    system_prompt=system_prompt
                )
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"   {split_name}.jsonl: {len(split_df):,} samples ({file_size_mb:.2f} MB)")
    
    # Save chapter metadata
    print(f"\nSaving chapter metadata...")
    save_cluster_metadata(chapter_stats_after, args.output_dir, CHAPTERS_TO_DROP, 
                         args.cap_percentage, args.floor_percentage)
    
    print("\n" + "=" * 80)
    print("Dataset sampled splits created successfully!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - train.jsonl ({len(train_df):,} samples, capped-proportional)")
    print(f"  - val.jsonl ({len(val_df):,} samples, natural)")
    print(f"  - test.jsonl ({len(test_df):,} samples, natural)")
    print(f"  - chapter_metadata.json")


if __name__ == "__main__":
    main()
