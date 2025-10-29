# ICD Chapter Prediction with Mistral Fine-Tuning API

## Table of Contents

- [Overview](#overview)
  - [What is ICD?](#what-is-icd)
  - [Project Objective](#project-objective)
- [Project Structure](#project-structure)
- [Fine Tuning Results](#fine-tuning-results)
- [Setup](#setup)
  - [Install Dependencies](#install-dependencies)
  - [Download MIMIC-IV Data](#download-mimic-iv-data)
- [Data Preparation](#data-preparation)
  - [Build the Dataset](#build-the-dataset)
  - [Explore the Dataset](#explore-the-dataset-optional)
  - [Create Data Splits](#create-data-splits)
- [Fine-Tuning](#fine-tuning)
  - [Configuration](#configuration)
  - [Run Training](#run-training)
  - [Run Evaluation](#run-evaluation)
- [Quick Start](#quick-start)

---

## Overview

### What is ICD?

ICD (International Classification of Diseases) is a medical classification system maintained by the WHO for coding diagnoses, symptoms, and procedures. ICD-10 codes are organized into chapters (A-Z), each representing a category of diseases or conditions (e.g., Chapter I = Circulatory diseases, Chapter J = Respiratory diseases).

### Project Objective

This repository fine-tunes Mistral language models to predict ICD-10 chapters from clinical discharge notes. The model learns to classify medical text into diagnostic categories based on primary diagnoses from the MIMIC-IV dataset.

### Fine-Tuning Results

#### Training Settings

| Setting                | Value                                             |
| ---------------------- | ------------------------------------------------- |
| Model                  | ministral-8b-latest                               |
| Dataset                | 24,000 discharge notes (MIMIC IV)                 |
| Total Tokens Processed | 25.23M                                            |
| Epochs / Steps         | 1 epoch / 98 steps                                |
| Learning Rate          | 1.5e-5                                            |
| Weight Decay           | 0.05                                              |
| Warmup Fraction        | 0.08                                              |
| Gradient Clip Norm     | 1.0                                               |
| Cost                   | **€22.71**                                        |

#### Test Evaluation Results

| Metric      | ministral-8b-latest (Base) | ministral-8b-latest (FT) |     Δ Gain |
| ----------- | -------------------------: | -----------------------: | ---------: |
| Accuracy    |                      0.660 |                **0.788** | **+19.4%** |
| Macro-F1    |                      0.493 |                **0.713** | **+44.6%** |
| Weighted-F1 |                      0.654 |                **0.782** | **+19.6%** |
| MCC         |                      0.619 |                **0.762** | **+23.1%** |

**Model ID:**  
`ft:ministral-8b-latest:ebea8cff:20251028:mini8b-icd-chap:9dd1cf6a`

## Project Structure

```
icd-chapter-mistral/
├── data_preparation/          # Data processing scripts
│   ├── build_dataset.py       # Merge and clean raw data
│   ├── sample_data_splits.py  # Create train/val/test splits
│   ├── explore_dataset.ipynb  # Data exploration notebook
│   └── note_cleaner.py        # Note cleaning utilities
├── fine_tuning/
│   ├── configs/
│   │   └── config.yaml        # Training configuration
│   ├── src/
│   │   ├── train.py           # Training script
│   │   ├── eval.py            # Evaluation script
│   │   ├── mistral_api.py     # Mistral API client
│   │   ├── datasets.py        # Data loading utilities
│   │   └── metrics.py         # Evaluation metrics
│   ├── data/                  # Default location for JSONL data
│   └── outputs/               # Evaluation results
└── mimic_data/
    ├── raw/                   # Raw MIMIC-IV files
    ├── csv/                   # Processed CSV files
    └── jsonl/                 # Train/val/test splits
```

---

## Setup

### Install Dependencies

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable Python dependency management.

#### 1. Install UV

**On macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Install Project Dependencies

```bash
# Install all dependencies from pyproject.toml
uv sync
```

### Download MIMIC-IV Data

There is a `test.json` file in the fine_tuning/data/ directory for quick testing, but to train on real data, follow these steps:

You need to download the MIMIC-IV public dataset from PhysioNet.

**Dataset URL:** https://physionet.org/content/mimiciv/3.1/#files-panel

Download these three files:
- `diagnoses_icd.csv.gz` - ICD codes corresponding to discharge notes
- `d_icd_diagnoses.csv.gz` - ICD code labels and descriptions
- `discharge.csv.gz` - Clinical discharge notes

**Store all files in:** `mimic_data/raw/`

```bash
# Your directory structure should look like:
mimic_data/
├── raw/
│   ├── diagnoses_icd.csv.gz
│   ├── d_icd_diagnoses.csv.gz
│   └── discharge.csv.gz
```

---

## Data Preparation

### Build the Dataset

Run `build_dataset.py` to merge and clean the raw data:

```bash
python data_preparation/build_dataset.py
```

**What it does:**
- Merges discharge notes with ICD diagnosis codes
- Keeps only PRIMARY diagnoses (seq_num = 1)
- Cleans notes to focus on 5 key sections:
  - Discharge Diagnosis
  - Chief Complaint
  - History of Present Illness
  - Hospital Course
  - Procedures

**Output:** `mimic_data/csv/merged_icd_notes.csv.gz`

### Explore the Dataset (Optional)

Use the Jupyter notebook to analyze the prepared data:

```bash
jupyter notebook data_preparation/explore_dataset.ipynb
```

This notebook provides visualizations and statistics about:
- Chapter distribution
- Note lengths
- Data quality metrics

### Create Data Splits

Generate train/validation/test splits in JSONL format:

```bash
python data_preparation/sample_data_splits.py [OPTIONS]
```

#### Key Options:

**Input/Output:**
- `--input` - Path to merged CSV (default: `mimic_data/csv/merged_icd_notes.csv.gz`)
- `--output-dir` - Output directory (default: `mimic_data/jsonl`)
- `--system-prompt` - System prompt file path

**Split Ratios:**
- `--train-ratio` - Train set ratio (default: 0.80)
- `--val-ratio` - Validation set ratio (default: 0.10)
- `--test-ratio` - Test set ratio (default: 0.10)

**Sample Limits:**
- `--max-train-samples` - Max training samples (default: 2000)
- `--max-val-samples` - Max validation samples (default: 250)
- `--max-test-samples` - Max test samples (default: 250)

**Balancing Strategy:**
- `--cap-percentage` - Cap for overrepresented chapters in train (default: 15%)
- `--floor-percentage` - Minimum percentage for underrepresented chapters (default: 2%)

**Note Filtering:**
- `--percentile-low` - Lower percentile for note length filtering (default: 5.0)
- `--percentile-high` - Upper percentile for note length filtering (default: 90.0)

**Other:**
- `--seed` - Random seed for reproducibility (default: 42)

#### Sampling Strategy:

- **Train Set:** Capped-proportional sampling to balance classes while preserving clinical distribution
- **Val/Test Sets:** Natural distribution (mirrors real-world frequencies)
- **Splits by `subject_id`** to prevent patient data leakage

**Outputs:** `mimic_data/jsonl/`
- `train.jsonl` - Training data
- `val.jsonl` - Validation data
- `test.jsonl` - Test data
- `chapter_metadata.json` - Chapter descriptions and statistics

---

## Fine-Tuning

### Configuration

Edit `fine_tuning/configs/config.yaml` to customize your training run:

```yaml
# Experiment metadata
run_name: "icd-chap-1"  # Experiment name
seed: 42

# Data paths (relative to fine_tuning/)
train_path: "data/train.jsonl"
val_path: "data/val.jsonl"
test_path: "data/test.jsonl"

# Model
model: "ministral-8b-latest"

# Training hyperparameters
train:
  epochs: 1
  learning_rate: 1.5e-5
  weight_decay: 0.05
  warmup_fraction: 0.08
  
# Weights & Biases logging
wandb:
  enabled: true
  project: "icd_chapter-ft"
```

**Note:** By default, the config expects training data in `fine_tuning/data/`. You can either:
1. Copy JSONL files from `mimic_data/jsonl/` to `fine_tuning/data/`, or
2. Update the paths in `config.yaml` to point to `../../mimic_data/jsonl/`

### Run Training

```bash
cd fine_tuning
python src/train.py --config configs/config.yaml
```

**Requirements:**
- Set `MISTRAL_API_KEY` environment variable
- (Optional) Set `WANDB_API_KEY` for experiment tracking

**What happens:**
1. Uploads training/validation data to Mistral API
2. Creates a fine-tuning job (need to manually start it on Mistral dashboard)
4. Logs metrics to Weights & Biases (if enabled)

### Run Evaluation

After training completes, evaluate your model on the test set:

```bash
cd fine_tuning
python src/eval.py --model-id <your-model-id> --config configs/config.yaml
```

**Example:**
```bash
python src/eval.py --model-id ft:ministral-8b:abc123 --config configs/config.yaml
```

**Optional:**
- `--test-path` - Override test data path from config

**Outputs:** `fine_tuning/outputs/<model-id>/`
- `predictions.jsonl` - Model predictions with ground truth
- `metrics.json` - Accuracy, F1, precision, recall
- `confusion_matrix.png` - Visual confusion matrix
- `per_class_metrics.json` - Per-chapter performance
- `error_analysis.json` - Detailed error patterns

---

## Quick Start

```bash
# 0. Install dependencies with UV
uv sync

# There is a test.json file in the fine_tuning/data/ directory for quick evaluation, 
# If you just want to run the evaluation on that file, you can skip to step 5.

# 1. Download MIMIC-IV data to mimic_data/raw/

# 2. Build dataset
uv run python data_preparation/build_dataset.py

# 3. Create splits
uv run python data_preparation/sample_data_splits.py

# 4. Copy data to fine_tuning directory (or update config paths)
cp mimic_data/jsonl/*.jsonl fine_tuning/data/

# 5. Set up API keys
export MISTRAL_API_KEY="your-api-key"
export WANDB_API_KEY="your-api-key"

# 6. Train model
cd fine_tuning
uv run python src/train.py

# 7. Evaluate model (use model ID from training output)
uv run python src/eval.py --model-id ft:ministral-8b:<model-id>
