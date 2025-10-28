#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds a compressed CSV with note cleaning for ICD prediction.

This version uses note_cleaner.py which keeps ONLY:
- [DISCHARGE_DIAGNOSIS]
- [CHIEF_COMPLAINT]
- [HISTORY_OF_PRESENT_ILLNESS]
- [HOSPITAL_COURSE]
- [PROCEDURES] (capped to avoid bloat)

All other sections are removed to focus the model on core diagnostic information.

Output columns:
- subject_id, hadm_id
- note_id (chosen), note_text (discharge), cleaned_text
- icd_codes (list - PRIMARY only), seq_nums (list - always [1])
- icd_codes_str (single code), seq_nums_str (always "1")
- icd_versions (unique list), n_codes (always 1)
- icd_long_titles (list), icd_long_titles_str (single title)

Strategy:
1) Keep only discharge notes (DS). If multiple per hadm_id, keep the one with the highest note_seq.
2) Filter diagnoses to only seq_num 1 (PRIMARY diagnosis only).
3) Group diagnoses per hadm_id, preserving seq_num for each code.
4) Map long_title from d_icd_diagnoses (by icd_code + icd_version).
5) Merge notes + diagnoses on hadm_id (+ subject_id for safety).
6) Clean discharge notes using note_cleaner (5 sections only).
"""

import os
import sys
import pandas as pd
import logging


from data_preparation.note_cleaner import clean_discharge


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_D_ICD = os.path.join(PROJECT_ROOT, "mimic_data/raw/d_icd_diagnoses.csv.gz")
INPUT_DX = os.path.join(PROJECT_ROOT, "mimic_data/raw/diagnoses_icd.csv.gz")
INPUT_NOTES = os.path.join(PROJECT_ROOT, "mimic_data/raw/discharge.csv.gz")
OUTPUT = os.path.join(PROJECT_ROOT, "mimic_data/csv/merged_icd_notes.csv.gz")

os.makedirs(os.path.join(PROJECT_ROOT, "mimic_data/csv"), exist_ok=True)

logger.info("=" * 80)
logger.info("ICD DATASET BUILDER")
logger.info("=" * 80)

logger.info("Step 1/8: Loading ICD catalog...")
# 1) Load ICD catalog (code + version → long_title)
d_icd = pd.read_csv(INPUT_D_ICD, sep=",")
d_icd["icd_code"] = d_icd["icd_code"].astype(str).str.strip()
d_icd["icd_version"] = d_icd["icd_version"].astype(int)
logger.info(f"  OK - Loaded {len(d_icd)} ICD codes")

logger.info("Step 2/8: Loading diagnoses...")
# 2) Load diagnoses per admission
dx = pd.read_csv(INPUT_DX, sep=",")
dx["icd_code"] = dx["icd_code"].astype(str).str.strip()
dx["icd_version"] = dx["icd_version"].astype(int)
logger.info(f"  OK - Loaded {len(dx)} diagnosis records")

# Filter to only include ICD-10 codes and seq_num 1 (PRIMARY diagnosis only)
dx = dx[(dx["icd_version"] == 10) & (dx["seq_num"] == 1)].copy()
logger.info(f"  OK - Filtered to {len(dx)} ICD-10 diagnosis records (PRIMARY only, seq_num=1)")

logger.info("Step 3/8: Mapping ICD long titles...")
# 3) Map long_title into diagnoses rows
dx = dx.merge(
    d_icd.rename(columns={"long_title": "icd_long_title"}),
    on=["icd_code", "icd_version"],
    how="left"
)
logger.info(f"  OK - Mapped titles to {len(dx)} records")

logger.info("Step 4/8: Grouping diagnoses by admission...")
# 4) Group by admission (and subject for safety)
# Note: Since we filtered to seq_num 1 only, each admission has exactly 1 primary diagnosis
agg_dx = (
    dx.sort_values(["hadm_id", "seq_num"])
      .groupby(["subject_id", "hadm_id"], as_index=False)
      .agg({
          "icd_code": list,
          "seq_num": list,
          "icd_version": lambda s: list(sorted(set(map(int, s)))),
          "icd_long_title": list
      })
)
agg_dx["icd_codes_str"] = agg_dx["icd_code"].apply(lambda lst: ",".join(map(str, lst)))
agg_dx["seq_nums_str"] = agg_dx["seq_num"].apply(lambda lst: ",".join(map(str, lst)))
agg_dx["icd_long_titles_str"] = agg_dx["icd_long_title"].apply(
    lambda lst: " | ".join([t for t in lst if pd.notna(t)])
)
agg_dx["n_codes"] = agg_dx["icd_code"].apply(len)
logger.info(f"  OK - Grouped into {len(agg_dx)} admissions (PRIMARY diagnosis only)")

logger.info("Step 5/8: Loading discharge notes...")
# 5) Load discharge notes and keep one per hadm_id (highest note_seq)
notes = pd.read_csv(INPUT_NOTES, sep=",")
logger.info(f"  OK - Loaded {len(notes)} discharge notes")

logger.info("Step 6/8: Filtering and selecting notes...")
# Filter DS (if column exists)
if "note_type" in notes.columns:
    notes = notes[notes["note_type"].astype(str).str.upper() == "DS"].copy()

# Keep last by note_seq; fallback to latest storetime if needed
if "note_seq" in notes.columns:
    notes = (notes.sort_values(["hadm_id", "note_seq"], ascending=[True, False])
                  .groupby("hadm_id", as_index=False).head(1))
else:
    if "storetime" in notes.columns:
        notes["storetime"] = pd.to_datetime(notes["storetime"], errors="coerce")
        notes = (notes.sort_values(["hadm_id", "storetime"], ascending=[True, False])
                      .groupby("hadm_id", as_index=False).head(1))
    else:
        notes = notes.drop_duplicates(subset=["hadm_id"])

notes_sel = notes[["note_id", "subject_id", "hadm_id", "text"]].rename(
    columns={"text": "note_text"}
)
logger.info(f"  OK - Selected {len(notes_sel)} unique discharge notes")

logger.info("Step 7/8: Merging notes with diagnoses...")
# 6) Merge notes + grouped diagnoses
merged = notes_sel.merge(
    agg_dx,
    on=["subject_id", "hadm_id"],
    how="inner"  # inner ensures we only keep rows that have labels
)
logger.info(f"  OK - Merged dataset has {len(merged)} records")

logger.info("Step 8/8: Cleaning discharge notes with cleaner...")
# 8) Clean discharge notes with cleaner (5 sections only)
logger.info("   Processing discharge notes...")
logger.info(f"   Keeping ONLY: DISCHARGE_DIAGNOSIS, CHIEF_COMPLAINT, HISTORY_OF_PRESENT_ILLNESS, HOSPITAL_COURSE, PROCEDURES")

cleaned_texts = []
for idx, raw_text in enumerate(merged['note_text'].astype(str)):
    if idx % 500 == 0:
        logger.info(f"     Cleaned {idx}/{len(merged)} notes...")
    
    try:
        cleaned = clean_discharge(raw_text)
        cleaned_texts.append(cleaned)
    except Exception as e:
        logger.error(f"     Error cleaning note {idx}: {e}")
        # Fallback to raw text if cleaning fails
        cleaned_texts.append(raw_text)

merged['cleaned_text'] = cleaned_texts
logger.info(f"   OK - Cleaned all {len(cleaned_texts)} discharge notes")

# 7) Rename & select columns
merged = merged.rename(columns={
    "icd_code": "icd_codes",
    "seq_num": "seq_nums",
    "icd_version": "icd_versions",
    "icd_long_title": "icd_long_titles"
})

final_cols = [
    "subject_id", "hadm_id", "note_id", "note_text", "cleaned_text",
    "n_codes", "icd_codes", "seq_nums", "icd_codes_str", "seq_nums_str", "icd_versions",
    "icd_long_titles", "icd_long_titles_str"
]

for c in final_cols:
    if c not in merged.columns:
        merged[c] = pd.NA

merged = merged[final_cols]

# 9) Save
logger.info(f"\nSaving dataset to: {OUTPUT}")
merged.to_csv(OUTPUT, index=False, compression="gzip")

logger.info("=" * 80)
logger.info("Clean dataset built successfully")
logger.info("=" * 80)
logger.info(f"Output file: {OUTPUT}")
logger.info(f"Total rows: {len(merged)}")
logger.info(f"File size: {os.path.getsize(OUTPUT) / 1024 / 1024:.2f} MB")
logger.info("=" * 80)
