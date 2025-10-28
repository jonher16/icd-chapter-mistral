"""
Discharge note cleaner for ICD code prediction.
Keeps only the 5 most essential sections:
- DISCHARGE_DIAGNOSIS
- CHIEF_COMPLAINT  
- HISTORY_OF_PRESENT_ILLNESS
- HOSPITAL_COURSE
- PROCEDURES (capped to avoid bloat)

All other sections are removed to focus the model on core diagnostic information.
"""

import re
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Maximum character length for PROCEDURES section to avoid bloat
MAX_PROCEDURES_LENGTH = 1000


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace and blank lines."""
    # Remove trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.M)
    
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove lines with only dashes or equals
    text = re.sub(r'^[-=_]{3,}$', '', text, flags=re.M)
    
    return text.strip()


def remove_administrative_info(text: str) -> str:
    """Remove administrative and identifying information."""
    # Remove header lines (Name, Unit No, dates, etc.)
    text = re.sub(r'^Name:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Unit No:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Admission Date:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Discharge Date:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Date of Birth:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Service:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Attending:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Sex:.*?$', '', text, flags=re.M | re.I)
    text = re.sub(r'^Followup Instructions:.*?$', '', text, flags=re.M | re.I)
    
    return text


def preprocess_text(text: str) -> str:
    """Preprocess text by adding line breaks before section headers."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Redact PHI markers
    text = re.sub(r'\[?\*{2}.+?\*{2}\]?', '[REDACTED]', text)
    text = re.sub(r'_{3,}', '[REDACTED]', text)
    
    # Add line breaks before section headers to help with extraction
    section_patterns = [
        (r'Chief Complaint:', '\n\nChief Complaint:\n'),
        (r'History of Present Illness:', '\n\nHistory of Present Illness:\n'),
        (r'Brief Hospital Course:', '\n\nBrief Hospital Course:\n'),
        (r'Hospital Course:', '\n\nHospital Course:\n'),
        (r'Discharge Diagnosis:', '\n\nDischarge Diagnosis:\n'),
        (r'Discharge Diagnoses:', '\n\nDischarge Diagnoses:\n'),
        (r'Final Diagnosis:', '\n\nFinal Diagnosis:\n'),
        (r'Major Surgical or Invasive Procedure:', '\n\nMajor Surgical or Invasive Procedure:\n'),
        (r'Procedures:', '\n\nProcedures:\n'),
    ]
    
    for pattern, replacement in section_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Remove administrative info
    text = remove_administrative_info(text)
    
    return text


def extract_sections(text: str) -> Dict[str, str]:
    """
    Extract only the 5 essential sections using regex.
    Returns a dict with normalized section names.
    """
    # Define section patterns (case-insensitive)
    section_definitions = [
        # Discharge Diagnosis variants
        ("DISCHARGE_DIAGNOSIS", [
            r'Discharge Diagnosis:?\s*\n',
            r'Discharge Diagnoses:?\s*\n',
            r'Final Diagnosis:?\s*\n',
        ]),
        # Chief Complaint variants
        ("CHIEF_COMPLAINT", [
            r'Chief Complaint:?\s*\n',
            r'CC:?\s*\n',
        ]),
        # History of Present Illness variants
        ("HISTORY_OF_PRESENT_ILLNESS", [
            r'History of Present Illness:?\s*\n',
            r'HPI:?\s*\n',
        ]),
        # Hospital Course variants
        ("HOSPITAL_COURSE", [
            r'Hospital Course:?\s*\n',
            r'Brief Hospital Course:?\s*\n',
        ]),
        # Procedures variants
        ("PROCEDURES", [
            r'Major Surgical or Invasive Procedure:?\s*\n',
            r'Procedures:?\s*\n',
            r'Operations:?\s*\n',
        ]),
    ]
    
    # Also find positions of sections we DON'T want (to use as boundaries)
    exclude_patterns = [
        r'Past Medical History:?\s*\n',
        r'Physical Exam:?\s*\n',
        r'Medications on Admission:?\s*\n',
        r'Discharge Medications:?\s*\n',
        r'Discharge Instructions:?\s*\n',
        r'Discharge Condition:?\s*\n',
        r'Discharge Disposition:?\s*\n',
        r'Followup Instructions:?\s*\n',
        r'Social History:?\s*\n',
        r'Family History:?\s*\n',
        r'Allergies:?\s*\n',
        r'Pertinent Results:?\s*\n',
        r'Labs:?\s*\n',
        r'Laboratory Data:?\s*\n',
    ]
    
    # Find all section positions (both kept and excluded)
    all_section_positions = []
    
    for normalized_name, patterns in section_definitions:
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                all_section_positions.append((match.start(), normalized_name, pattern, match, True))
    
    # Add excluded sections as boundaries
    for pattern in exclude_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            all_section_positions.append((match.start(), "EXCLUDE", pattern, match, False))
    
    # Sort by position
    all_section_positions.sort(key=lambda x: x[0])
    
    # Extract text between sections (only for kept sections)
    kept_sections = {}
    for i, (start_pos, normalized_name, pattern, match, is_kept) in enumerate(all_section_positions):
        # Skip excluded sections
        if not is_kept:
            continue
        
        # Find where this section ends (start of next section or end of text)
        if i + 1 < len(all_section_positions):
            end_pos = all_section_positions[i + 1][0]
        else:
            end_pos = len(text)
        
        # Extract section body
        body = text[start_pos:end_pos].strip()
        
        # Remove the section header from the body
        body = re.sub(pattern, '', body, count=1, flags=re.IGNORECASE).strip()
        
        # Clean the body
        body = normalize_whitespace(body)
        
        # Skip empty sections
        if not body or len(body) < 10:
            continue
        
        # Cap PROCEDURES section to avoid bloat
        if normalized_name == "PROCEDURES" and len(body) > MAX_PROCEDURES_LENGTH:
            body = body[:MAX_PROCEDURES_LENGTH] + "... [truncated]"
        
        # If section already exists, append (consolidate duplicates)
        if normalized_name in kept_sections:
            kept_sections[normalized_name] += "\n\n" + body
        else:
            kept_sections[normalized_name] = body
    
    return kept_sections


def clean_discharge(text: str) -> str:
    """
    Main cleaning function - keeps only 5 essential sections.
    
    Returns:
        Cleaned text with format:
        [DISCHARGE_DIAGNOSIS]
        content
        
        [CHIEF_COMPLAINT]
        content
        
        [HISTORY_OF_PRESENT_ILLNESS]
        content
        
        [HOSPITAL_COURSE]
        content
        
        [PROCEDURES]
        content (capped)
    """
    # Preprocess
    text = preprocess_text(text)
    
    # Extract sections
    sections = extract_sections(text)
    
    # If no sections found, return minimal cleaned text
    if not sections:
        logger.warning("No sections extracted - returning minimally cleaned text")
        return normalize_whitespace(text)
    
    # Reconstruct with sections in priority order
    priority_order = [
        "DISCHARGE_DIAGNOSIS",
        "CHIEF_COMPLAINT",
        "HISTORY_OF_PRESENT_ILLNESS",
        "HOSPITAL_COURSE",
        "PROCEDURES",
    ]
    
    output_sections = []
    for section_name in priority_order:
        if section_name in sections:
            output_sections.append(f"[{section_name}]\n{sections[section_name]}")
    
    result = "\n\n".join(output_sections)
    result = normalize_whitespace(result)
    
    return result