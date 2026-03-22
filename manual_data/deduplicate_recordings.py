#!/usr/bin/env python3
"""
Analyze and deduplicate respiratory sound recordings.

This script identifies duplicate recordings for the same patient/session
and keeps only the one with the best quality metrics (highest entropy 
and lowest zero-crossing rate variance).
"""

import csv
import re
from collections import defaultdict
from pathlib import Path


EXCLUDED_DIAGNOSES = {'Audio files', 'Unknown'}


def extract_recording_id(filename: str) -> str:
    """
    Extract the recording ID from filename.
    
    E.g., "104_1b1_Al_sc_Litt3200.wav" -> "104_1b1"
    """
    # Match pattern: patient_id + session (e.g., "104_1b1")
    match = re.match(r"(\d+_\d[a-z]\d)", filename)
    return match.group(1) if match else filename


def load_data(csv_path: str) -> list[dict]:
    """Load CSV data into memory."""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data


def filter_noisy_diagnoses(data: list[dict]) -> tuple[list[dict], int]:
    """Remove rows with noisy diagnosis labels."""
    filtered = [row for row in data if row.get('diagnosis') not in EXCLUDED_DIAGNOSES]
    dropped = len(data) - len(filtered)
    return filtered, dropped


def find_duplicates(data: list[dict]) -> dict:
    """
    Group recordings by patient_id to find duplicates.
    Keep only ONE row per patient_id.
    
    Returns:
        Dict mapping patient_id -> list of row indices
    """
    duplicates = defaultdict(list)
    
    for idx, row in enumerate(data):
        patient_id = row['patient_id']
        duplicates[patient_id].append(idx)
    
    return {k: v for k, v in duplicates.items() if len(v) > 1}


def calculate_quality_score(row: dict) -> float:
    """
    Calculate quality score for a recording.
    
    Higher entropy and lower ZCR variance = better quality
    """
    try:
        entropy = float(row.get('entropy', 0))
        zcr_var = float(row.get('zcr_variance', float('inf')))
        # Normalize ZCR variance to be inversely proportional (lower is better)
        # Add 1 to avoid division by zero
        score = entropy / (1 + zcr_var)
        return score
    except (ValueError, TypeError):
        return 0


def select_best_recording(indices: list[int], data: list[dict]) -> int:
    """
    Select the best recording from a group of duplicates (by index).
    
    Criteria: highest entropy, lowest ZCR variance
    """
    rows = [data[idx] for idx in indices]
    scored = [(calculate_quality_score(row), idx) for idx, row in zip(indices, rows)]
    scored.sort(reverse=True, key=lambda x: x[0])
    
    best_score, best_idx = scored[0]
    return best_idx


def deduplicate(input_csv: str, output_csv: str) -> dict:
    """
    Main deduplication function.
    
    Returns stats dict with:
        - total_records: Original number of records
        - duplicates_found: Groups of duplicates found
        - records_dropped: Total records removed
        - remaining_records: Final number of records
    """
    print("Loading data...")
    data = load_data(input_csv)
    total_records = len(data)
    print(f"Loaded {total_records} records")

    data, noisy_dropped = filter_noisy_diagnoses(data)
    print(f"Dropped noisy diagnoses ({', '.join(sorted(EXCLUDED_DIAGNOSES))}): {noisy_dropped}")
    print(f"Records after diagnosis cleanup: {len(data)}")
    
    print("\nFinding duplicates...")
    duplicates = find_duplicates(data)
    print(f"Found {len(duplicates)} duplicate groups")
    
    # Build set of indices to keep (best from each duplicate group)
    keep_indices = set()
    dropped_count = 0
    best_selections = []
    
    print("\nSelecting best recordings...")
    for patient_id, indices_group in sorted(duplicates.items()):
        best_idx = select_best_recording(indices_group, data)
        best_selections.append((patient_id, data[best_idx]['filename']))
        keep_indices.add(best_idx)
        dropped_count += len(indices_group) - 1
    
    # Build set of all indices that are part of duplicate groups
    all_duplicate_indices = set()
    for indices_group in duplicates.values():
        all_duplicate_indices.update(indices_group)
    
    # Keep rows by index:
    # 1. Not part of any duplicate group, OR
    # 2. Part of a duplicate group AND selected as best
    final_data = []
    for idx, row in enumerate(data):
        if idx in all_duplicate_indices:
            # This row is part of a duplicate group - keep only if it's the best
            if idx in keep_indices:
                final_data.append(row)
        else:
            # Not part of any duplicate group - keep it
            final_data.append(row)
    
    print(f"\nDuplicate groups details:")
    for patient_id, best_filename in sorted(best_selections):
        print(f"  Patient {patient_id}: Kept {best_filename}")
    
    print(f"\nWriting deduplicated data to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        if final_data:
            fieldnames = final_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_data)
    
    stats = {
        'total_records': total_records,
        'noisy_diagnoses_dropped': noisy_dropped,
        'records_after_cleanup': len(data),
        'duplicates_found': len(duplicates),
        'records_dropped': dropped_count,
        'remaining_records': len(final_data),
    }
    
    return stats


if __name__ == '__main__':
    # Setup paths
    script_dir = Path(__file__).parent
    input_file = script_dir / 'combined_respiratory_features_reindexed.csv'
    output_file = script_dir / 'combined_respiratory_features_deduplicated.csv'
    
    print(f"Deduplicating: {input_file}\n")
    
    stats = deduplicate(str(input_file), str(output_file))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original records:     {stats['total_records']}")
    print(f"Noisy rows dropped:   {stats['noisy_diagnoses_dropped']}")
    print(f"After cleanup:        {stats['records_after_cleanup']}")
    print(f"Duplicate groups:     {stats['duplicates_found']}")
    print(f"Records dropped:      {stats['records_dropped']}")
    print(f"Final records:        {stats['remaining_records']}")
    total_reduction = stats['total_records'] - stats['remaining_records']
    print(f"Total reduction:      {total_reduction} ({100*total_reduction/stats['total_records']:.1f}%)")
    print("="*60)
    print(f"\nOutput saved to: {output_file}")
