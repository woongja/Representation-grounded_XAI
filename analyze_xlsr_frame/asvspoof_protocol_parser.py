"""
ASVspoof 2019 LA Protocol Parser

This module parses the ASVspoof 2019 LA protocol file and groups files by attack type.

Protocol format:
    Column 1: speaker ID (unused)
    Column 2: file base name (audio is basename + ".flac")
    Column 3: unused ("-")
    Column 4: attack type ("-" for bonafide, "A01"~"A06" for spoof)
    Column 5: label ("bonafide" or "spoof")

Example entries:
    LA_0079 LA_T_1138215 - - bonafide
    LA_0083 LA_T_9228662 - A02 spoof
"""

import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def parse_asvspoof_protocol(protocol_path: str, base_dir: str = None) -> Dict[str, List[str]]:
    """
    Parse ASVspoof 2019 LA protocol file and group files by attack type.

    Args:
        protocol_path: Path to the protocol file (e.g., ASVspoof2019.LA.cm.train.trn.txt)
        base_dir: Base directory path for audio files (optional).
                  If provided, full paths will be: base_dir / filename.flac
                  If None, only returns filenames with .flac extension

    Returns:
        Dictionary mapping attack types to file lists:
        {
            "bonafide": [list of bonafide file paths],
            "A01": [list of A01 attack file paths],
            "A02": [list of A02 attack file paths],
            ...
            "A06": [list of A06 attack file paths]
        }

    Example:
        >>> grouped_files = parse_asvspoof_protocol(
        ...     protocol_path="/path/to/ASVspoof2019.LA.cm.train.trn.txt",
        ...     base_dir="/path/to/flac"
        ... )
        >>> len(grouped_files["bonafide"])
        25380
    """
    if not os.path.exists(protocol_path):
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")

    # Initialize dictionary for grouping files by attack type
    grouped_files = defaultdict(list)

    print(f"[INFO] Parsing protocol file: {protocol_path}")

    with open(protocol_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split by whitespace
            parts = line.split()

            if len(parts) != 5:
                print(f"[WARNING] Line {line_num}: Expected 5 columns, got {len(parts)}. Skipping.")
                continue

            # Extract relevant columns
            speaker_id = parts[0]      # Column 1 (unused)
            file_base = parts[1]       # Column 2: file base name
            unused = parts[2]          # Column 3 (unused, always "-")
            attack_type = parts[3]     # Column 4: attack type or "-" for bonafide
            label = parts[4]           # Column 5: "bonafide" or "spoof"

            # Construct filename
            filename = f"{file_base}.flac"

            # Construct full path if base_dir is provided
            if base_dir is not None:
                file_path = os.path.join(base_dir, filename)
            else:
                file_path = filename

            # Determine attack type category
            if label == "bonafide":
                # For bonafide, attack_type column is "-"
                category = "bonafide"
            elif label == "spoof":
                # For spoof, attack_type is "A01" ~ "A06"
                category = attack_type
            else:
                print(f"[WARNING] Line {line_num}: Unknown label '{label}'. Skipping.")
                continue

            # Add to grouped dictionary
            grouped_files[category].append(file_path)

    # Convert defaultdict to regular dict for cleaner output
    grouped_files = dict(grouped_files)

    # Print statistics
    print(f"\n[INFO] Protocol parsing complete!")
    print(f"[INFO] File counts by attack type:")
    total_files = 0
    for category in sorted(grouped_files.keys()):
        count = len(grouped_files[category])
        total_files += count
        print(f"  {category:>10}: {count:>6} files")
    print(f"  {'TOTAL':>10}: {total_files:>6} files")

    return grouped_files


def get_attack_type_statistics(grouped_files: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Get statistics of file counts for each attack type.

    Args:
        grouped_files: Dictionary from parse_asvspoof_protocol()

    Returns:
        Dictionary mapping attack types to counts
    """
    stats = {category: len(files) for category, files in grouped_files.items()}
    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse ASVspoof 2019 LA protocol file and group by attack type"
    )
    parser.add_argument(
        '--protocol_path',
        type=str,
        required=True,
        help='Path to ASVspoof protocol file (e.g., ASVspoof2019.LA.cm.train.trn.txt)'
    )
    parser.add_argument(
        '--base_dir',
        type=str,
        default=None,
        help='Base directory for audio files (optional). If provided, creates full file paths.'
    )

    args = parser.parse_args()

    # Parse protocol
    print("="*80)
    print("ASVspoof 2019 LA Protocol Parser")
    print("="*80)

    grouped_files = parse_asvspoof_protocol(
        protocol_path=args.protocol_path,
        base_dir=args.base_dir
    )

    # Print example files from each category
    print("\n" + "="*80)
    print("Example files from each category:")
    print("="*80)
    for category in sorted(grouped_files.keys()):
        print(f"\n{category}:")
        for i, file_path in enumerate(grouped_files[category][:3], 1):
            print(f"  {i}. {file_path}")
        if len(grouped_files[category]) > 3:
            print(f"  ... and {len(grouped_files[category]) - 3} more files")

    print("\n" + "="*80)
    print("✓ Done!")
    print("="*80)
