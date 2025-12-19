"""
ITW Protocol Parser

Parse ITW protocol file and organize audio files by label (bonafide/spoof).
"""

import os
from typing import Dict, List


def parse_itw_protocol(protocol_path: str, base_dir: str) -> Dict[str, List[str]]:
    """
    Parse ITW protocol file and return dictionary mapping labels to file paths.

    Protocol format:
        file_name subset label

    Example:
        0.wav eval spoof
        4.wav eval bonafide

    Args:
        protocol_path: Path to protocol file
        base_dir: Base directory containing audio files

    Returns:
        label_dict: Dictionary mapping labels to file paths
            {
                "bonafide": ["/path/to/file1.wav", ...],
                "spoof": ["/path/to/file2.wav", ...]
            }
    """
    label_dict = {
        "bonafide": [],
        "spoof": []
    }

    print("="*80)
    print("Parsing ITW Protocol")
    print("="*80)
    print(f"Protocol: {protocol_path}")
    print(f"Base dir: {base_dir}")

    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            file_name, subset, label = parts

            # Construct full path
            file_path = os.path.join(base_dir, file_name)

            # Check if file exists
            if not os.path.exists(file_path):
                print(f"[WARNING] File not found: {file_path}")
                continue

            # Add to dictionary
            if label in label_dict:
                label_dict[label].append(file_path)

    # Print statistics
    print("\n[INFO] Protocol parsing complete!")
    print(f"  Bonafide: {len(label_dict['bonafide'])} files")
    print(f"  Spoof:    {len(label_dict['spoof'])} files")
    print(f"  Total:    {len(label_dict['bonafide']) + len(label_dict['spoof'])} files")
    print("="*80)

    return label_dict


if __name__ == "__main__":
    # Test the parser
    protocol_path = "/home/woongjae/ADD_LAB/Wav-Spec_ADD/protocols/protocol_itw.txt"
    base_dir = "/home/woongjae/ADD_LAB/Datasets/itw"

    label_dict = parse_itw_protocol(protocol_path, base_dir)

    print("\nSample files:")
    for label in ["bonafide", "spoof"]:
        print(f"\n{label}:")
        for file_path in label_dict[label][:3]:
            print(f"  {file_path}")
