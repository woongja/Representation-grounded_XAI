#!/bin/bash

################################################################################
# ITW Dataset SSL Embedding Extraction Script
#
# This script extracts embeddings from ITW dataset
# using vanilla wav2vec2-XLSR model and saves them by label
#
# Usage:
#   bash run_extract_embedding_itw.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# ITW protocol file
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Wav-Spec_ADD/protocols/protocol_itw.txt"

# Audio base directory (containing .wav files)
AUDIO_BASE_DIR="/home/woongjae/ADD_LAB/Datasets/itw"

# Output directory (will create subdirectories: bonafide, spoof)
OUTPUT_DIR="/nvme3/wj/embeddings_itw/"

# Device
DEVICE="cuda"

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "ITW Dataset - SSL Embedding Extraction"
echo "================================================================================"
echo "Protocol:       $PROTOCOL_PATH"
echo "Audio dir:      $AUDIO_BASE_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Device:         $DEVICE"
echo "================================================================================"
echo ""
echo "This will extract embeddings and save them in:"
echo "  $OUTPUT_DIR/bonafide/*.pt"
echo "  $OUTPUT_DIR/spoof/*.pt"
echo ""
echo "Each .pt file contains embeddings of shape (T, 1024)"
echo "where T is the number of frames (varies per audio file)"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Run extraction
python extract_embedding_itw.py \
    --protocol_path "$PROTOCOL_PATH" \
    --audio_base_dir "$AUDIO_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

# Check if successful
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Embedding extraction failed! Check the error message above."
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "EXTRACTION COMPLETE!"
echo "================================================================================"
echo "Total time: $((ELAPSED / 60)) minutes ($ELAPSED seconds)"
echo ""
echo "Embeddings saved to: $OUTPUT_DIR"
echo ""
echo "Directory structure:"
echo "  $OUTPUT_DIR/"
echo "    ├── bonafide/"
echo "    │   ├── 0.pt"
echo "    │   ├── 1.pt"
echo "    │   └── ..."
echo "    ├── spoof/"
echo "    │   ├── 2.pt"
echo "    │   └── ..."
echo "    └── embedding_index.pkl"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
