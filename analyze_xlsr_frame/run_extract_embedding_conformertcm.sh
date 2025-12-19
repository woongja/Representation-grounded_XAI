#!/bin/bash

################################################################################
# Fine-tuned ConformerTCM SSL Embedding Extraction Script
#
# This script extracts embeddings from ASVspoof 2019 LA dataset
# using fine-tuned ConformerTCM model (SSL frontend only) and saves them by attack type
#
# Usage:
#   bash run_extract_embedding_conformertcm.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# ASVspoof 2019 LA protocol file
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Audio base directory (containing .flac files)
AUDIO_BASE_DIR="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_train/flac"

# Fine-tuned ConformerTCM model path
MODEL_PATH="/home/woongjae/wildspoof/tcm_add/avg_5_best.pth"

# Output directory (will create subdirectories: bonafide, A01, A02, ..., A06)
OUTPUT_DIR="/nvme3/wj/embeddings_conformertcm/"

# Device
DEVICE="cuda"

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "Fine-tuned ConformerTCM SSL Embedding Extraction"
echo "================================================================================"
echo "Protocol:       $PROTOCOL_PATH"
echo "Audio dir:      $AUDIO_BASE_DIR"
echo "Model path:     $MODEL_PATH"
echo "Output dir:     $OUTPUT_DIR"
echo "Device:         $DEVICE"
echo "================================================================================"
echo ""
echo "This will extract SSL embeddings (frontend only) and save them in:"
echo "  $OUTPUT_DIR/bonafide/*.pt"
echo "  $OUTPUT_DIR/A01/*.pt"
echo "  $OUTPUT_DIR/A02/*.pt"
echo "  $OUTPUT_DIR/A03/*.pt"
echo "  $OUTPUT_DIR/A04/*.pt"
echo "  $OUTPUT_DIR/A05/*.pt"
echo "  $OUTPUT_DIR/A06/*.pt"
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
python extract_embedding_conformertcm.py \
    --protocol_path "$PROTOCOL_PATH" \
    --audio_base_dir "$AUDIO_BASE_DIR" \
    --model_path "$MODEL_PATH" \
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
echo "    │   ├── LA_T_xxxxxxx.pt"
echo "    │   └── ..."
echo "    ├── A01/"
echo "    ├── A02/"
echo "    ├── A03/"
echo "    ├── A04/"
echo "    ├── A05/"
echo "    ├── A06/"
echo "    └── embedding_index.pkl"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
