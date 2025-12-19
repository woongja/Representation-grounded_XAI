#!/bin/bash

################################################################################
# ITW Dataset Temporal Difference Learning Analysis Script
#
# This script analyzes temporal difference learning for ITW dataset
# comparing bonafide vs spoof
#
# Usage:
#   bash run_analyze_temporal_itw.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# Embedding directory (choose one of the following)
EMBEDDING_DIR="/nvme3/wj/embeddings_itw/"                    # Vanilla XLSR
# EMBEDDING_DIR="/nvme3/wj/embeddings_itw_aasist/"           # Fine-tuned AASIST
# EMBEDDING_DIR="/nvme3/wj/embeddings_itw_conformertcm/"     # Fine-tuned ConformerTCM

# Output directory
OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results/ITW"
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_aasist/ITW"
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_conformertcm/ITW"

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "ITW Dataset - Temporal Difference Learning Analysis"
echo "================================================================================"
echo "Embedding dir:      $EMBEDDING_DIR"
echo "Output dir:         $OUTPUT_DIR"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Run analysis
python analyze_temporal_difference_itw.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --output_dir "$OUTPUT_DIR"

# Check if successful
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Analysis failed! Check the error message above."
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "ANALYSIS COMPLETE!"
echo "================================================================================"
echo "Total time: $((ELAPSED / 60)) minutes ($ELAPSED seconds)"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - bonafide_vs_spoof_itw.png"
echo "  - results_itw.pkl"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
