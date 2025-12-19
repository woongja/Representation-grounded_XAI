#!/bin/bash

################################################################################
# t-SNE Visualization Script for ITW Dataset (Frame-level)
#
# This script creates t-SNE visualizations of SSL embeddings at frame-level
# comparing bonafide vs spoof
#
# Usage:
#   bash run_tsne_itw.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# Embedding directory (choose one of the following)
# EMBEDDING_DIR="/nvme3/wj/embeddings_itw/"                    # Vanilla XLSR
# EMBEDDING_DIR="/nvme3/wj/embeddings_itw_aasist/"           # Fine-tuned AASIST
EMBEDDING_DIR="/nvme3/wj/embeddings_itw_conformertcm/"     # Fine-tuned ConformerTCM

# Output directory
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results/ITW"
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_aasist/ITW"
OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_conformertcm/ITW"

# Sampling parameters
MAX_FILES_PER_ATTACK=100    # Number of files to sample per label
MAX_FRAMES_PER_FILE=50      # Number of frames to sample per file

# t-SNE parameters
PERPLEXITY=30               # t-SNE perplexity (5-50 is typical)
N_ITER=1000                 # Number of iterations
RANDOM_STATE=42             # Random seed for reproducibility

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "t-SNE Visualization (Frame-level) - ITW Dataset"
echo "================================================================================"
echo "Embedding dir:          $EMBEDDING_DIR"
echo "Output dir:             $OUTPUT_DIR"
echo "Max files per label:    $MAX_FILES_PER_ATTACK"
echo "Max frames per file:    $MAX_FRAMES_PER_FILE"
echo "Perplexity:             $PERPLEXITY"
echo "Iterations:             $N_ITER"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Run t-SNE visualization (using ITW-specific frame-level script)
python visualize_tsne_itw_frame.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_files_per_attack $MAX_FILES_PER_ATTACK \
    --max_frames_per_file $MAX_FRAMES_PER_FILE \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --random_state $RANDOM_STATE

# Check if successful
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] t-SNE visualization failed! Check the error message above."
    exit 1
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "================================================================================"
echo "VISUALIZATION COMPLETE!"
echo "================================================================================"
echo "Total time: $((ELAPSED / 60)) minutes ($ELAPSED seconds)"
echo ""
echo "Plots saved to: $OUTPUT_DIR"
echo "  - tsne_itw_frame.png"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
