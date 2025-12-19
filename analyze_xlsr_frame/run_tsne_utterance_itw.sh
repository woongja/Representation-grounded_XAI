#!/bin/bash

################################################################################
# t-SNE Visualization Script for ITW Dataset (Utterance-level)
#
# This script creates t-SNE visualizations of SSL embeddings at utterance-level
# (mean pooling across time dimension)
# comparing bonafide vs spoof
#
# Usage:
#   bash run_tsne_utterance_itw.sh
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
MAX_FILES_PER_LABEL=""    # Leave empty for all files, or set a number (e.g., 500)

# t-SNE parameters
PERPLEXITY=30               # t-SNE perplexity (5-50 is typical)
N_ITER=1000                 # Number of iterations
RANDOM_STATE=42             # Random seed for reproducibility

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "t-SNE Visualization (Utterance-level) - ITW Dataset"
echo "================================================================================"
echo "Embedding dir:          $EMBEDDING_DIR"
echo "Output dir:             $OUTPUT_DIR"
echo "Max files per label:    ${MAX_FILES_PER_LABEL:-All}"
echo "Perplexity:             $PERPLEXITY"
echo "Iterations:             $N_ITER"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Build command
CMD="python visualize_tsne_itw.py \
    --embedding_dir \"$EMBEDDING_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --random_state $RANDOM_STATE"

# Add max_files_per_label if set
if [ -n "$MAX_FILES_PER_LABEL" ]; then
    CMD="$CMD --max_files_per_label $MAX_FILES_PER_LABEL"
fi

# Run t-SNE visualization
eval $CMD

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
echo "  - tsne_itw_utterance.png"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
