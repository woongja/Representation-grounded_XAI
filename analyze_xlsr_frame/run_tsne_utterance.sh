#!/bin/bash

################################################################################
# t-SNE Visualization Script (Utterance-level)
#
# This script creates t-SNE visualizations of SSL embeddings at utterance-level
# (mean pooling across time dimension)
# colored by attack type (bonafide, A01-A06)
#
# Usage:
#   bash run_tsne_utterance.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# Embedding directory (choose one of the following)
# EMBEDDING_DIR="/nvme3/wj/embeddings/"                    # Vanilla XLSR
# EMBEDDING_DIR="/nvme3/wj/embeddings_finetuned/"        # Fine-tuned AASIST
EMBEDDING_DIR="/nvme3/wj/embeddings_conformertcm/"     # Fine-tuned ConformerTCM

# Output directory
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results/LA19"
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_aasist/LA19"
OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_conformertcm/LA19"

# Sampling parameters
MAX_FILES_PER_ATTACK=""    # Leave empty for all files, or set a number (e.g., 500)

# t-SNE parameters
PERPLEXITY=30               # t-SNE perplexity (5-50 is typical)
N_ITER=1000                 # Number of iterations
RANDOM_STATE=42             # Random seed for reproducibility

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "t-SNE Visualization (Utterance-level)"
echo "================================================================================"
echo "Embedding dir:          $EMBEDDING_DIR"
echo "Output dir:             $OUTPUT_DIR"
echo "Max files per attack:   ${MAX_FILES_PER_ATTACK:-All}"
echo "Perplexity:             $PERPLEXITY"
echo "Iterations:             $N_ITER"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Build command
CMD="python visualize_tsne_utterance.py \
    --embedding_dir \"$EMBEDDING_DIR\" \
    --output_dir \"$OUTPUT_DIR\" \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --random_state $RANDOM_STATE"

# Add max_files_per_attack if set
if [ -n "$MAX_FILES_PER_ATTACK" ]; then
    CMD="$CMD --max_files_per_attack $MAX_FILES_PER_ATTACK"
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
echo "  - tsne_by_attack_utterance.png  (all attack types)"
echo "  - tsne_binary_utterance.png     (bonafide vs deepfake)"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
