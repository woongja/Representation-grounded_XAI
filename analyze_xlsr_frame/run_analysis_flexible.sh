#!/bin/bash

################################################################################
# Flexible Analysis Pipeline for Temporal Difference Learning
#
# This script allows you to enable/disable each step independently
#
# Usage:
#   1. Edit the paths in "Configuration" section
#   2. Set RUN_STEP1/RUN_STEP2 to true/false
#   3. Run: bash run_analysis_flexible.sh
################################################################################

# =============================================================================
# Configuration - EDIT THESE PATHS
# =============================================================================

# Protocol file path
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

# Audio base directory (directory containing .flac files)
AUDIO_BASE_DIR="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_train/flac"

# Embedding directory (fast SSD recommended)
EMBEDDING_DIR="/nvme3/wj/embeddings/"

# Output directory for results
OUTPUT_DIR="./results/"

# Device (cuda or cpu)
DEVICE="cuda"

# Batch size for embedding extraction (adjust based on GPU memory)
BATCH_SIZE=32

# =============================================================================
# Step Control - SET TO true/false TO ENABLE/DISABLE EACH STEP
# =============================================================================

# Step 1: Extract embeddings (set to false if already done)
RUN_STEP1=true

# Step 2: Figure 2 analysis (cosine similarity)
RUN_STEP2=true

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "Temporal Difference Learning - Analysis Pipeline (Figure 2 Only)"
echo "================================================================================"
echo "Protocol: $PROTOCOL_PATH"
echo "Audio base dir: $AUDIO_BASE_DIR"
echo "Embedding dir: $EMBEDDING_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Batch size: $BATCH_SIZE"
echo ""
echo "Steps to run:"
echo "  Step 1 (Embedding extraction): $RUN_STEP1"
echo "  Step 2 (Figure 2 analysis):    $RUN_STEP2"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# =============================================================================
# Step 1: Extract SSL Embeddings
# =============================================================================

if [ "$RUN_STEP1" = true ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 1: Extracting SSL Embeddings"
    echo "================================================================================"
    echo "This step may take 2-4 hours for ~25,000 files"
    echo "Resume mode is enabled - safe to interrupt and restart"
    echo "-------------------------------------------------------------------------------"
    echo ""

    python extract_embeddings.py \
        --protocol_path "$PROTOCOL_PATH" \
        --audio_base_dir "$AUDIO_BASE_DIR" \
        --output_dir "$EMBEDDING_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE"

    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Step 1 failed! Check the error message above."
        exit 1
    fi

    STEP1_END=$(date +%s)
    STEP1_TIME=$((STEP1_END - START_TIME))

    echo ""
    echo "[INFO] Step 1 completed in $((STEP1_TIME / 60)) minutes"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "STEP 1: SKIPPED (RUN_STEP1=false)"
    echo "================================================================================"
    echo "Using existing embeddings from: $EMBEDDING_DIR"
    echo ""
    STEP1_END=$START_TIME
    STEP1_TIME=0
fi

# =============================================================================
# Step 2: Figure 2 Analysis - Cosine Similarity (Equation 1 & 2)
# =============================================================================

if [ "$RUN_STEP2" = true ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2: Figure 2 Analysis - Cosine Similarity (Equation 1 & 2)"
    echo "================================================================================"
    echo "Computing direction vectors and cosine similarities..."
    echo "-------------------------------------------------------------------------------"
    echo ""

    python figure2_cosine_similarity_analysis.py \
        --embedding_dir "$EMBEDDING_DIR" \
        --output_dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Step 2 failed! Check the error message above."
        exit 1
    fi

    STEP2_END=$(date +%s)
    STEP2_TIME=$((STEP2_END - STEP1_END))

    echo ""
    echo "[INFO] Step 2 completed in $STEP2_TIME seconds"
    echo ""
else
    echo ""
    echo "================================================================================"
    echo "STEP 2: SKIPPED (RUN_STEP2=false)"
    echo "================================================================================"
    echo ""
    STEP2_END=$STEP1_END
    STEP2_TIME=0
fi

# =============================================================================
# Summary
# =============================================================================

TOTAL_TIME=$((STEP2_END - START_TIME))

echo ""
echo "================================================================================"
echo "PIPELINE COMPLETE!"
echo "================================================================================"
echo "Total time: $((TOTAL_TIME / 60)) minutes ($TOTAL_TIME seconds)"
echo ""

if [ "$RUN_STEP1" = true ]; then
    echo "Step 1 (Embedding extraction): $((STEP1_TIME / 60)) minutes"
fi

if [ "$RUN_STEP2" = true ]; then
    echo "Step 2 (Figure 2 analysis):     $STEP2_TIME seconds"
fi

echo ""
echo "================================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "================================================================================"
echo ""

if [ "$RUN_STEP2" = true ]; then
    echo "Generated files:"
    echo "  Figure 2:"
    echo "    - figure2_results.pkl"
    echo "    - figure2_cosine_similarity_boxplots.png"
    echo "    - figure2_cosine_similarity_boxplots.pdf"
    echo "    - figure2_mean_boxplot.png"
    echo "    - figure2_std_boxplot.png"
    echo ""
fi

echo "================================================================================"
echo "✓ All Done!"
echo "================================================================================"
