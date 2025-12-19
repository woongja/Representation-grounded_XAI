#!/bin/bash

################################################################################
# Utterance vs Frames t-SNE Visualization Script
#
# This script visualizes how close individual frames are to their
# utterance-level representation in t-SNE space for each attack type.
#
# Usage:
#   bash run_utterance_vs_frames.sh
################################################################################

# =============================================================================
# Configuration
# =============================================================================

# Embedding directory (choose one of the following)
# EMBEDDING_DIR="/nvme3/wj/embeddings/"                    # Vanilla XLSR
# EMBEDDING_DIR="/nvme3/wj/embeddings_finetuned/"        # Fine-tuned AASIST
EMBEDDING_DIR="/nvme3/wj/embeddings_conformertcm/"     # Fine-tuned ConformerTCM

# Audio base directory (contains .flac files)
AUDIO_BASE_DIR="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019/ASVspoof2019_LA_train/flac"

# Output directory
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results/LA19/utterance_vs_frames"
# OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_aasist/LA19/utterance_vs_frames"
OUTPUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/analyze_xlsr_frame/results_conformertcm/LA19/utterance_vs_frames"

# t-SNE parameters
PERPLEXITY=30               # t-SNE perplexity (5-50 is typical)
N_ITER=1000                 # Number of iterations
RANDOM_STATE=42             # Random seed for reproducibility

# =============================================================================
# Script Start
# =============================================================================

echo "================================================================================"
echo "Utterance vs Frames t-SNE Visualization"
echo "================================================================================"
echo "Embedding dir:      $EMBEDDING_DIR"
echo "Audio base dir:     $AUDIO_BASE_DIR"
echo "Output dir:         $OUTPUT_DIR"
echo "Perplexity:         $PERPLEXITY"
echo "Iterations:         $N_ITER"
echo "Random seed:        $RANDOM_STATE"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date +%s)

# Run visualization
python visualize_utterance_vs_frames.py \
    --embedding_dir "$EMBEDDING_DIR" \
    --audio_base_dir "$AUDIO_BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --random_state $RANDOM_STATE

# Check if successful
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Visualization failed! Check the error message above."
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
echo ""
echo "t-SNE plots (utterance vs frames):"
echo "  - utterance_vs_frames_bonafide.png"
echo "  - utterance_vs_frames_A01.png"
echo "  - utterance_vs_frames_A02.png"
echo "  - utterance_vs_frames_A03.png"
echo "  - utterance_vs_frames_A04.png"
echo "  - utterance_vs_frames_A05.png"
echo "  - utterance_vs_frames_A06.png"
echo ""
echo "Waveform plots (frame distance mapping):"
echo "  - waveform_distance_bonafide.png"
echo "  - waveform_distance_A01.png"
echo "  - waveform_distance_A02.png"
echo "  - waveform_distance_A03.png"
echo "  - waveform_distance_A04.png"
echo "  - waveform_distance_A05.png"
echo "  - waveform_distance_A06.png"
echo ""
echo "================================================================================"
echo "✓ Done!"
echo "================================================================================"
