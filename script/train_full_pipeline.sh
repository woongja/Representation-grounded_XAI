#!/bin/bash

# ============================================================================
# FULL PIPELINE: Stage 1 + Stage 2
# ============================================================================
# This script trains the complete 2-stage system:
# - Stage 1: Audio Deepfake Detector
# - Stage 2: XAI Module (on frozen Stage 1)
#
# Usage:
#   bash train_full_pipeline.sh              # Run both stages
#   bash train_full_pipeline.sh --stage1     # Run Stage 1 only
#   bash train_full_pipeline.sh --stage2     # Run Stage 2 only (requires pretrained Stage 1)
# ============================================================================

# ========================
# Parse Arguments
# ========================
RUN_STAGE1=true
RUN_STAGE2=true

if [ "$1" == "--stage1" ]; then
    RUN_STAGE2=false
    echo "Running STAGE 1 ONLY"
elif [ "$1" == "--stage2" ]; then
    RUN_STAGE1=false
    echo "Running STAGE 2 ONLY"
else
    echo "Running FULL PIPELINE (Stage 1 + Stage 2)"
fi

# ========================
# 공통 설정
# ========================
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/protocols/ASVspoof2019_LA_train_dev.txt"
DEVICE="MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"

# Output directory
OUT_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out"
mkdir -p ${OUT_DIR}

# Model save paths
STAGE1_MODEL="${OUT_DIR}/stage1_detector.pth"
STAGE2_MODEL="${OUT_DIR}/stage2_xai.pth"

# ========================
# STAGE 1: Detector Training
# ========================
if [ "$RUN_STAGE1" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                   STAGE 1: Detector Training                   ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Config: conformertcm_baseline.yaml"
    echo "  Output: ${STAGE1_MODEL}"
    echo ""

    CONFIG_STAGE1="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/conformertcm_baseline.yaml"

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${DEVICE} python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
      --database_path ${DATABASE_PATH} \
      --protocol_path ${PROTOCOL_PATH} \
      --config ${CONFIG_STAGE1} \
      --batch_size 32 \
      --num_epochs 30 \
      --max_lr 1e-4 \
      --weight_decay 1e-4 \
      --patience 10 \
      --seed 1234 \
      --model_save_path ${STAGE1_MODEL} \
      --comment "stage1_detector" \
      --algo 3

    STAGE1_EXIT_CODE=$?

    if [ $STAGE1_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ STAGE 1 FAILED (Exit code: ${STAGE1_EXIT_CODE})"
        exit 1
    fi

    echo ""
    echo "✅ STAGE 1 COMPLETE"
    echo "   Model saved to: ${STAGE1_MODEL}"
    echo ""

    # Wait a moment
    sleep 2
fi

# ========================
# Check Stage 1 Model Exists
# ========================
if [ "$RUN_STAGE2" = true ]; then
    # Check if we should use the just-trained model or existing one
    if [ "$RUN_STAGE1" = true ]; then
        # Use the model we just trained
        PRETRAINED_MODEL=${STAGE1_MODEL}
        echo "Using newly trained Stage 1 model: ${PRETRAINED_MODEL}"
    else
        # Check if Stage 1 model exists
        if [ -f "${STAGE1_MODEL}" ]; then
            PRETRAINED_MODEL=${STAGE1_MODEL}
            echo "Using existing Stage 1 model: ${PRETRAINED_MODEL}"
        elif [ -f "/home/woongjae/ADD_LAB/Representation-grounded_XAI/avg_5_best.pth" ]; then
            PRETRAINED_MODEL="/home/woongjae/ADD_LAB/Representation-grounded_XAI/avg_5_best.pth"
            echo "Using fallback Stage 1 model: ${PRETRAINED_MODEL}"
        else
            echo ""
            echo "❌ ERROR: No Stage 1 model found!"
            echo "   Please run Stage 1 first or provide a pretrained model."
            exit 1
        fi
    fi
fi

# ========================
# STAGE 2: XAI Module Training
# ========================
if [ "$RUN_STAGE2" = true ]; then
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                   STAGE 2: XAI Module Training                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "  Config: xai_stage2.yaml"
    echo "  Pretrained detector: ${PRETRAINED_MODEL}"
    echo "  Output: ${STAGE2_MODEL}"
    echo ""

    # Create temporary config with correct pretrained path
    CONFIG_STAGE2="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_stage2.yaml"
    TEMP_CONFIG="${OUT_DIR}/xai_stage2_temp.yaml"

    # Copy config and update pretrained path
    cp ${CONFIG_STAGE2} ${TEMP_CONFIG}

    # Update pretrained checkpoint path in temp config
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|pretrained_checkpoint:.*|pretrained_checkpoint: \"${PRETRAINED_MODEL}\"|" ${TEMP_CONFIG}
    else
        # Linux
        sed -i "s|pretrained_checkpoint:.*|pretrained_checkpoint: \"${PRETRAINED_MODEL}\"|" ${TEMP_CONFIG}
    fi

    echo "  Updated config to use: ${PRETRAINED_MODEL}"
    echo ""

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${DEVICE} python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
      --database_path ${DATABASE_PATH} \
      --protocol_path ${PROTOCOL_PATH} \
      --config ${TEMP_CONFIG} \
      --batch_size 24 \
      --num_epochs 50 \
      --max_lr 1e-4 \
      --weight_decay 1e-4 \
      --patience 10 \
      --seed 1234 \
      --model_save_path ${STAGE2_MODEL} \
      --comment "stage2_xai" \
      --algo 0

    STAGE2_EXIT_CODE=$?

    # Clean up temp config
    rm -f ${TEMP_CONFIG}

    if [ $STAGE2_EXIT_CODE -ne 0 ]; then
        echo ""
        echo "❌ STAGE 2 FAILED (Exit code: ${STAGE2_EXIT_CODE})"
        exit 1
    fi

    echo ""
    echo "✅ STAGE 2 COMPLETE"
    echo "   Model saved to: ${STAGE2_MODEL}"
    echo ""
fi

# ========================
# Final Summary
# ========================
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      PIPELINE COMPLETE! 🎉                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

if [ "$RUN_STAGE1" = true ]; then
    echo "  ✓ Stage 1 (Detector):  ${STAGE1_MODEL}"
fi

if [ "$RUN_STAGE2" = true ]; then
    echo "  ✓ Stage 2 (XAI):       ${STAGE2_MODEL}"
fi

echo ""
echo "  All models saved in: ${OUT_DIR}/"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
