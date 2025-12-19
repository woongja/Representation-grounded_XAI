#!/bin/bash

# ========================
# STAGE 2: XAI Module Training
# ========================
# This script trains the XAI module on a FROZEN pretrained detector.
#
# Prerequisites:
# - Stage 1 detector must be trained and saved as avg_5_best.pth
#
# What this does:
# 1. Loads pretrained detector from avg_5_best.pth
# 2. Freezes ALL detector parameters
# 3. Trains ONLY the XAI module (Importance Network + Prototypes)

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"
CONFIG_FILE="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_stage2.yaml"
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/protocols/ASVspoof2019_LA_train_dev.txt"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/xai_stage2_best.pth"
COMMENT="xai_stage2"

# ========================
# STAGE 2 XAI Training 실행
# ========================
echo "============================================"
echo "STAGE 2: XAI Module Training"
echo "============================================"
echo "  Pretrained detector: avg_5_best.pth"
echo "  Config: xai_stage2.yaml"
echo "  Output: ${MODEL_SAVE_PATH}"
echo "============================================"
echo ""

OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=MIG-6e4275af-2db0-51f1-a601-7ad8a1002745 python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 24 \
  --num_epochs 50 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 10 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3

echo ""
echo "============================================"
echo "STAGE 2 Training Complete!"
echo "Model saved to: ${MODEL_SAVE_PATH}"
echo "============================================"
