#!/bin/bash

# ========================
# SC-XAI Model E Training Script
# Model E: Full SC-XAI (All Explainability Losses Enabled)
# Consistency + Sensitivity + Sparsity
# ASVspoof2019 LA Dataset
# ========================

# 데이터베이스 경로
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"

# 프로토콜 파일
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/protocols/ASVspoof2019_LA_train_dev.txt"

# Config 및 저장 경로
CONFIG_FILE="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_model_e.yaml"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/xai_asvspoof_e_full_scxai_asvspoof2019.pth"
COMMENT="xai_model_e_full_scxai_consistency_sensitivity_sparsity"

# ========================
# Training 실행
# ========================
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=MIG-57de94a5-be15-5b5a-b67e-e118352d8a59 python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 16 \
  --num_epochs 50 \
  --max_lr 1e-4 \
  --weight_decay 1e-4 \
  --patience 10 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3
