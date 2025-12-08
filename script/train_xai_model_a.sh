#!/bin/bash

# ========================
# SC-XAI Model A Training Script
# Model A: Baseline (No Explainability Losses)
# ASVspoof2019 LA Dataset
# ========================

# 데이터베이스 경로
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"

# 프로토콜 파일
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/protocols/ASVspoof2019_LA_train_dev.txt"

# Config 및 저장 경로
CONFIG_FILE="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_model_a.yaml"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/xai_asv19_a.pth"
COMMENT="xai_model_a_baseline_no_xai_losses"

# ========================
# Training 실행
# ========================
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=MIG-6e4275af-2db0-51f1-a601-7ad8a1002745 python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
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
