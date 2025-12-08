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
CONFIG_FILE="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_model_d.yaml"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/xai_asv19_d.pth"
COMMENT="xai_model_a_baseline_no_xai_losses"

# ========================
# Training 실행
# ========================
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
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
