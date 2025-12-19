#!/bin/bash

# ========================
# 경로 및 설정
# ========================
DATABASE_PATH="/home/woongjae/ADD_LAB/Datasets/ASVspoof/ASVspoof2019"
CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_baseline.yaml"
PROTOCOL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/protocols/ASVspoof2019_LA_train_dev.txt"
MODEL_SAVE_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/conformertcm_xai.pth"
COMMENT="conformertcm_xai"
# --batch_size 24
# ========================
# Balanced Training 실행
# ========================
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494 python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
  --database_path ${DATABASE_PATH} \
  --protocol_path ${PROTOCOL_PATH} \
  --config ${CONFIG_FILE} \
  --batch_size 32 \
  --num_epochs 100 \
  --max_lr 1e-6 \
  --weight_decay 1e-4 \
  --patience 10 \
  --seed 1234 \
  --model_save_path ${MODEL_SAVE_PATH} \
  --comment ${COMMENT} \
  --algo 3