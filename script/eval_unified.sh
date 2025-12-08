#!/bin/bash

# ===================================
# 통합 평가 스크립트
# 모든 모델 타입에 대해 사용 가능
# [사용법] bash eval_unified.sh <dataset_name> <gpu_id> <results_dir> <model_path> <config_file>
# ===================================

# ========================
# 인자 확인
# ========================
if [ $# -ne 5 ]; then
  echo "❌ Usage: bash eval_unified.sh <dataset_name> <gpu_id> <results_dir> <model_path> <config_file>"
  echo ""
  echo "Arguments:"
  echo "  dataset_name : Dataset to evaluate (itw, add2022, wildspoof, deepen, asv19_noise, df21_noise)"
  echo "  gpu_id       : GPU device ID (e.g., MIG-8cdeef83-092c-5a8d-a748-452f299e1df0)"
  echo "  results_dir  : Directory to save evaluation results"
  echo "  model_path   : Path to model checkpoint (.pth)"
  echo "  config_file  : Path to config file (.yaml)"
  echo ""
  echo "Example:"
  echo "  bash eval_unified.sh itw MIG-xxx /path/to/results /path/to/model.pth /path/to/config.yaml"
  exit 1
fi

DATASET=$1
GPU_ID=$2
RESULTS_DIR=$3
MODEL_PATH=$4
CONFIG_FILE=$5

# ========================
# 설정
# ========================
# 공통 데이터셋 정보
DATASET_YAML="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/datasets_base.yaml"

# 결과 저장 경로 (자동 생성)
EVAL_OUTPUT="${RESULTS_DIR}/eval_${DATASET}.txt"

# ========================
# YAML 파서 (yq로 읽기)
# ========================
DATABASE_PATH=$(yq ".${DATASET}.database_path" ${DATASET_YAML})
PROTOCOL_PATH=$(yq ".${DATASET}.protocol_path" ${DATASET_YAML})

# 🔧 따옴표 제거
DATABASE_PATH=$(echo $DATABASE_PATH | sed 's/"//g')
PROTOCOL_PATH=$(echo $PROTOCOL_PATH | sed 's/"//g')

# ========================
# 값 확인
# ========================
if [ "$DATABASE_PATH" == "null" ] || [ "$PROTOCOL_PATH" == "null" ]; then
  echo "❌ Dataset '${DATASET}' not found in ${DATASET_YAML}"
  echo "Available datasets: itw, add2022, wildspoof, deepen, asv19_noise, df21_noise"
  exit 1
fi

# 모델 파일 확인
if [ ! -f "$MODEL_PATH" ]; then
  echo "❌ Model file not found: ${MODEL_PATH}"
  exit 1
fi

# Config 파일 확인
if [ ! -f "$CONFIG_FILE" ]; then
  echo "❌ Config file not found: ${CONFIG_FILE}"
  exit 1
fi

# 결과 디렉토리 생성
mkdir -p ${RESULTS_DIR}

echo "=========================================="
echo "🚀 Unified Model Evaluation"
echo "=========================================="
echo "📊 Dataset: ${DATASET}"
echo "📂 Database: ${DATABASE_PATH}"
echo "📜 Protocol: ${PROTOCOL_PATH}"
echo "🤖 Model: ${MODEL_PATH}"
echo "📝 Config: ${CONFIG_FILE}"
echo "💾 Output: ${EVAL_OUTPUT}"
echo "🎮 GPU: ${GPU_ID}"
echo "=========================================="

# ========================
# 평가 실행
# ========================
CUDA_VISIBLE_DEVICES=${GPU_ID} python /home/woongjae/ADD_LAB/Representation-grounded_XAI/main.py \
  --eval \
  --database_path "${DATABASE_PATH}" \
  --protocol_path "${PROTOCOL_PATH}" \
  --config "${CONFIG_FILE}" \
  --model_path "${MODEL_PATH}" \
  --eval_output "${EVAL_OUTPUT}" \
  --batch_size 32

# ========================
# 결과 확인
# ========================
if [ $? -eq 0 ]; then
  echo ""
  echo "=========================================="
  echo "✅ Evaluation completed successfully!"
  echo "=========================================="
  echo "📊 Results saved to: ${EVAL_OUTPUT}"
else
  echo ""
  echo "=========================================="
  echo "❌ Evaluation failed!"
  echo "=========================================="
  exit 1
fi
