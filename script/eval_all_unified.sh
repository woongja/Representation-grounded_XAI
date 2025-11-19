#!/bin/bash

# ===================================
# 통합 모델 - 모든 데이터셋 평가 자동 실행 (파라미터 버전)
# [사용법] bash eval_all_unified.sh <gpu_id> <results_dir> <model_path> <config_file>
# ===================================

# ========================
# 인자 확인
# ========================
if [ $# -ne 4 ]; then
  echo "❌ Usage: bash eval_all_unified.sh <gpu_id> <results_dir> <model_path> <config_file>"
  echo ""
  echo "Arguments:"
  echo "  gpu_id      : GPU device ID (e.g., MIG-8cdeef83-092c-5a8d-a748-452f299e1df0)"
  echo "  results_dir : Directory to save evaluation results"
  echo "  model_path  : Path to model checkpoint (.pth)"
  echo "  config_file : Path to config file (.yaml)"
  echo ""
  echo "Example:"
  echo "  bash eval_all_unified.sh MIG-xxx /home/woongjae/wildspoof/SFM-ADD/results/curriculum /home/woongjae/wildspoof/SFM-ADD/out/model.pth /home/woongjae/wildspoof/SFM-ADD/configs/config.yaml"
  exit 1
fi

GPU_ID=$1
RESULTS_DIR=$2
MODEL_PATH=$3
CONFIG_FILE=$4

echo "=========================================="
echo "🚀 Unified Model - Evaluating All Datasets"
echo "=========================================="
echo "🎮 GPU: ${GPU_ID}"
echo "📁 Results: ${RESULTS_DIR}"
echo "🤖 Model: ${MODEL_PATH}"
echo "📝 Config: ${CONFIG_FILE}"
echo "=========================================="
echo ""

DATASETS=("wildspoof" "spoofceleb_aug" "itw"  "deepen" "asv19_noise" "df21_noise")

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for DATASET in "${DATASETS[@]}"; do
  echo "=========================================="
  echo "🔍 Evaluating: ${DATASET}"
  echo "=========================================="

  if [ -z "$ALGO" ]; then
    bash "${SCRIPT_DIR}/eval_unified.sh" "${DATASET}" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"
  else
    bash "${SCRIPT_DIR}/eval_unified.sh" "${DATASET}" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}" "${ALGO}"
  fi

  # 오류 발생 시 중단
  if [ $? -ne 0 ]; then
    echo "❌ Error occurred while evaluating ${DATASET}. Stopping."
    exit 1
  fi

  echo "✅ Finished evaluation for ${DATASET}"
  echo ""
done

echo "=========================================="
echo "🎉 All evaluations completed successfully!"
echo "=========================================="
echo ""
echo "📊 Results saved in: ${RESULTS_DIR}/"
