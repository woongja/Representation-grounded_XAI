#!/bin/bash

# ===================================
# 통합 평가 스크립트
# 여기에서 모델과 설정만 바꿔서 사용하세요
# 원하는 모델의 주석을 해제하고 실행하면 됩니다
# ===================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===================================
# 공통 설정
# ===================================
GPU_ID="MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"

# ===================================
# 1. WavSpec baseline(asv19)
# ===================================
RESULTS_DIR="/home/woongjae/ADD_LAB/Representation-grounded_XAI/results/b"
MODEL_PATH="/home/woongjae/ADD_LAB/Representation-grounded_XAI/out/xai_asv19_b.pth"
CONFIG_FILE="/home/woongjae/ADD_LAB/Representation-grounded_XAI/configs/xai_model_b.yaml"

bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"


# ===================================
# 💡 사용 방법
# ===================================
# 1. 위에서 평가하고 싶은 모델의 주석(#)을 제거하세요
# 2. 다른 모델들은 주석 처리하세요
# 3. bash scripts/eval_script.sh 실행 (프로젝트 루트에서)
#    또는 cd scripts && bash eval_script.sh
#
# 또는 단일 데이터셋만 평가하려면:
# bash scripts/eval_unified.sh <dataset> "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"
# 예: bash scripts/eval_unified.sh itw "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"
