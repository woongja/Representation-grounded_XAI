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
GPU_ID="MIG-46b32d1b-f775-5b7d-a987-fb8ebc049494"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/baseline"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_baseline_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_baseline.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/concat"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_concat_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_concat.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/gated"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_gated_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_gated.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/covariance"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_covariance_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_covariance.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/covariance_diagonal"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_covariance_diagonal_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_covariance.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
# RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/crossattn"
# MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_crossattn_spoofceleb_aug.pth"
# CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_crossattn.yaml"

# bash "${SCRIPT_DIR}/eval_all_unified.sh" "${GPU_ID}" "${RESULTS_DIR}" "${MODEL_PATH}" "${CONFIG_FILE}"

# ===================================
# 1. ConformerTCM baseline model (spoofceleb_aug)
# ===================================
RESULTS_DIR="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/results/spoofceleb_aug/attnmask"
MODEL_PATH="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/out/conformertcm_fusion_attnmask_spoofceleb_aug.pth"
CONFIG_FILE="/home/woongjae/ADD_LAB/SSL_Fusion_ADD/configs/conformertcm_fusion_attnmask.yaml"

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
