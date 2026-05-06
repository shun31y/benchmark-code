#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWMDM_DIR="${ROOT_DIR}/FlowMDM"
PYTHON_BIN="${FLOWMDM_DIR}/.venv/bin/python"
MODEL_PATH="${FLOWMDM_DIR}/results/humanml/ELMAv3_anap_hml3d/model000550000.pt"
EVAL_FILE="${FLOWMDM_DIR}/dataset/HumanML3D/humanml_test_set_anaphora.json"
GPU_ID="${GPU_ID:-0}"
EVAL_MODE="${EVAL_MODE:-final}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing ${PYTHON_BIN}. Run bash scripts/setup_flowmdm_env.sh first."
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${EVAL_FILE}" ]]; then
  echo "Missing evaluation file: ${EVAL_FILE}"
  echo "Create the dataset symlink at ${FLOWMDM_DIR}/dataset/HumanML3D first."
  exit 1
fi

cd "${FLOWMDM_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" MPLBACKEND=Agg "${PYTHON_BIN}" -m runners.eval \
  --model_path ./results/humanml/ELMAv3_anap_hml3d/model000550000.pt \
  --dataset humanml \
  --eval_mode "${EVAL_MODE}" \
  --guidance_param 2.5 \
  --transition_length 60 \
  --scenario anaphora \
  --eval_file ./dataset/HumanML3D/humanml_test_set_anaphora.json \
  --device 0 \
  "$@"
