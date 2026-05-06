#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWMDM_DIR="${ROOT_DIR}/FlowMDM"

: "${FLOWMDM_SUPPORT_FILES_URL:?Set FLOWMDM_SUPPORT_FILES_URL to the support-files zip URL before running this script.}"
: "${FLOWMDM_PRETRAINED_MODELS_URL:?Set FLOWMDM_PRETRAINED_MODELS_URL to the pretrained-models zip URL before running this script.}"

cd "${FLOWMDM_DIR}"

echo "SMPL body model files are not handled by this script."
echo "Prepare body_models/ separately, e.g. with runners/prepare/download_smpl_files.sh."
bash runners/prepare/download_glove.sh
bash runners/prepare/download_t2m_evaluators.sh
bash runners/prepare/download_pretrained_models.sh
