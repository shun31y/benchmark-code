#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOWMDM_DIR="${ROOT_DIR}/FlowMDM"

cd "${FLOWMDM_DIR}"
bash scripts/setup_uv_env.sh

echo
echo "Next steps:"
echo "  ln -s /path/to/HumanML3D ${FLOWMDM_DIR}/dataset/HumanML3D"
echo "  export FLOWMDM_SUPPORT_FILES_URL=<gdrive-url>"
echo "  export FLOWMDM_PRETRAINED_MODELS_URL=<gdrive-url>"
echo "  bash ${ROOT_DIR}/scripts/prepare_flowmdm_assets.sh"
