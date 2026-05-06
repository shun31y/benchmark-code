#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

ARCHIVE_URL="${FLOWMDM_PRETRAINED_MODELS_URL:-https://drive.google.com/file/d/REPLACE_WITH_RESULTS_HUMANML_ID/view?usp=sharing}"
ARCHIVE_NAME="${FLOWMDM_PRETRAINED_MODELS_ARCHIVE:-results_humanml_pretrained_models.zip}"

if [[ "${ARCHIVE_URL}" == *"REPLACE_WITH_RESULTS_HUMANML_ID"* ]]; then
  echo "FLOWMDM_PRETRAINED_MODELS_URL is still a placeholder."
  echo "Set FLOWMDM_PRETRAINED_MODELS_URL to the shared GDrive URL for ${ARCHIVE_NAME}."
  exit 1
fi

echo -e "Downloading pretrained HumanML models"
gdown --fuzzy "${ARCHIVE_URL}" -O "${ARCHIVE_NAME}"

unzip -o "${ARCHIVE_NAME}" -d .
echo -e "Cleaning\n"
rm -f "${ARCHIVE_NAME}"

echo -e "Downloading done!"
