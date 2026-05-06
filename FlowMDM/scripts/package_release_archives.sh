#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/release_assets}"

mkdir -p "${OUT_DIR}"

SUPPORT_ARCHIVE="${OUT_DIR}/flowmdm_support_files.zip"
MODELS_ARCHIVE="${OUT_DIR}/results_humanml_pretrained_models.zip"

rm -f "${SUPPORT_ARCHIVE}" "${MODELS_ARCHIVE}"

cd "${ROOT_DIR}"

zip -q -r "${SUPPORT_ARCHIVE}" \
  dataset/t2m_mean.npy \
  dataset/t2m_std.npy \
  dataset/HML_Mean_Gen.npy \
  dataset/HML_Std_Gen.npy \
  t2m/humanml/text_mot_match/model/finest.tar

zip -q -r "${MODELS_ARCHIVE}" \
  results/humanml/FlowMDM/args.json \
  results/humanml/FlowMDM/model000500000.pt \
  results/humanml/ELMAv3_anap_hml3d/args.json \
  results/humanml/ELMAv3_anap_hml3d/model000550000.pt \
  results/humanml/ELMAv3_anap_hml3d/opt000550000.pt \
  results/humanml/ELMAv3_anap_hml3d_hist/args.json \
  results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt \
  results/humanml/ELMAv3_anap_hml3d_hist/opt000550000.pt

echo "Created ${SUPPORT_ARCHIVE}"
echo "Created ${MODELS_ARCHIVE}"
