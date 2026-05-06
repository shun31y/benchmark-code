#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-${ROOT_DIR}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

cd "${ROOT_DIR}"

uv python install "${PYTHON_VERSION}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

PYTHON_BIN="${VENV_DIR}/bin/python"

# chumpy requires pip to be available in the environment when build isolation is disabled.
uv pip install --python "${PYTHON_BIN}" pip setuptools==65.5.0 wheel
uv pip install --python "${PYTHON_BIN}" --no-build-isolation chumpy==0.70
uv pip install --python "${PYTHON_BIN}" -r requirements-uv.txt
uv pip install --python "${PYTHON_BIN}" setuptools==65.5.0

"${PYTHON_BIN}" -m spacy download en_core_web_sm

echo
echo "Environment is ready at ${VENV_DIR}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
