#!/usr/bin/env bash
set -euo pipefail

START_IDX="${1:-0}"
COUNT="${2:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-./results/humanml/ELMAv3_anap_hml3d_hist/model000550000.pt}"
EVAL_FILE="${EVAL_FILE:-./dataset/HumanML3D/humanml_test_set_anaphora_fullgen.json}"
OUT_ROOT="${OUT_ROOT:-./qualitative/anaphora_fullgen_600f}"
GUIDANCE_PARAM="${GUIDANCE_PARAM:-2.5}"
HISTORY_CURRENT_WEIGHT="${HISTORY_CURRENT_WEIGHT:-0.5}"
USE_HISTORY_TEXT="${USE_HISTORY_TEXT:-1}"

mkdir -p "$OUT_ROOT"

END_IDX=$((START_IDX + COUNT - 1))
echo "Generating ${COUNT} samples from indices ${START_IDX}..${END_IDX}"
echo "Model: ${MODEL_PATH}"
echo "Eval file: ${EVAL_FILE}"
echo "Output root: ${OUT_ROOT}"
echo "Use history text: ${USE_HISTORY_TEXT}"

GEN_ARGS=(
  -m runners.generate
  --model_path "$MODEL_PATH"
  --dataset humanml
  --num_repetitions 1
  --guidance_param "$GUIDANCE_PARAM"
  --use_chunked_att
)

if [[ "$USE_HISTORY_TEXT" == "1" ]]; then
  GEN_ARGS+=(
    --use_history_text
    --history_current_weight "$HISTORY_CURRENT_WEIGHT"
  )
fi

for idx in $(seq "$START_IDX" "$END_IDX"); do
  instr_path="${OUT_ROOT}/instruction_${idx}.json"
  sample_out="${OUT_ROOT}/sample_${idx}"

  "$PYTHON_BIN" - <<'PY' "$idx" "$instr_path" "$EVAL_FILE"
import json
import sys

idx = int(sys.argv[1])
out_path = sys.argv[2]
eval_file = sys.argv[3]

with open(eval_file) as f:
    data = json.load(f)

if not 0 <= idx < len(data):
    raise IndexError(f"idx={idx} is out of range for {eval_file} (size={len(data)})")

item = data[idx]
payload = {
    "text": item["text"],
    "lengths": [100, 100, 100, 100, 100, 100],
    "history_text": item["history_text"],
}

with open(out_path, "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"idx={idx} id={item['id']}")
print(f"wrote {out_path}")
PY

  "$PYTHON_BIN" "${GEN_ARGS[@]}" \
    --instructions_file "$instr_path" \
    --output_dir "$sample_out"
done

echo "Done. Outputs are under ${OUT_ROOT}"
