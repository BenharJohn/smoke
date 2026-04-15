#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 MODEL_PATH [RUN_NAME] [PILOT_EXAMPLES] [DEVICE]" >&2
  exit 1
fi

MODEL_PATH="$1"
RUN_NAME="${2:-bigmodel}"
PILOT_EXAMPLES="${3:-32}"
DEVICE="${4:-cuda}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export KMP_DUPLICATE_LIB_OK=TRUE

EXTRACT3_CONFIG="configs/_generated_extract_${RUN_NAME}_3prompt.json"
EXTRACT_PILOT_CONFIG="configs/_generated_extract_${RUN_NAME}_pilot.json"
TRAIN_SMOKE_CONFIG="configs/_generated_train_${RUN_NAME}_smoke.json"

EXTRACT3_OUTPUT="runs/extract_${RUN_NAME}_3prompt"
EXTRACT_PILOT_OUTPUT="runs/extract_${RUN_NAME}_pilot"
TRAIN_SMOKE_OUTPUT="runs/train_${RUN_NAME}_smoke"

cat > "$EXTRACT3_CONFIG" <<EOF
{
  "dataset": "data/gemma_smoke_prompts.jsonl",
  "teacher_model_path": "${MODEL_PATH}",
  "teacher_quantization": "none",
  "output_dir": "${EXTRACT3_OUTPUT}",
  "max_examples": 3,
  "max_length": 256,
  "device": "${DEVICE}"
}
EOF

cat > "$EXTRACT_PILOT_CONFIG" <<EOF
{
  "dataset": "data/gemma_smoke_prompts.jsonl",
  "teacher_model_path": "${MODEL_PATH}",
  "teacher_quantization": "none",
  "output_dir": "${EXTRACT_PILOT_OUTPUT}",
  "max_examples": ${PILOT_EXAMPLES},
  "max_length": 512,
  "device": "${DEVICE}"
}
EOF

cat > "$TRAIN_SMOKE_CONFIG" <<EOF
{
  "manifest": "${EXTRACT3_OUTPUT}/manifest.jsonl",
  "output_dir": "${TRAIN_SMOKE_OUTPUT}",
  "projection_model_path": "${MODEL_PATH}",
  "epochs": 1,
  "max_examples": 3,
  "batch_size": 1,
  "grad_accum_steps": 1,
  "learning_rate": 0.0001,
  "checkpoint_every": 1,
  "device": "${DEVICE}",
  "dtype": "float16"
}
EOF

echo
echo "Step 1: 3-prompt extraction"
python -m peagle_q.cli extract --config "$EXTRACT3_CONFIG"

MANIFEST_PATH="${EXTRACT3_OUTPUT}/manifest.jsonl"
TENSOR_DIR="${EXTRACT3_OUTPUT}/tensors"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Expected manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi
if [[ ! -d "$TENSOR_DIR" ]]; then
  echo "Expected tensor directory not found: $TENSOR_DIR" >&2
  exit 1
fi

TENSOR_COUNT="$(find "$TENSOR_DIR" -maxdepth 1 -type f -name '*.pt' | wc -l | tr -d ' ')"
if [[ "$TENSOR_COUNT" -lt 3 ]]; then
  echo "Expected at least 3 tensor files, found $TENSOR_COUNT" >&2
  exit 1
fi

echo
echo "Verified first step:"
echo "  manifest: $MANIFEST_PATH"
echo "  tensors : $TENSOR_DIR ($TENSOR_COUNT files)"

echo
echo "Next commands:"
echo "  1) Scale extraction:"
echo "     python -m peagle_q.cli extract --config $EXTRACT_PILOT_CONFIG"
echo "  2) Run tiny training smoke:"
echo "     python -m peagle_q.cli train --config $TRAIN_SMOKE_CONFIG"
