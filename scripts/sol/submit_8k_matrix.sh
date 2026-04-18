#!/usr/bin/env bash
# submit_8k_matrix.sh — submit the full 8k training + evaluation matrix.
#
# Usage:
#   bash scripts/sol/submit_8k_matrix.sh \
#       --account YOUR_ACCOUNT \
#       --partition YOUR_PARTITION \
#       [--qos YOUR_QOS] \
#       [--w8a8]          # also run the W8A8 (H2/H3) chain
#
# Dependency chain:
#   smoke_{bf16,awq}
#     → extract_{bf16,awq}_8k
#       → train_{bf16,awq}_8k  (seed 42, 1, 7 in parallel)
#         → eval_{bf16_on_bf16, bf16_on_awq, awq_on_awq}_8k
#
# W8A8 chain (--w8a8 flag, requires PEAGLE_W8A8_MODEL to be set):
#   smoke_w8a8
#     → extract_w8a8_8k
#       → train_w8a8_8k
#         → eval_w8a8_on_w8a8
#   train_bf16_8k (above) → eval_bf16_on_w8a8
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACCOUNT=""
PARTITION=""
QOS=""
RUN_W8A8=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)   ACCOUNT="$2";    shift 2 ;;
    --partition) PARTITION="$2";  shift 2 ;;
    --qos)       QOS="$2";        shift 2 ;;
    --w8a8)      RUN_W8A8=true;   shift   ;;
    *)           EXTRA_ARGS+=("$1"); shift ;;
  esac
done

SBATCH_ARGS=(--chdir="${ROOT}")
[[ -n "${ACCOUNT}"   ]] && SBATCH_ARGS+=(--account="${ACCOUNT}")
[[ -n "${PARTITION}" ]] && SBATCH_ARGS+=(--partition="${PARTITION}")
[[ -n "${QOS}"       ]] && SBATCH_ARGS+=(--qos="${QOS}")
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && SBATCH_ARGS+=("${EXTRA_ARGS[@]}")

# Submits one job; returns the Slurm job ID.
# Usage: submit_job <dependency_expr|""> <command> <config> <time>
submit_job() {
  local dependency="$1" command="$2" config="$3" time_limit="$4"
  local args=("${SBATCH_ARGS[@]}" --gpus=1 "--time=${time_limit}")
  [[ -n "${dependency}" ]] && args+=("--dependency=afterok:${dependency}")
  local output
  output=$(sbatch "${args[@]}" \
    --export=ALL,PEAGLE_COMMAND="${command}",PEAGLE_CONFIG="${config}" \
    "${ROOT}/scripts/sol/run_peagle_job.slurm")
  awk '{print $NF}' <<<"${output}"
}

C="${ROOT}/configs/sol"
echo "Submitting 8k matrix from ${ROOT}"
echo ""

# ── Phase 1: smoke ────────────────────────────────────────────────────────────
BF16_SMOKE=$(submit_job "" smoke "${C}/qwen3_8b_smoke.json" "01:00:00")
echo "smoke_bf16:          ${BF16_SMOKE}"

AWQ_SMOKE=$(submit_job "" smoke "${C}/qwen3_8b_smoke_awq.json" "01:00:00")
echo "smoke_awq:           ${AWQ_SMOKE}"

# ── Phase 2: extract ──────────────────────────────────────────────────────────
BF16_EXTRACT=$(submit_job "${BF16_SMOKE}" extract "${C}/qwen3_8b_extract_bf16_8k.json" "06:00:00")
echo "extract_bf16_8k:     ${BF16_EXTRACT}"

AWQ_EXTRACT=$(submit_job "${AWQ_SMOKE}" extract "${C}/qwen3_8b_extract_awq_8k.json" "06:00:00")
echo "extract_awq_8k:      ${AWQ_EXTRACT}"

# ── Phase 3: train (seed 42 + two replication seeds in parallel) ──────────────
BF16_TRAIN=$(submit_job "${BF16_EXTRACT}" train "${C}/qwen3_8b_train_bf16_8k.json" "08:00:00")
echo "train_bf16_8k:       ${BF16_TRAIN}"

BF16_TRAIN_S1=$(submit_job "${BF16_EXTRACT}" train "${C}/qwen3_8b_train_bf16_8k_seed1.json" "08:00:00")
echo "train_bf16_8k_seed1: ${BF16_TRAIN_S1}"

BF16_TRAIN_S7=$(submit_job "${BF16_EXTRACT}" train "${C}/qwen3_8b_train_bf16_8k_seed7.json" "08:00:00")
echo "train_bf16_8k_seed7: ${BF16_TRAIN_S7}"

AWQ_TRAIN=$(submit_job "${AWQ_EXTRACT}" train "${C}/qwen3_8b_train_awq_8k.json" "08:00:00")
echo "train_awq_8k:        ${AWQ_TRAIN}"

AWQ_TRAIN_S1=$(submit_job "${AWQ_EXTRACT}" train "${C}/qwen3_8b_train_awq_8k_seed1.json" "08:00:00")
echo "train_awq_8k_seed1:  ${AWQ_TRAIN_S1}"

AWQ_TRAIN_S7=$(submit_job "${AWQ_EXTRACT}" train "${C}/qwen3_8b_train_awq_8k_seed7.json" "08:00:00")
echo "train_awq_8k_seed7:  ${AWQ_TRAIN_S7}"

# ── Phase 4: eval (speculation_mode=both → sequential τ/α + parallel speedup) ─
EVAL_BF16_BF16=$(submit_job "${BF16_TRAIN}" evaluate "${C}/qwen3_8b_eval_bf16_on_bf16_8k.json" "02:00:00")
echo "eval_bf16_on_bf16_8k: ${EVAL_BF16_BF16}"

EVAL_BF16_AWQ=$(submit_job "${BF16_TRAIN}" evaluate "${C}/qwen3_8b_eval_bf16_on_awq_8k.json" "02:00:00")
echo "eval_bf16_on_awq_8k:  ${EVAL_BF16_AWQ}"

EVAL_AWQ_AWQ=$(submit_job "${AWQ_TRAIN}" evaluate "${C}/qwen3_8b_eval_awq_on_awq_8k.json" "02:00:00")
echo "eval_awq_on_awq_8k:   ${EVAL_AWQ_AWQ}"

# ── Optional W8A8 chain (H2/H3 hypotheses) ───────────────────────────────────
if [[ "${RUN_W8A8}" == "true" ]]; then
  if [[ -z "${PEAGLE_W8A8_MODEL:-}" ]]; then
    echo ""
    echo "ERROR: --w8a8 requires PEAGLE_W8A8_MODEL to be set."
    echo "  export PEAGLE_W8A8_MODEL=<hf-repo-or-path>"
    echo "  echo \"export PEAGLE_W8A8_MODEL=\$PEAGLE_W8A8_MODEL\" >> ~/.bashrc"
    exit 1
  fi

  W8A8_SMOKE=$(submit_job "" smoke "${C}/qwen3_8b_smoke_w8a8.json" "01:00:00")
  echo "smoke_w8a8:           ${W8A8_SMOKE}"

  W8A8_EXTRACT=$(submit_job "${W8A8_SMOKE}" extract "${C}/qwen3_8b_extract_w8a8_8k.json" "06:00:00")
  echo "extract_w8a8_8k:      ${W8A8_EXTRACT}"

  W8A8_TRAIN=$(submit_job "${W8A8_EXTRACT}" train "${C}/qwen3_8b_train_w8a8_8k.json" "08:00:00")
  echo "train_w8a8_8k:        ${W8A8_TRAIN}"

  # H2: BF16-drafter on W8A8-verifier (drafter already trained above)
  EVAL_BF16_W8A8=$(submit_job "${BF16_TRAIN}" evaluate "${C}/qwen3_8b_eval_bf16_on_w8a8.json" "02:00:00")
  echo "eval_bf16_on_w8a8:    ${EVAL_BF16_W8A8}"

  # H3: W8A8-drafter on W8A8-verifier
  EVAL_W8A8_W8A8=$(submit_job "${W8A8_TRAIN}" evaluate "${C}/qwen3_8b_eval_w8a8_on_w8a8.json" "02:00:00")
  echo "eval_w8a8_on_w8a8:    ${EVAL_W8A8_W8A8}"
fi

echo ""
echo "All jobs submitted. Watch with:"
echo "  squeue -u \${USER}"
echo ""
echo "Key result files (after jobs complete):"
echo "  \$PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_bf16_8k.json"
echo "  \$PEAGLE_RUN_ROOT/qwen3_8b/eval_bf16_on_awq_8k.json"
echo "  \$PEAGLE_RUN_ROOT/qwen3_8b/eval_awq_on_awq_8k.json"
