#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACCOUNT=""
PARTITION=""
QOS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SBATCH_ARGS=(--chdir="${ROOT}")
if [[ -n "${ACCOUNT}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi
if [[ -n "${PARTITION}" ]]; then
  SBATCH_ARGS+=(--partition="${PARTITION}")
fi
if [[ -n "${QOS}" ]]; then
  SBATCH_ARGS+=(--qos="${QOS}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  SBATCH_ARGS+=("${EXTRA_ARGS[@]}")
fi

submit_job() {
  local dependency="$1"
  local command="$2"
  local config="$3"
  local time_limit="$4"

  local args=("${SBATCH_ARGS[@]}" --gpus=1 --time="${time_limit}")
  if [[ -n "${dependency}" ]]; then
    args+=(--dependency="afterok:${dependency}")
  fi

  local output
  output=$(sbatch "${args[@]}" \
    --export=ALL,PEAGLE_COMMAND="${command}",PEAGLE_CONFIG="${config}" \
    "${ROOT}/scripts/sol/run_peagle_job.slurm")

  local job_id
  job_id=$(awk '{print $NF}' <<<"${output}")
  echo "${job_id}"
}

echo "Submitting 1k pilot chain from ${ROOT}"

BF16_SMOKE_JOB=$(submit_job "" "smoke" "${ROOT}/configs/sol/qwen3_8b_smoke.json" "01:00:00")
echo "smoke_bf16: ${BF16_SMOKE_JOB}"

AWQ_SMOKE_JOB=$(submit_job "" "smoke" "${ROOT}/configs/sol/qwen3_8b_smoke_awq.json" "01:00:00")
echo "smoke_awq: ${AWQ_SMOKE_JOB}"

BF16_EXTRACT_JOB=$(submit_job "${BF16_SMOKE_JOB}" "extract" "${ROOT}/configs/sol/qwen3_8b_extract_bf16_1k.json" "02:00:00")
echo "extract_bf16_1k: ${BF16_EXTRACT_JOB}"

AWQ_EXTRACT_JOB=$(submit_job "${AWQ_SMOKE_JOB}" "extract" "${ROOT}/configs/sol/qwen3_8b_extract_awq_1k.json" "02:00:00")
echo "extract_awq_1k: ${AWQ_EXTRACT_JOB}"

BF16_TRAIN_JOB=$(submit_job "${BF16_EXTRACT_JOB}" "train" "${ROOT}/configs/sol/qwen3_8b_train_bf16_1k.json" "04:00:00")
echo "train_bf16_1k: ${BF16_TRAIN_JOB}"

AWQ_TRAIN_JOB=$(submit_job "${AWQ_EXTRACT_JOB}" "train" "${ROOT}/configs/sol/qwen3_8b_train_awq_1k.json" "04:00:00")
echo "train_awq_1k: ${AWQ_TRAIN_JOB}"

EVAL_BF16_BF16_JOB=$(submit_job "${BF16_TRAIN_JOB}" "evaluate" "${ROOT}/configs/sol/qwen3_8b_eval_bf16_on_bf16.json" "02:00:00")
echo "eval_bf16_on_bf16: ${EVAL_BF16_BF16_JOB}"

EVAL_BF16_AWQ_JOB=$(submit_job "${BF16_TRAIN_JOB}:${AWQ_EXTRACT_JOB}" "evaluate" "${ROOT}/configs/sol/qwen3_8b_eval_bf16_on_awq.json" "02:00:00")
echo "eval_bf16_on_awq: ${EVAL_BF16_AWQ_JOB}"

EVAL_AWQ_AWQ_JOB=$(submit_job "${AWQ_TRAIN_JOB}" "evaluate" "${ROOT}/configs/sol/qwen3_8b_eval_awq_on_awq.json" "02:00:00")
echo "eval_awq_on_awq: ${EVAL_AWQ_AWQ_JOB}"

echo ""
echo "All jobs submitted."
echo "Run 'squeue -u ${USER}' to watch progress."
