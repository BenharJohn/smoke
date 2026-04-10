#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
Usage:
  $0 <branch> <command> <config> [remote] [sbatch args...]

Examples:
  $0 my-branch smoke configs/sol/qwen3_8b_smoke.json origin --account acct --partition gpu --gpus=1
  $0 my-branch train configs/sol/qwen3_8b_train_bf16_1k.json origin --account acct --partition gpu --gpus=1 --time=04:00:00
EOF
  exit 1
fi

BRANCH="$1"
COMMAND="$2"
CONFIG="$3"
shift 3

REMOTE="origin"
if [[ $# -gt 0 && "$1" != --* ]]; then
  REMOTE="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${ROOT}/scripts/sol/bootstrap_branch.sh" "${BRANCH}" "${REMOTE}"

export PEAGLE_ROOT="${PEAGLE_ROOT:-$ROOT}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi

SBATCH_ARGS=(--chdir="${ROOT}")
if [[ $# -gt 0 ]]; then
  SBATCH_ARGS+=("$@")
fi

echo "Submitting ${COMMAND} from branch ${BRANCH}"
sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,PEAGLE_COMMAND="${COMMAND}",PEAGLE_CONFIG="${CONFIG}" \
  "${ROOT}/scripts/sol/run_peagle_job.slurm"
