#!/bin/bash
#SBATCH --job-name=peagle-q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${PEAGLE_COMMAND:-}" ]]; then
  echo "Set PEAGLE_COMMAND in the sbatch export list." >&2
  exit 1
fi

if [[ -z "${PEAGLE_CONFIG:-}" ]]; then
  echo "Set PEAGLE_CONFIG in the sbatch export list." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${ROOT}/logs"

if [[ -n "${PEAGLE_ROOT:-}" ]]; then
  cd "${PEAGLE_ROOT}"
else
  cd "${ROOT}"
fi

echo "PEAGLE_COMMAND=${PEAGLE_COMMAND}"
echo "PEAGLE_CONFIG=${PEAGLE_CONFIG}"
echo "PEAGLE_ROOT=${PEAGLE_ROOT:-$ROOT}"
echo "PEAGLE_RUN_ROOT=${PEAGLE_RUN_ROOT:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"${SCRIPT_DIR}/run_cli.sh" "${PEAGLE_COMMAND}" --config "${PEAGLE_CONFIG}"
