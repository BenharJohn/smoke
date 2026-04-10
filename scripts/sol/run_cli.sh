#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PEAGLE_ROOT="${PEAGLE_ROOT:-$ROOT}"
export PEAGLE_VENV="${PEAGLE_VENV:-$PEAGLE_ROOT/.venv-sol}"
export PEAGLE_BENCHMARK="${PEAGLE_BENCHMARK:-$PEAGLE_ROOT/benchmarks/mini_mt_bench.jsonl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ -z "${PEAGLE_RUN_ROOT:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    export PEAGLE_RUN_ROOT="${SCRATCH}/peagle-q/${USER}"
  else
    export PEAGLE_RUN_ROOT="${PEAGLE_ROOT}/runs"
  fi
fi

if [[ ! -f "${PEAGLE_VENV}/bin/activate" ]]; then
  echo "Missing virtual environment at ${PEAGLE_VENV}. Run scripts/sol/setup_env.sh first." >&2
  exit 1
fi

if [[ -z "${PEAGLE_CACHE_ROOT:-}" ]]; then
  export PEAGLE_CACHE_ROOT="${PEAGLE_RUN_ROOT}/cache"
fi

export HF_HOME="${HF_HOME:-${PEAGLE_CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PEAGLE_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PEAGLE_CACHE_ROOT}/pip}"

source "${PEAGLE_VENV}/bin/activate"
mkdir -p "${PEAGLE_RUN_ROOT}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"
cd "${PEAGLE_ROOT}"

python -m peagle_q.cli "$@"
