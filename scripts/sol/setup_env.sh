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

mkdir -p "${PEAGLE_RUN_ROOT}"

if [[ -z "${PEAGLE_CACHE_ROOT:-}" ]]; then
  export PEAGLE_CACHE_ROOT="${PEAGLE_RUN_ROOT}/cache"
fi

export HF_HOME="${HF_HOME:-${PEAGLE_CACHE_ROOT}/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PEAGLE_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PEAGLE_CACHE_ROOT}/pip}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}"

if [[ ! -d "${PEAGLE_VENV}" ]]; then
  python3 -m venv "${PEAGLE_VENV}"
fi

source "${PEAGLE_VENV}/bin/activate"
if [[ "${PEAGLE_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --upgrade pip
  python -m pip install -e "${PEAGLE_ROOT}[research,quant,dev]"
fi

cat <<EOF
Environment ready.
PEAGLE_ROOT=${PEAGLE_ROOT}
PEAGLE_VENV=${PEAGLE_VENV}
PEAGLE_RUN_ROOT=${PEAGLE_RUN_ROOT}
PEAGLE_CACHE_ROOT=${PEAGLE_CACHE_ROOT}
PEAGLE_BENCHMARK=${PEAGLE_BENCHMARK}
HF_HOME=${HF_HOME}
EOF
