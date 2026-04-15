#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

choose_python_bin() {
  if [[ -n "${PEAGLE_PYTHON_BIN:-}" ]]; then
    if command -v "${PEAGLE_PYTHON_BIN}" >/dev/null 2>&1; then
      printf '%s\n' "${PEAGLE_PYTHON_BIN}"
      return 0
    fi
    echo "PEAGLE_PYTHON_BIN is set but not executable on PATH: ${PEAGLE_PYTHON_BIN}" >&2
    exit 1
  fi

  local candidate
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "Could not find a usable python3 interpreter on PATH." >&2
  exit 1
}

python_version_ok() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'
}

python_version_string() {
  local python_bin="$1"
  "${python_bin}" -c 'import sys; print(".".join(str(x) for x in sys.version_info[:3]))'
}

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

PYTHON_BIN="$(choose_python_bin)"
PYTHON_VERSION="$(python_version_string "${PYTHON_BIN}")"

if ! python_version_ok "${PYTHON_BIN}"; then
  cat >&2 <<EOF
peagle-q requires Python 3.10+ but detected ${PYTHON_BIN} (${PYTHON_VERSION}).

On Sol, load a newer Python first, then rerun this script. For example:
  module avail python
  module load python/3.10

Or point the bootstrap at a specific binary:
  export PEAGLE_PYTHON_BIN=python3.10
EOF
  exit 1
fi

if [[ -f "${PEAGLE_VENV}/bin/python" ]]; then
  EXISTING_VENV_VERSION="$(python_version_string "${PEAGLE_VENV}/bin/python")"
  if ! python_version_ok "${PEAGLE_VENV}/bin/python"; then
    cat >&2 <<EOF
Existing virtual environment at ${PEAGLE_VENV} uses Python ${EXISTING_VENV_VERSION}, which is too old for this project.

Delete that environment after loading Python 3.10+, then rerun:
  rm -rf "${PEAGLE_VENV}"
  bash "${PEAGLE_ROOT}/scripts/sol/setup_env.sh"
EOF
    exit 1
  fi
fi

if [[ ! -f "${PEAGLE_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${PEAGLE_VENV}"
fi

source "${PEAGLE_VENV}/bin/activate"
if [[ "${PEAGLE_SKIP_PIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -e "${PEAGLE_ROOT}[research,quant,dev]"

  # gptqmodel is a transitive dependency of autoawq but is NOT needed for AWQ
  # inference.  Version 6.x crashes on Sol at import time with:
  #   RuntimeError: Tensor.item() cannot be called on meta tensors
  # because exllamav3_torch.py runs scalar ops during module-level init,
  # which conflicts with Sol's sitecustomize.py wrapper.
  # We load AWQ models via AutoAWQForCausalLM.from_quantized() directly,
  # so gptqmodel is not required.
  python -m pip uninstall -y gptqmodel 2>/dev/null || true
fi

cat <<EOF
Environment ready.
PEAGLE_PYTHON_BIN=${PYTHON_BIN}
PEAGLE_PYTHON_VERSION=${PYTHON_VERSION}
PEAGLE_ROOT=${PEAGLE_ROOT}
PEAGLE_VENV=${PEAGLE_VENV}
PEAGLE_RUN_ROOT=${PEAGLE_RUN_ROOT}
PEAGLE_CACHE_ROOT=${PEAGLE_CACHE_ROOT}
PEAGLE_BENCHMARK=${PEAGLE_BENCHMARK}
HF_HOME=${HF_HOME}
EOF
