#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<EOF
Usage:
  $0 <branch> <command> <config> [remote]

Examples:
  $0 my-branch smoke configs/sol/qwen3_8b_smoke.json
  $0 my-branch train configs/sol/qwen3_8b_train_bf16_1k.json
EOF
  exit 1
fi

BRANCH="$1"
COMMAND="$2"
CONFIG="$3"
REMOTE="${4:-origin}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${ROOT}/scripts/sol/bootstrap_branch.sh" "${BRANCH}" "${REMOTE}"

export PEAGLE_ROOT="${PEAGLE_ROOT:-$ROOT}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${ROOT}/${CONFIG}"
fi

echo "Running ${COMMAND} with ${CONFIG}"
bash "${ROOT}/scripts/sol/run_cli.sh" "${COMMAND}" --config "${CONFIG}"
