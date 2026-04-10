#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
Usage:
  $0 <branch> [remote] [submit_1k_pilot args...]

Examples:
  $0 my-branch
  $0 my-branch origin --account acct --partition gpu
EOF
  exit 1
fi

BRANCH="$1"
shift

REMOTE="origin"
if [[ $# -gt 0 && "$1" != --* ]]; then
  REMOTE="$1"
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${ROOT}/scripts/sol/bootstrap_branch.sh" "${BRANCH}" "${REMOTE}"
bash "${ROOT}/scripts/sol/submit_1k_pilot.sh" "$@"
