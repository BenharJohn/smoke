#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <branch> [remote]" >&2
  exit 1
fi

BRANCH="$1"
REMOTE="${2:-origin}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT}"

if [[ ! -d .git ]]; then
  echo "This directory is not a git repository: ${ROOT}" >&2
  exit 1
fi

echo "Fetching ${REMOTE}..."
git fetch "${REMOTE}" --prune

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "Checking out existing local branch ${BRANCH}"
  git checkout "${BRANCH}"
else
  echo "Creating local branch ${BRANCH} from ${REMOTE}/${BRANCH}"
  git checkout -b "${BRANCH}" --track "${REMOTE}/${BRANCH}"
fi

echo "Pulling latest ${REMOTE}/${BRANCH}"
git pull --ff-only "${REMOTE}" "${BRANCH}"

echo "Refreshing environment"
bash "${ROOT}/scripts/sol/setup_env.sh"

echo ""
echo "Branch is ready."
echo "Current branch: $(git branch --show-current)"
