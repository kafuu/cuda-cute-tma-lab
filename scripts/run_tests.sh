#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${1:-local-sm120-release}"

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}"
ctest --preset "${PRESET}"

if [[ -x "${ROOT}/.venv/bin/pytest" ]]; then
    "${ROOT}/.venv/bin/pytest" "${ROOT}/2_dsl_gemm/tests"
elif command -v pytest >/dev/null 2>&1; then
    pytest "${ROOT}/2_dsl_gemm/tests"
else
    echo "pytest is not installed; Python DSL tests were not run."
fi
