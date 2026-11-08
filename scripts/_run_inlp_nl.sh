#!/usr/bin/env bash

set -u
set -o pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export PYTHONIOENCODING=utf-8

SUMBER="${1:-ERM}"
LOG="logs/inlp_nl_${SUMBER,,}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "log: $LOG"
${PYTHON:-python} -u scripts/audit/npdann_inlp.py \
    --source "$SUMBER" \
    --device cpu \
    --out "results/motor/npdann_inlp_nl_${SUMBER,,}.npz" \
    2>&1 | tee -a "$LOG"
