#!/usr/bin/env bash
set -u
set -o pipefail

PY=${PYTHON:-python}
S5="42,123,2025,7,17"
NPZ="motor_multiclass_w4096_s4096_ss_env.npz"

# IDENTIK dengan BASE seluruh tabel, HANYA --extremes_out yang dibuang.
BASE="--backbone tsitplus --no_augment --fixed_budget --fp32 --epochs 50 --n_folds 8"

DONE=logs/.done_lodo
mkdir -p "$DONE"
LOG="logs/lodo_$(date +%Y%m%d_%H%M%S).log"

stage() {
    local name="$1"; shift
    if [ -f "$DONE/$name" ]; then
        echo "=== [LEWATI] $name (sudah selesai)" | tee -a "$LOG"
        return 0
    fi
    echo "=== [MULAI] $name" | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG"; then
        touch "$DONE/$name"
        echo "=== [SELESAI] $name" | tee -a "$LOG"
    else
        echo "=== [GAGAL] $name — penanda TIDAK ditulis, akan diulang" | tee -a "$LOG"
        return 1
    fi
}

echo "log: $LOG"
echo "LODO 8 lipatan x 5 seed, ERM saja (lambda_max 0.0)" | tee -a "$LOG"

for f in 0 1 2 3 4 5 6 7; do
    stage "lodo_fold${f}" $PY scripts/train_motor_dann.py $BASE \
        --seeds "$S5" --npz "$NPZ" --lambda_max 0.0 \
        --fold_idx "$f" --name "LODO_baseline" || exit 1
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Analisis: $PY scripts/audit/audit_lodo.py" | tee -a "$LOG"
