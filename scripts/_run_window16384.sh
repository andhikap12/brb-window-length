#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s16384_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_w16384
LOG=logs/w16384_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

if [ ! -f "data/processed/motor/$NPZ" ]; then
    echo "[ABORT] $NPZ belum ada. Jalankan preprocessing dulu (lihat README)."
    exit 1
fi

stage() {
    local name="$1"; shift
    if [ -f "$DONE/$name" ]; then
        echo "=== [LEWATI] $name — sudah selesai $(cat "$DONE/$name")" | tee -a "$LOG"
        return 0
    fi
    echo "" | tee -a "$LOG"
    echo "############ $name ############" | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG"; then
        date '+%Y-%m-%d %H:%M:%S' > "$DONE/$name"
        echo "=== [SELESAI] $name" | tee -a "$LOG"
    else
        echo "=== [GAGAL] $name — penanda TIDAK ditulis, akan diulang" | tee -a "$LOG"
        return 1
    fi
}

echo "log: $LOG"
echo "penanda selesai: $DONE/"

stage erm_w16384_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384_baseline

echo "" | tee -a "$LOG"
echo "=== TAHAP n=8 SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku, bukan dari log:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py  (tambahkan entri W16384_baseline)" | tee -a "$LOG"
echo "Baca terhadap C-P1..C-P3 di PREREGISTRATION.md SEBELUM menyimpulkan." | tee -a "$LOG"
