#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s4096_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_w16384ovl
LOG=logs/w16384ovl_$(date +%Y%m%d_%H%M%S).log
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

stage erm_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384OVL_baseline

S12="1,3,5,11,13,23,31,57,99,101,202,777"

stage erm_w16384ovl_s12 $PY scripts/train_motor_dann.py $BASE --seeds $S12 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384OVL_baseline_s12

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku dengan dataset yang COCOK:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py --methods 'UJI E: ERM w16384-ovl' \\" | tee -a "$LOG"
echo "      --npz $NPZ --seeds $S8 --out results/motor/audit_erm_w16384ovl_s8.npz" | tee -a "$LOG"
