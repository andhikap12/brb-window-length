#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_domain_index
LOG=logs/domain_index_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

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

stage didx_f2s_bt_supcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --method dannsupcon $BT $SC --domain_index f2s --name DIDX_f2s_BT_dannsupcon


echo "" | tee -a "$LOG"
echo "=== TAHAP n=8 SELESAI ===" | tee -a "$LOG"
echo "A-P2/A-P4 dibaca dari log: grep 'GRL-REG-EPOCH\\|corr(D(z),u)' $LOG | tail -60" | tee -a "$LOG"
echo "A-P1/A-P3 butuh audit: $PY scripts/audit/audit_ckpt_ext.py (arahkan ke ckpt DIDX)" | tee -a "$LOG"
echo "Baca terhadap A-P1..A-P4 di PREREGISTRATION.md SEBELUM memutuskan n=20." | tee -a "$LOG"
