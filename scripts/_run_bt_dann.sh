#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
S12="1,3,5,11,13,23,31,57,99,101,202,777"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_bt_dann
LOG=logs/bt_dann_$(date +%Y%m%d_%H%M%S).log
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
echo "PRASYARAT: scripts/audit/sanity_bt_dann.py harus lolos 11/11 sebelum ini dijalankan."

stage bt_dannsupcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 --method dannsupcon \
    $BT $SC --name BT_dannsupcon_r03

stage bt_dann_s8       $PY scripts/train_motor_dann.py $BASE --seeds $S8 $BT \
    --name BT_dann_r03

# ---- TAHAP 3-4: lengkapi ke n=20 ----
stage bt_dann_s12       $PY scripts/train_motor_dann.py $BASE --seeds $S12 $BT \
    --name BT_dann_r03_s12

stage bt_dannsupcon_s12 $PY scripts/train_motor_dann.py $BASE --seeds $S12 --method dannsupcon \
    $BT $SC --name BT_dannsupcon_r03_s12

echo "" | tee -a "$LOG"
echo "=== SEMUA TAHAP SELESAI ===" | tee -a "$LOG"
echo "Agregasi: $PY scripts/audit/aggregate_seeds.py" | tee -a "$LOG"
echo "Periksa rasio: grep 'BT-DANN.*-BAL ep' $LOG | tail -20" | tee -a "$LOG"
