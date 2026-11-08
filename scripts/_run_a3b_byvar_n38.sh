#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

# 12 seed yang lengan A sudah punya (dari S14) tetapi lengan B belum
S_B_KEJAR="1,3,5,11,13,23,31,57,99,101,202,777"
# 18 seed baru, dipakai KEDUA lengan supaya berpasangan
S_BARU="301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318"

BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_a3b
mkdir -p "$DONE" logs
LOG=logs/a3b_byvar_$(date +%Y%m%d_%H%M%S).log

stage() {
    local name="$1"; shift
    if [ -f "$DONE/$name" ]; then
        echo "=== [LEWATI] $name — selesai $(cat "$DONE/$name")" | tee -a "$LOG"
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
echo "A3b — by_var=True di n=38. Kriteria A3b-P1..P3 terkunci di komentar skrip ini."

# ---- lengan B (by_var) mengejar 12 seed yang lengan A sudah punya ----
stage b_kejar12  $PY scripts/train_motor_dann.py $BASE --seeds "$S_B_KEJAR" --by_var \
    --lambda_max 0.0 --name BYVAR_baseline

# ---- 18 seed baru, KEDUA lengan ----
stage b_baru18   $PY scripts/train_motor_dann.py $BASE --seeds "$S_BARU" --by_var \
    --lambda_max 0.0 --name BYVAR_baseline

stage a_baru18   $PY scripts/train_motor_dann.py $BASE --seeds "$S_BARU" \
    --lambda_max 0.0 --name S20_baseline

echo "" | tee -a "$LOG"
echo "=== SELESAI === total 48 run baru -> n=38 berpasangan" | tee -a "$LOG"
echo "Analisis: $PY scripts/audit/audit_a3b.py" | tee -a "$LOG"
