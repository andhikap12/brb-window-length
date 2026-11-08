#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s4096_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_w16384method
LOG=logs/w16384method_$(date +%Y%m%d_%H%M%S).log
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

stage supcon_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --method supcon $SC --name W16384OVL_supcon

# ---- TAHAP 2: BT-DANN+SupCon r0.3 — lengan BT, Sec. IV-A makalah ----
stage bt_dannsupcon_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --method dannsupcon $BT $SC --name W16384OVL_BT_dannsupcon_r03

echo "" | tee -a "$LOG"
echo "=== SEMUA TAHAP SELESAI ===" | tee -a "$LOG"
echo "Per-torsi + probe WAJIB dari audit backbone-beku dengan dataset yang COCOK:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py \\" | tee -a "$LOG"
echo "      --methods 'UJI E: ERM w16384-ovl;UJI F: SupCon w16384-ovl;UJI F: BT+SupCon w16384-ovl' \\" | tee -a "$LOG"
echo "      --npz $NPZ --seeds $S8 --out results/motor/audit_method_w16384ovl_s8.npz" | tee -a "$LOG"
echo "Periksa rasio BT: grep 'BT-DANN.*-BAL ep' $LOG | tail -20" | tee -a "$LOG"
