#!/usr/bin/env bash

set -u
set -o pipefail   # WAJIB — lihat catatan di bawah
cd "$(dirname "$0")/.."


PY=${PYTHON:-python}
S5="42,123,2025,7,17"          # IDENTIK dengan _run_lodo.sh, supaya berpasangan dengan ERM
NPZ="motor_multiclass_w4096_s4096_ss_env.npz"
BASE="--backbone tsitplus --no_augment --fixed_budget --fp32 --epochs 50 --n_folds 8"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_lodo_metode
LOG="logs/lodo_metode_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$DONE" logs

if [ ! -f "data/processed/motor/$NPZ" ]; then
  echo "[ABORT] $NPZ belum ada."; exit 1
fi

stage() {
    local name="$1"; shift
    if [ -f "$DONE/$name" ]; then
        echo "=== [LEWATI] $name — sudah selesai $(cat "$DONE/$name")" | tee -a "$LOG"
        return 0
    fi
    echo "" | tee -a "$LOG"
    echo "############ $name ############" | tee -a "$LOG"
    local t0=$(date +%s)
    if "$@" 2>&1 | tee -a "$LOG"; then
        date '+%Y-%m-%d %H:%M:%S' > "$DONE/$name"
        echo "=== [SELESAI] $name dalam $((($(date +%s)-t0)/60)) menit" | tee -a "$LOG"
    else
        echo "=== [GAGAL] $name — penanda TIDAK ditulis, akan diulang" | tee -a "$LOG"
        return 1
    fi
}

echo "log: $LOG"
echo "LODO 8 lipatan x 5 seed, tiga metode terkuat. Seed $S5" | tee -a "$LOG"

for f in 0 1 2 3 4 5 6 7; do
  stage "lodo_supcon_f${f}" $PY scripts/train_motor_dann.py $BASE \
      --seeds "$S5" --npz "$NPZ" --method supcon $SC \
      --fold_idx "$f" --name "LODOM_supcon" || exit 1
done

for f in 0 1 2 3 4 5 6 7; do
  stage "lodo_dannsupcon_f${f}" $PY scripts/train_motor_dann.py $BASE \
      --seeds "$S5" --npz "$NPZ" --method dannsupcon --lambda_max 1.0 $SC \
      --fold_idx "$f" --name "LODOM_dannsupcon" || exit 1
done

for f in 0 1 2 3 4 5 6 7; do
  stage "lodo_btdidx_f${f}" $PY scripts/train_motor_dann.py $BASE \
      --seeds "$S5" --npz "$NPZ" --method dannsupcon $BT $SC --domain_index f2s \
      --fold_idx "$f" --name "LODOM_BT_didx" || exit 1
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Pembanding ERM LODO sudah ada (prefix LODO_baseline, seed $S5)." | tee -a "$LOG"
echo "Analisis: $PY scripts/audit/audit_lodo.py" | tee -a "$LOG"
