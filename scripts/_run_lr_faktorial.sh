#!/usr/bin/env bash

set -u
set -o pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
S8="42,123,2025,7,17,2024,2026,88"
LR=3e-4
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --lr $LR"

DONE=logs/.done_lr_faktorial
LOG=logs/lr_faktorial_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

# lengan -> berkas npz. Keempatnya sel faktorial R3, identik dengan kampanye TSSequencerPlus.
declare -A NPZ=(
  [w4096sub]="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
  [w16384]="motor_multiclass_w16384_s16384_ss_env.npz"
  [w4096full]="motor_multiclass_w4096_s4096_ss_env.npz"
  [w16384ovl]="motor_multiclass_w16384_s4096_ss_env.npz"
)
declare -A NAMA=(
  [w4096sub]="LR3E4_W4096SUB_baseline"
  [w16384]="LR3E4_W16384_baseline"
  [w4096full]="LR3E4_W4096FULL_baseline"
  [w16384ovl]="LR3E4_W16384OVL_baseline"
)

for k in "${!NPZ[@]}"; do
  if [ ! -f "data/processed/motor/${NPZ[$k]}" ]; then
    echo "[ABORT] ${NPZ[$k]} belum ada."
    exit 1
  fi
done

for k in "${!NAMA[@]}"; do
  case "${NAMA[$k]}" in
    LR3E4_*) ;;
    *) echo "[ABORT] nama '${NAMA[$k]}' tanpa awalan LR3E4_ akan menimpa kampanye lr 1e-3."; exit 1 ;;
  esac
done

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
echo "lr = $LR (pembanding: 1e-3), backbone tsitplus, seed $S8" | tee -a "$LOG"

for k in w4096sub w16384 w4096full w16384ovl; do
  stage "erm_${k}_lr3e4_s8" $PY scripts/train_motor_dann.py $BASE --seeds "$S8" \
      --npz "${NPZ[$k]}" --lambda_max 0.0 --name "${NAMA[$k]}"
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku, satu perintah per lengan:" | tee -a "$LOG"
for k in w4096sub w16384 w4096full w16384ovl; do
  echo "  $PY scripts/audit/audit_ckpt_ext.py --npz ${NPZ[$k]} --seeds $S8 \\" | tee -a "$LOG"
  echo "      --out results/motor/audit_lr3e4_${k}_s8.npz" | tee -a "$LOG"
done
