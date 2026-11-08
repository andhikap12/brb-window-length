#!/usr/bin/env bash
set -u
set -o pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
  echo "Pemakaian: bash scripts/_run_hp_faktorial.sh <backbone> <lr> [awalan]"
  echo "  backbone: tsitplus | tssequencerplus"
  exit 2
fi

BACKBONE="$1"
LR="$2"

case "$BACKBONE" in
  tsitplus|tssequencerplus) ;;
  *) echo "[ABORT] backbone '$BACKBONE' tidak dikenal registry."; exit 1 ;;
esac

AWALAN="${3:-$(echo "HP_${BACKBONE}_lr${LR}" | tr '.-' '_')}"
TAG="$(echo "$AWALAN" | tr 'A-Z' 'a-z')"

PY=${PYTHON:-python}
S8="42,123,2025,7,17,2024,2026,88"
BASE="--backbone $BACKBONE --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --lr $LR"

DONE="logs/.done_hp_${TAG}"
LOG="logs/hp_${TAG}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$DONE" logs

declare -A NPZ=(
  [w4096sub]="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
  [w16384]="motor_multiclass_w16384_s16384_ss_env.npz"
  [w4096full]="motor_multiclass_w4096_s4096_ss_env.npz"
  [w16384ovl]="motor_multiclass_w16384_s4096_ss_env.npz"
)

for k in "${!NPZ[@]}"; do
  if [ ! -f "data/processed/motor/${NPZ[$k]}" ]; then
    echo "[ABORT] ${NPZ[$k]} belum ada."
    exit 1
  fi
done

case "$AWALAN" in
  W4096SUB_baseline|W16384_baseline|W4096FULL_baseline|W16384OVL_baseline|TSSEQ_*)
    echo "[ABORT] awalan '$AWALAN' milik kampanye lr 1e-3 yang sudah ada. Pilih yang lain."
    exit 1 ;;
esac

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
echo "sel: backbone=$BACKBONE lr=$LR awalan=$AWALAN" | tee -a "$LOG"
echo "seed: $S8" | tee -a "$LOG"

# Termurah ke termahal: salah konfigurasi terlihat dalam menit pertama, bukan setelah sejam.
for k in w4096sub w16384 w4096full w16384ovl; do
  stage "erm_${k}_${TAG}" $PY scripts/train_motor_dann.py $BASE --seeds "$S8" \
      --npz "${NPZ[$k]}" --lambda_max 0.0 --name "${AWALAN}_${k}"
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Audit per lengan WAJIB dijalankan sebelum angkanya dibaca:" | tee -a "$LOG"
for k in w4096sub w16384 w4096full w16384ovl; do
  echo "  $PY scripts/audit/audit_ckpt_ext.py --npz ${NPZ[$k]} --seeds $S8 \\" | tee -a "$LOG"
  echo "      --out results/motor/audit_${TAG}_${k}_s8.npz" | tee -a "$LOG"
done
echo "Lalu: $PY scripts/audit/audit_hp_grid.py" | tee -a "$LOG"
