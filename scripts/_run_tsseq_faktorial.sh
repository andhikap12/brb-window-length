#!/usr/bin/env bash
# _run_tsseq_faktorial.sh — faktorial 2x2 R3 diulang dengan backbone NON-TRANSFORMER.
#
# MENGAPA
# =======
# "Backbone tunggal" adalah batasan utama paper (Sec. IV-B dan VI makalah CENIM 2026).
# Kampanye ini mengulang faktorial R3 (panjang jendela x jumlah window) dengan
# TSSequencerPlus, yaitu BiLSTM yang menggantikan self-attention, dan mempertahankan
# tokenisasi, LayerNorm, positional encoding, serta CLS-token IDENTIK dengan TSiTPlus.
# Yang diuji: apakah efek panjang jendela selamat di bias induktif lain.
#
# Alasan memilih TSSequencerPlus alih-alih CNN ada di docstring
# src/models/backbones/tssequencerplus.py. Ringkasnya: CNN akan mengembalikan
# BatchNorm + GAP sekaligus, sehingga hasilnya tidak dapat diatribusikan.
#
# BIAYA TERUKUR (RTX 4050 Laptop 6,4 GB, batch 32): TSSequencerPlus 0,014 s/step di
# T=4096 dan 0,031 s/step di T=16384, yaitu dua kali lebih cepat daripada TSiTPlus di
# jendela panjang. Taksiran seluruh kampanye ~1,5 sampai 2 jam.
#
# CATATAN JUJUR: di T=16384 backbone ini punya 3,76M parameter lawan 0,38M milik
# TSiTPlus, jadi perbandingannya TIDAK parameter-matched. Klaim yang diambil karena itu
# hanya kontras DI DALAM backbone (w16384 lawan w4096), bukan level absolut antar
# backbone.
#
# Protokol disalin utuh dari _run_w4096_sub.sh dan _run_w16384_ovl.sh: seed, flag, dan
# epoch sama, hanya --backbone yang berubah. Penanda selesai membuat skrip ini dapat
# diulang tanpa mengerjakan lengan yang sudah beres.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
S8="42,123,2025,7,17,2024,2026,88"
BASE="--backbone tssequencerplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_tsseq_faktorial
LOG=logs/tsseq_faktorial_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

# lengan -> berkas npz. Keempatnya sel faktorial R3.
declare -A NPZ=(
  [w4096sub]="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
  [w16384]="motor_multiclass_w16384_s16384_ss_env.npz"
  [w4096full]="motor_multiclass_w4096_s4096_ss_env.npz"
  [w16384ovl]="motor_multiclass_w16384_s4096_ss_env.npz"
)
declare -A NAMA=(
  [w4096sub]="TSSEQ_W4096SUB_baseline"
  [w16384]="TSSEQ_W16384_baseline"
  [w4096full]="TSSEQ_W4096FULL_baseline"
  [w16384ovl]="TSSEQ_W16384OVL_baseline"
)

for k in "${!NPZ[@]}"; do
  if [ ! -f "data/processed/motor/${NPZ[$k]}" ]; then
    echo "[ABORT] ${NPZ[$k]} belum ada."
    exit 1
  fi
done

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

# Urutan sengaja dari termurah ke termahal, supaya kegagalan konfigurasi terlihat
# dalam menit pertama alih-alih setelah satu jam.
for k in w4096sub w16384 w4096full w16384ovl; do
  stage "erm_${k}_tsseq_s8" $PY scripts/train_motor_dann.py $BASE --seeds "$S8" \
      --npz "${NPZ[$k]}" --lambda_max 0.0 --name "${NAMA[$k]}"
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku, satu perintah per lengan:" | tee -a "$LOG"
for k in w4096sub w16384 w4096full w16384ovl; do
  echo "  $PY scripts/audit/audit_ckpt_ext.py --npz ${NPZ[$k]} --seeds $S8 \\" | tee -a "$LOG"
  echo "      --out results/motor/audit_tsseq_${k}_s8.npz" | tee -a "$LOG"
done
