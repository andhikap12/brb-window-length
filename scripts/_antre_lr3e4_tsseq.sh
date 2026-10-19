#!/usr/bin/env bash
# _antre_lr3e4_tsseq.sh — tunggu kampanye lr TSiTPlus selesai, lalu kerjakan sisanya.
#
# Urutan, semuanya berurutan karena semuanya memakai GPU yang sama:
#   1. tunggu 4 penanda selesai milik _run_lr_faktorial.sh
#   2. audit 4 lengan TSiTPlus @ 3e-4          -> audit_lr3e4_*_s8.npz
#   3. kampanye TSSequencerPlus @ 3e-4          (~1 jam, terukur 2x lebih cepat)
#   4. audit 4 lengan TSSequencerPlus @ 3e-4   -> audit_lr3e4tsseq_*_s8.npz
#   5. cetak kisi backbone x lr yang sudah lengkap
#
# MENGAPA MENUNGGU, BUKAN JALAN BARENG
# ====================================
# Kartunya 6 GB. Dua kampanye sekaligus membuat keduanya OOM, dan yang lebih buruk,
# OOM di jam kedua meninggalkan kampanye separuh jadi yang penanda selesainya tidak
# ditulis sehingga tidak jelas mana yang sahih.
#
# MENUNGGU PENANDA, BUKAN PROSES
# ==============================
# Penanda hanya ditulis kalau lengannya BERHASIL (lihat fungsi stage di skrip kampanye).
# Jadi kalau kampanye lr gagal di tengah, jumlah penanda tidak pernah sampai 4 dan skrip
# ini berhenti dengan pesan alih-alih menumpuk kampanye baru di atas kegagalan. Itu
# perilaku yang diinginkan: jangan antre di atas hasil yang belum tentu sahih.
#
# --methods WAJIB DILEWATKAN. Tanpa itu audit_ckpt_ext.py mengulang SELURUH isi METHODS
# terhadap --npz yang diberikan, sehingga metode lain dievaluasi memakai dataset yang salah
# dan keluarannya berkunci banyak. audit_hp_grid.py menuntut satu kunci per berkas dan akan
# menolak berkas berkunci banyak; itu gerbangnya, bukan kebetulan.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
S8="42,123,2025,7,17,2024,2026,88"
LOG="logs/antre_lr3e4_tsseq_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

DONE_LR=logs/.done_lr_faktorial
BATAS_TUNGGU_DETIK=$((8 * 3600))

catat() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

declare -A NPZ=(
  [w4096sub]="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
  [w16384]="motor_multiclass_w16384_s16384_ss_env.npz"
  [w4096full]="motor_multiclass_w4096_s4096_ss_env.npz"
  [w16384ovl]="motor_multiclass_w16384_s4096_ss_env.npz"
)
declare -A LABEL_LR=(
  [w4096sub]="LR3E4: ERM w4096-sub"
  [w16384]="LR3E4: ERM w16384"
  [w4096full]="LR3E4: ERM w4096-penuh"
  [w16384ovl]="LR3E4: ERM w16384-ovl"
)
declare -A LABEL_TSSEQ=(
  [w4096sub]="LR3E4TSSEQ: ERM w4096-sub"
  [w16384]="LR3E4TSSEQ: ERM w16384"
  [w4096full]="LR3E4TSSEQ: ERM w4096-penuh"
  [w16384ovl]="LR3E4TSSEQ: ERM w16384-ovl"
)

catat "log: $LOG"

# ---------------------------------------------------------------- 1. tunggu
catat "menunggu 4 penanda di $DONE_LR (batas $((BATAS_TUNGGU_DETIK/3600)) jam)"
mulai=$(date +%s)
while true; do
  n=$(ls -1 "$DONE_LR" 2>/dev/null | wc -l)
  if [ "$n" -ge 4 ]; then
    catat "kampanye lr TSiTPlus selesai: $n/4 penanda"
    break
  fi
  if [ $(($(date +%s) - mulai)) -ge "$BATAS_TUNGGU_DETIK" ]; then
    catat "[BERHENTI] batas tunggu tercapai dengan $n/4 penanda."
    catat "           Kampanye lr kemungkinan gagal di tengah. Periksa"
    catat "           logs/lr_faktorial_*.log sebelum mengantre ulang."
    exit 1
  fi
  sleep 60
done

# Beri GPU waktu melepas memori sebelum beban berikutnya masuk.
sleep 30

# ---------------------------------------------------------------- 2. audit TSiTPlus 3e-4
catat "audit 4 lengan TSiTPlus @ 3e-4"
for k in w4096sub w16384 w4096full w16384ovl; do
  out="results/motor/audit_lr3e4_${k}_s8.npz"
  if [ -f "$out" ]; then
    catat "  [LEWATI] $out sudah ada"
    continue
  fi
  catat "  audit $k"
  if ! $PY scripts/audit/audit_ckpt_ext.py --npz "${NPZ[$k]}" --seeds "$S8" \
        --methods "${LABEL_LR[$k]}" --out "$out" >>"$LOG" 2>&1; then
    catat "  [GAGAL] audit $k — lihat $LOG"
    exit 1
  fi
done

# ---------------------------------------------------------------- 3. kampanye TSSEQ 3e-4
catat "kampanye TSSequencerPlus @ 3e-4 (awalan LR3E4TSSEQ)"
if ! bash scripts/_run_hp_faktorial.sh tssequencerplus 3e-4 LR3E4TSSEQ >>"$LOG" 2>&1; then
  catat "[GAGAL] kampanye TSSequencerPlus — lihat $LOG"
  exit 1
fi
catat "kampanye TSSequencerPlus selesai"

sleep 30

# ---------------------------------------------------------------- 4. audit TSSEQ 3e-4
catat "audit 4 lengan TSSequencerPlus @ 3e-4"
for k in w4096sub w16384 w4096full w16384ovl; do
  out="results/motor/audit_lr3e4tsseq_${k}_s8.npz"
  if [ -f "$out" ]; then
    catat "  [LEWATI] $out sudah ada"
    continue
  fi
  catat "  audit $k"
  if ! $PY scripts/audit/audit_ckpt_ext.py --npz "${NPZ[$k]}" --seeds "$S8" \
        --methods "${LABEL_TSSEQ[$k]}" --out "$out" >>"$LOG" 2>&1; then
    catat "  [GAGAL] audit $k — lihat $LOG"
    exit 1
  fi
done

# ---------------------------------------------------------------- 5. kisi
catat "kisi backbone x lr:"
$PY scripts/audit/audit_hp_grid.py 2>&1 | tee -a "$LOG"

catat "=== ANTREAN SELESAI ==="
