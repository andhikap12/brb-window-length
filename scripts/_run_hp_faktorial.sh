#!/usr/bin/env bash
# _run_hp_faktorial.sh — faktorial 2x2 R3 pada SATU sel (backbone, lr) mana pun.
#
# Pemakaian:
#     bash scripts/_run_hp_faktorial.sh <backbone> <lr> [awalan]
# Contoh:
#     bash scripts/_run_hp_faktorial.sh tssequencerplus 3e-4
#     bash scripts/_run_hp_faktorial.sh tsitplus        3e-4 LR3E4
#
# MENGAPA TERPARAMETER
# ====================
# Rancangan hiperparameter x backbone butuh satu kampanye per sel. Menyalin
# _run_tsseq_faktorial.sh sekali per sel akan melahirkan tujuh berkas nyaris identik yang
# pasti saling menyimpang begitu satu di antaranya diperbaiki. Skrip ini menggantikan
# keluarga itu; _run_lr_faktorial.sh (tsitplus @ 3e-4, dilepas 8 Agu 00:29) adalah kasus
# khususnya yang sengaja DIBIARKAN UTUH karena ia sedang berjalan saat berkas ini ditulis.
#
# RANCANGAN YANG SEDANG DILENGKAPI
# ================================
# Sel = (backbone, lr). Isi tiap sel = faktorial 2x2 R3 (panjang jendela x jumlah window),
# 4 lengan x 8 seed. Yang sudah ada dan yang belum, per 8 Agu 2026:
#
#                        lr 1e-3                    lr 3e-4
#   TSiTPlus             ADA (r3_*.npz)             BERJALAN (_run_lr_faktorial.sh)
#   TSSequencerPlus      ADA (audit_tsseq_*.npz)    BELUM  <- sel yang melengkapi 2x2
#
# Melengkapi sel keempat membuat backbone x lr jadi faktorial PENUH atas dua backbone yang
# berbagi tokenisasi, LayerNorm, positional encoding, dan CLS-token. Itu satu-satunya
# pasangan di repo ini yang isolasi bias induktifnya bersih.
#
# MENGAPA KELUARGA CNN TIDAK ADA DI TABEL ITU
# ===========================================
# InceptionTimePlus, ResNetPlus, dan CNN1D mengembalikan BatchNorm + Global Average Pooling
# sekaligus. `src/models/backbones/tsitplus.py` sudah mencatat bahwa BN+GAP membuat baseline
# auto-invarian terhadap beban BY CONSTRUCTION, jadi selnya akan terkonfusi dengan isu
# pencucian amplitudo dan tidak dapat diatribusikan ke lr. Skrip ini TIDAK menolaknya, tetapi
# hasil dari sana menjawab pertanyaan yang BERBEDA: "apakah efeknya muncul di rezim
# normalisasi lain", bukan "apakah efeknya artefak penyetelan". Jangan campur keduanya dalam
# satu kalimat.
#
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

# Awalan nama checkpoint. Wajib unik per sel: checkpoint disimpan sebagai
# {name}_{backbone}_seed{s}_fold0.pth dan audit menurunkan backbone dari akhiran nama itu.
# Dua sel dengan awalan sama akan saling menimpa TANPA peringatan.
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

# Gerbang tabrakan nama. Kampanye lr 1e-3 yang asli memakai awalan telanjang
# (W4096SUB_baseline dkk) dan TIDAK BOLEH tertimpa; awalan sel baru harus punya pembeda.
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
