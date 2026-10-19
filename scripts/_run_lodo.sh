#!/usr/bin/env bash
# _run_lodo.sh — LEAVE-ONE-TORQUE-OUT, 8 lipatan x 5 seed, ERM. TAHAN-MATI.
#
# Kriteria L-P1..L-P3 TERKUNCI di komentar ini, ditulis sebelum satu lipatan dijalankan.
# Hasilnya dilaporkan di Sec. V-A makalah CENIM 2026 (panel a gambar LODO).
#
# =====================================================================================
# MENGAPA UJI INI ADA
# =====================================================================================
# Audit fondasi 3 Agu 21:55 menemukan: 16 dari 16 skrip run memakai --extremes_out.
# Leave-one-torque-out 8 lipatan SUDAH terimplementasi dan merupakan DEFAULT, tetapi
# tidak pernah dipakai sekali pun. Seluruh proyek karena itu berdiri di atas SATU split
# domain, dan tidak ada domain interior yang pernah ditahan.
#
# Akibatnya M13 (target di luar convex hull) dan M14 (sinyal tidak terselesaikan)
# meramalkan urutan defisit yang SAMA dan tidak pernah terpisahkan. Bukti awal sudah
# melawan M13: torsi-7 juga di luar hull (18,2%) tetapi F1-nya 99,04 dan keterbacaan
# sinyalnya 91,0% -- normal.
#
# =====================================================================================
# YANG DIBACA DARI HASILNYA
# =====================================================================================
#   L-P1  Spearman rho antara F1 LODO per torsi dan keterbacaan A6 per torsi (8 titik).
#         LOLOS bila rho >= 0,70 -> keterbacaan sinyal meramalkan kesulitan model,
#         diukur TANPA melatih apa pun. M14 ditegakkan.
#
#   L-P2  M13 DICABUT sebagai penjelasan bila F1(torsi-7) berada DI DALAM rentang
#         interior -- berada di luar hull tidak dengan sendirinya menghukum.
#
#   L-P3  Kurva penuh, deskriptif.
#
# RAMBU: plafon di ujung atas; protokol berbeda dari extremes-out (7 torsi
# latih, bukan 6) sehingga perbandingan lintas-protokol DESKRIPTIF saja; n=5 seed.
#
# =====================================================================================
set -u
set -o pipefail

PY=${PYTHON:-python}
S5="42,123,2025,7,17"
NPZ="motor_multiclass_w4096_s4096_ss_env.npz"

# IDENTIK dengan BASE seluruh tabel, HANYA --extremes_out yang dibuang.
BASE="--backbone tsitplus --no_augment --fixed_budget --fp32 --epochs 50 --n_folds 8"

DONE=logs/.done_lodo
mkdir -p "$DONE"
LOG="logs/lodo_$(date +%Y%m%d_%H%M%S).log"

stage() {
    local name="$1"; shift
    if [ -f "$DONE/$name" ]; then
        echo "=== [LEWATI] $name (sudah selesai)" | tee -a "$LOG"
        return 0
    fi
    echo "=== [MULAI] $name" | tee -a "$LOG"
    if "$@" 2>&1 | tee -a "$LOG"; then
        touch "$DONE/$name"
        echo "=== [SELESAI] $name" | tee -a "$LOG"
    else
        echo "=== [GAGAL] $name — penanda TIDAK ditulis, akan diulang" | tee -a "$LOG"
        return 1
    fi
}

echo "log: $LOG"
echo "LODO 8 lipatan x 5 seed, ERM saja (lambda_max 0.0)" | tee -a "$LOG"

# Satu lipatan per pemanggilan (--fold_idx) supaya kegagalan di tengah tidak membuang
# lipatan yang sudah selesai. Penanda per lipatan.
for f in 0 1 2 3 4 5 6 7; do
    stage "lodo_fold${f}" $PY scripts/train_motor_dann.py $BASE \
        --seeds "$S5" --npz "$NPZ" --lambda_max 0.0 \
        --fold_idx "$f" --name "LODO_baseline" || exit 1
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Analisis: $PY scripts/audit/audit_lodo.py" | tee -a "$LOG"
