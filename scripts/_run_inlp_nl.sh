#!/usr/bin/env bash
# _run_inlp_nl.sh — probe nonlinear setelah INLP (Sec. IV-D makalah CENIM 2026).
#
# MENGAPA CPU, BUKAN GPU
# ======================
# Ekstraksi fiturnya kecil (inferensi, batch 64), tetapi kartunya 6,4 GB dan sedang
# dipakai kampanye LODO tiga-metode yang berjalan ~7 jam. OOM di sana membuang seluruh
# kampanye itu; menunggu beberapa menit lebih lama di CPU tidak membuang apa pun.
#
# MENGAPA ENAM DARI ENAM BELAS INTI
# =================================
# Uji asap 8 Agu memakai seluruh inti dan lipatan kampanye yang sedang berjalan melar
# dari ~10 menit ke ~19 menit. Dataloader kampanye itu berebut CPU dengan sklearn di
# sini. Enam inti membuat run ini lebih lambat tetapi tidak menyandera pekerjaan yang
# jauh lebih mahal.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export PYTHONIOENCODING=utf-8

SUMBER="${1:-ERM}"
LOG="logs/inlp_nl_${SUMBER,,}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "log: $LOG"
${PYTHON:-python} -u scripts/audit/npdann_inlp.py \
    --source "$SUMBER" \
    --device cpu \
    --out "results/motor/npdann_inlp_nl_${SUMBER,,}.npz" \
    2>&1 | tee -a "$LOG"
