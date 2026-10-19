#!/usr/bin/env bash
# _run_bt_dann.sh — BT-DANN (normalisasi gradien; Sec. IV-A dan IV-D makalah CENIM 2026).
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_byvar_seeds.sh.
#
# APA YANG DIUJI
#   lambda tidak lagi dijadwalkan menurut waktu, melainkan dihitung tiap batch agar
#   ||g_dom|| / ||g_fault|| tetap di rasio_target. Tanpa normalisasi rasio meledak
#   0,48 -> 5,06 dan 45/50 epoch dikuasai objektif adversarial (Sec. IV-D makalah).
#
# PREDIKSI TERUKUR (semuanya bisa gugur)
#   1. rasio efektif datar di 0,30 — terbaca di baris [BT-DANN-BAL ep N] tiap epoch
#   2. varians antar-seed TURUN (sd DANN+SupCon 5,80 tertinggi di seluruh tabel)
#   3. seed kolaps hilang
#   4. rerata naik ke SEKITAR ERM — bukan melampauinya (argumen langit-langit)
#
# PEMBANDING (n=20, aggregate_seeds.py 2026-08-02)
#   ERM 87,64 +/- 4,36 | DANN 87,01 +/- 4,17 | SupCon 89,85 +/- 4,62
#
# TAHAP 1-2 memberi n=8 lebih dulu (~50 menit) sebagai penapis; tahap 3-4 melengkapi
# ke n=20 memakai 12 seed yang sama dengan run perluasan.
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_bt_dann.sh
#
#   Ulang satu tahap : rm logs/.done_bt_dann/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_bt_dann
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
S12="1,3,5,11,13,23,31,57,99,101,202,777"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_bt_dann
LOG=logs/bt_dann_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

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
echo "PRASYARAT: scripts/audit/sanity_bt_dann.py harus lolos 11/11 sebelum ini dijalankan."

# ---- TAHAP 1-2: n=8, penapis cepat ----
# Varian SupCon DIDAHULUKAN (2026-08-02): hasil n=20 menunjukkan kerusakan GRL yang bisa
# diperbaiki terkonsentrasi di sana (DANN+SupCon - SupCon = -2,52; p = 0,049), sementara
# DANN polos hanya -0,63 dari ERM dan tak terukur. Pertanyaan menentukan dijawab duluan.
stage bt_dannsupcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 --method dannsupcon \
    $BT $SC --name BT_dannsupcon_r03

stage bt_dann_s8       $PY scripts/train_motor_dann.py $BASE --seeds $S8 $BT \
    --name BT_dann_r03

# ---- TAHAP 3-4: lengkapi ke n=20 ----
stage bt_dann_s12       $PY scripts/train_motor_dann.py $BASE --seeds $S12 $BT \
    --name BT_dann_r03_s12

stage bt_dannsupcon_s12 $PY scripts/train_motor_dann.py $BASE --seeds $S12 --method dannsupcon \
    $BT $SC --name BT_dannsupcon_r03_s12

echo "" | tee -a "$LOG"
echo "=== SEMUA TAHAP SELESAI ===" | tee -a "$LOG"
echo "Agregasi: $PY scripts/audit/aggregate_seeds.py" | tee -a "$LOG"
echo "Periksa rasio: grep 'BT-DANN.*-BAL ep' $LOG | tail -20" | tee -a "$LOG"
