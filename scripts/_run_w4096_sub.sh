#!/usr/bin/env bash
# _run_w4096_sub.sh — KONTROL UJI C: w4096 dengan jumlah window disamakan ke w16384.
# TAHAN-MATI: tahap yang sudah selesai dilewati.
#
# =====================================================================================
# MENGAPA KONTROL INI ADA
# =====================================================================================
# Uji C (2026-08-02 20:35) menemukan pola yang SEARAH dengan M14 tetapi tidak dapat
# dipertahankan, karena perlakuannya mengubah dua hal sekaligus:
#
#     (a) resolusi frekuensi   1,855 Hz -> 0,464 Hz     <- yang ingin diuji
#     (b) jumlah window latih  13.200   -> 3.200        <- efek samping
#
# Hasil Uji C (ERM, 8 seed berpasangan, audit backbone-beku):
#     torsi-0   78,64 -> 85,63   (+7,00; p = 0,21; 5/8)
#     torsi-7   99,05 -> 96,81   (-2,24; p = 0,18; 2/8)   <- sd 0,73 -> 4,09, F p = 0,0002
#     C-P1 (R0-R7 > 0 pada >=6/8): 5/8 -> GUGUR
#
# Torsi-7 tidak membutuhkan (a) sama sekali — pada 4,0 Nm jendela 4096 sudah memuat
# 3,16 periode kerusakan — sehingga ia hanya menanggung rugi dari (b). Selama keduanya
# bergerak bersamaan, +7,00 di torsi-0 tidak dapat diatribusikan ke resolusi.
#
# Lengan ini membekukan (b) dan menyisakan (a) sebagai satu-satunya yang berbeda:
#
#     w4096 penuh       13.200 window (33/rekaman)   resolusi RENDAH
#     w4096 subsampel    3.200 window ( 8/rekaman)   resolusi RENDAH   <- lengan ini
#     w16384             3.200 window ( 8/rekaman)   resolusi TINGGI
#
# Window subsampel diambil di indeks j = 0, 4, 8, ... yaitu 4096 sampel PERTAMA dari
# tiap blok 16384, sehingga letak waktunya BERIMPIT dengan window w16384 — bukan sekadar
# tersebar merata (scripts/make_subsample_w4096.py, 4/4 verifikasi struktur cocok).
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1) dan menyebut sumbunya (pelajaran D-P2).
# =====================================================================================
#   CTRL-P1 (utama)  Perbandingan yang menentukan: w16384 versus w4096-SUBSAMPEL,
#                    berpasangan pada 8 seed yang sama, pada torsi-0.
#
#       M14 DIDUKUNG bila  torsi-0(w16384) > torsi-0(w4096-sub)  pada >= 6 dari 8 seed.
#       M14 DITOLAK  bila  tidak.
#
#       Ini menghabiskan ruang hasil. Bila didukung, kenaikan torsi-0 berasal dari
#       resolusi dan bukan dari perbedaan data. Bila ditolak, +7,00 pada Uji C adalah
#       artefak jumlah window, dan klaim resolusi-jendela (Sec. IV-B makalah) turun dari "keyakinan sedang"
#       menjadi DITOLAK.
#
#   CTRL-P2 (kontrol kewarasan)  torsi-0(w4096-penuh) > torsi-0(w4096-subsampel).
#       Membuang 4/5 data harus merugikan. Bila TIDAK merugikan sama sekali, maka
#       jumlah window bukan variabel yang berpengaruh di rentang ini, dan seluruh
#       kekhawatiran confound (b) gugur dengan sendirinya — hasil Uji C boleh dibaca
#       apa adanya. Kedua kemungkinan informatif; tidak ada yang membatalkan uji.
#
#   CTRL-P3 (varians)  Laporkan sd antar-seed dan uji-F untuk ketiga lengan, TERLEPAS
#       dari CTRL-P1. Torsi-7 meledak dari 0,73 ke 4,09 di Uji C; perlu diketahui
#       apakah itu efek resolusi atau efek data.
#
# ATURAN DAYA: n=8 PENAPIS. Hitung daya sebelum mengusulkan perluasan — pelajaran Uji D,
# di mana n=20 hanya berdaya 0,393 untuk kontras d = 0,397.
#
# PRASYARAT: ${PYTHON:-python} -u scripts/make_subsample_w4096.py
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w4096_sub.sh
#
#   Ulang : rm -rf logs/.done_w4096sub
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_w4096sub
LOG=logs/w4096sub_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

if [ ! -f "data/processed/motor/$NPZ" ]; then
    echo "[ABORT] $NPZ belum ada. Jalankan scripts/make_subsample_w4096.py dulu."
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

# ERM saja, alasan sama dengan Uji C: yang ditanya apakah SINYALNYA ada, bukan metode
# mana yang terbaik. Seed dan seluruh flag lain identik dengan kedua lengan pembanding.
stage erm_w4096sub_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --lambda_max 0.0 --name W4096SUB_baseline

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku dengan dataset yang COCOK:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py --methods 'KONTROL: ERM w4096-sub' \\" | tee -a "$LOG"
echo "      --npz $NPZ --seeds $S8 --out results/motor/audit_erm_w4096sub_s8.npz" | tee -a "$LOG"
