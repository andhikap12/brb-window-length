#!/usr/bin/env bash
# _run_lodo_metode.sh — LODO 8 lipatan untuk TIGA METODE TERKUAT, mencabut sebagian B7.
#
# MENGAPA
# =======
# Batasan split tunggal (Sec. VI makalah CENIM 2026): split domain tunggal TERCABUT untuk ERM (LODO 8 lipatan,
# 4 Agu) tetapi TETAP BERLAKU untuk keluarga metode, yang seluruhnya masih satu split.
# Artinya separuh NEGATIF kontribusi paper — "tidak ada intervensi sisi-objektif yang
# terpisah dari baseline" — berdiri di atas fondasi yang lebih tipis daripada separuh
# positifnya. Separuh positif kini punya n=20, LODO 8 lipatan, dan 4 sel hiperparameter;
# separuh negatif punya satu split.
#
# MENGAPA HANYA TIGA, BUKAN LIMA BELAS
# ====================================
# LODO memakai 8 lipatan x 5 seed, yaitu ~5x biaya satu lengan extremes-out. Terukur dari
# kampanye 8 Agu: w4096-penuh 8 seed satu split = 31 menit, jadi LODO ~2,5 jam per metode.
# Lima belas metode = ~39 jam dan itu tidak masuk akal.
#
# Yang dipilih adalah tiga metode dengan Rec-Acc torsi-0 TERTINGGI di Tabel I, yaitu titik
# paling rentan bagi klaim negatif: kalau metode yang PALING mungkin mengalahkan ERM tetap
# tidak terpisah di bawah 8 lipatan, klaim negatifnya jauh lebih kuat daripada sebelumnya.
# Menguji metode yang lemah tidak menambah apa pun, karena ia sudah kalah di satu split.
#
#   89,25  BT+SupCon regresi-idx   (dannsupcon + bt_ratio 0.3 + domain_index f2s)
#   87,50  SupCon
#   86,00  DANN+SupCon
#
# INI PENCABUTAN SEBAGIAN, BUKAN PENUH. Sesudah kampanye ini B7 harus berbunyi: tercabut
# untuk ERM dan untuk ketiga lengan terkuat; MASIH BERLAKU untuk dua belas lengan sisanya.
# Jangan menuliskannya sebagai tercabut penuh.
#
# PROTOKOL disalin utuh dari _run_lodo.sh: BASE identik dengan seluruh tabel dengan
# --extremes_out DIBUANG dan --n_folds 8 dipasang, seed S5 SAMA dengan ERM LODO supaya
# perbandingannya berpasangan per seed. Satu lipatan per pemanggilan dengan --fold_idx,
# penanda per lipatan, sehingga mati di tengah tidak membuang lipatan yang sudah selesai.

set -u
set -o pipefail   # WAJIB — lihat catatan di bawah
cd "$(dirname "$0")/.."

# CACAT PIPEFAIL, ditemukan dan diperbaiki 8 Agu 2026
# ===================================================
# `stage()` menandai tahap selesai lewat `if "$@" 2>&1 | tee -a "$LOG"; then`. Tanpa
# `pipefail`, status keluar sebuah pipeline di bash adalah status perintah TERAKHIR,
# yaitu `tee`, yang hampir selalu 0. Jadi python yang jatuh pun menghasilkan status 0,
# penanda tetap ditulis, dan cabang `[GAGAL]` di bawah adalah KODE MATI yang praktis
# tidak pernah dijalankan.
#
# Terverifikasi:
#   set -u; if bash -c 'exit 42' | tee -a /dev/null; then echo SUKSES; else echo GAGAL; fi
#   -> SUKSES   (dengan `set -o pipefail` -> GAGAL)
#
# Akibatnya bukan tahap yang hilang, melainkan tahap hilang yang MENYAMAR SELESAI:
# karena penandanya ada, setiap penjalanan ulang MELEWATINYA, jadi lubangnya permanen
# dan senyap. Korban nyatanya `lodo_supcon_f5` (8 Agu 08:29): keluar dalam 0 menit tanpa
# satu baris keluaran pun, ditandai SELESAI, lima hasil torsi-2 hilang.
#
# Kelima belas `scripts/_run_*.sh` LAIN masih memakai pola yang sama dan BELUM
# diperbaiki. Periksa dengan `scripts/audit/verifikasi_penanda.py` sebelum mempercayai
# penanda kampanye mana pun.

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

# Urutan sengaja: SupCon dulu (paling murah, tanpa GRL dan tanpa BT), supaya salah
# konfigurasi terlihat dalam menit pertama alih-alih setelah dua jam.
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
