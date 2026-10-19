#!/usr/bin/env bash
# _run_w16384_ovl.sh — UJI E: jendela 16384 dengan stride 4096 (tumpang tindih 75%).
# Mengambil resolusi tinggi DAN volume data sekaligus. TAHAN-MATI.
#
# =====================================================================================
# MENGAPA UJI INI ADA — dekomposisi yang mendahuluinya
# =====================================================================================
# Kontrol Uji C (2026-08-02 21:13, ERM, 8 seed berpasangan, audit backbone-beku)
# memisahkan dua efek yang di Uji C saling meniadakan:
#
#   lengan              window   resolusi   torsi-0        torsi-7
#   w4096 penuh         13.200   rendah     78,64 +/- 9,0  99,05 +/- 0,7
#   w4096 subsampel      3.200   rendah     67,59 +/- 7,0  96,23 +/- 4,3
#   w16384               3.200   TINGGI     85,63 +/-14,2  96,81 +/- 4,1
#
#   efek DATA     (13.200 -> 3.200) : -11,05 poin  (p = 0,0207; 6/8)
#   efek RESOLUSI (4096 -> 16384)   : +18,04 poin  (p = 0,0164; Wilcoxon 0,0391; 6/8)
#   neto yang terlihat di Uji C     :  +7,00 poin
#
# Disosiasinya persis prediksi M14: resolusi menolong torsi-0 (+18,04) dan TIDAK
# menolong torsi-7 (+0,58; p = 0,79) yang jendela 4096-nya sudah memuat 3,16 periode.
#
# Uji ini menanyakan: bila kedua efek diambil BERSAMAAN, apakah keduanya aditif?
#
#   w16384 stride 4096 -> ~12.000 window (resolusi TINGGI + volume ~w4096 penuh)
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1); sumbu varians disebut terpisah
# (pelajaran D-P2); daya dihitung sebelum mengusulkan perluasan (pelajaran Uji D).
# =====================================================================================
#   E-P1 (utama)  Pembanding yang relevan secara praktis adalah w4096 PENUH — yaitu
#                 konfigurasi yang dipakai SELURUH baris Tabel I makalah (torsi-0 = 78,64).
#
#       LOLOS bila  torsi-0(w16384-ovl) > torsi-0(w4096-penuh) pada >= 6 dari 8 seed.
#       GUGUR bila  tidak.
#
#       Ini pembanding yang lebih keras daripada w4096-subsampel, dan sengaja dipilih:
#       klaim yang berguna adalah "lebih baik daripada yang dipakai
#       sekarang", bukan "lebih baik daripada baseline yang dilemahkan".
#
#   E-P2 (aditivitas)  Prediksi aditif: torsi-0 ~ 67,59 + 18,04 + 11,05 = 96,7.
#       Dicatat sebagai angka acuan, BUKAN ambang lulus. Efeknya diperkirakan di
#       BAWAH prediksi aditif karena tumpang tindih 75% membuat window latih
#       berkorelasi, sehingga ukuran sampel efektif < 12.000. Yang dilaporkan adalah
#       selisih terukur terhadap prediksi, bukan vonis lulus/gagal.
#
#   E-P3 (spesifisitas)  torsi-7 TIDAK boleh naik terukur. M14 memprediksi resolusi
#       tidak menolong domain yang sudah teresolusi; bila torsi-7 ikut naik sebesar
#       torsi-0, maka yang bekerja BUKAN resolusi melainkan sekadar volume/augmentasi
#       dari tumpang tindih — dan seluruh atribusi M14 harus ditinjau ulang.
#
#   E-P4 (varians)  Laporkan sd antar-seed dan uji-F terhadap w4096 penuh TERLEPAS
#       dari E-P1. sd torsi-0 di w16384 non-tumpang-tindih tinggi (14,15); perlu
#       diketahui apakah volume data menormalkannya kembali.
#
# =====================================================================================
# AMBANG E-P1 DI n=20 — DITETAPKAN 3 Agu 2026, 03:47
# Ditetapkan TANPA melihat satu pun nilai hasil S12. Koreksi ketepatan (03:48): saat
# ambang ini ditulis, checkpoint seed-1 ternyata sudah tersimpan (1 dari 12) — jadi
# pernyataan "belum ada tahap yang selesai" akan keliru. Yang benar dan yang menjadi
# dasar keabsahan pra-registrasi ini: TIDAK ADA angka hasil S12 yang pernah dibaca —
# log hanya diperiksa untuk memastikan tahap S8 dilewati dan S12 berjalan. Nilai Win-F1
# atau per-torsi S12 pertama kali akan dibaca pada audit setelah seluruh 12 seed selesai.
# =====================================================================================
# Kriteria di atas ditulis untuk n=8 dan TIDAK mendefinisikan ambangnya di n=20.
# Menetapkannya setelah melihat hasil akan membatalkan pra-registrasi (aturan
# pelaporan no. 5), jadi ditetapkan sekarang:
#
#   E-P1 di n=20:  LOLOS bila torsi-0(w16384-ovl) > torsi-0(w4096-penuh) pada
#                  >= 15 dari 20 seed berpasangan.
#
#   Dasarnya perpanjangan proporsional yang setia: 6/8 = 75% -> 15/20 = 75%.
#   TIDAK ada ambang magnitudo baru yang ditambahkan. E-P1 memang kriteria
#   KONSISTENSI; menambah gerbang besar-efek sekarang adalah persis pelanggaran yang
#   aturan no. 5 larang. Besar efek dilaporkan deskriptif, bukan sebagai vonis.
#
#   E-P2, E-P3, E-P4 berlaku apa adanya, dinilai ulang pada 20 seed gabungan.
#
# PEMBANDING SUDAH ADA DAN SEED-NYA COCOK PERSIS.
#   results/motor/audit_ext_n20.npz -> 'ERM_(baseline)', n=20, seed identik dengan
#   S8 + S12. Terukur: torsi-0 74,18 +/- 9,87 | torsi-7 98,77 | F1 87,64.
#
#   Catat: baseline w4096 TURUN dari 78,64 (n=8) ke 74,18 (n=20). Penyusutan saat
#   diperluas adalah pola nyata di proyek ini (defisit DANN -2,78 -> -0,63). Karena
#   itu selisih di n=20 bisa MELEBAR walau w16384-ovl ikut turun sedikit. Kedua
#   sub-kasus dilaporkan terpisah; tidak ada yang dipilih setelah melihat angka.
#
# YANG DIAWASI LEBIH KETAT DARIPADA E-P1: E-P3.
#   Bila di n=20 torsi-7 IKUT NAIK terukur, mekanismenya bukan resolusi melainkan
#   volume window berkorelasi, dan seluruh atribusi M14 harus ditinjau ulang. Di n=8
#   torsi-7 justru turun (-1,19), jadi arahnya aman sejauh ini. E-P1 gagal hanya
#   melemahkan satu klaim; E-P3 gagal merobohkan penjelasannya.
#
# AUDIT SETELAH TAHAP INI SELESAI (WAJIB, --npz tidak boleh lupa):
#   $PY scripts/audit/audit_ckpt_ext.py --methods 'UJI E: ERM w16384-ovl' \
#       --npz motor_multiclass_w16384_s4096_ss_env.npz \
#       --seeds 42,123,2025,7,17,2024,2026,88,1,3,5,11,13,23,31,57,99,101,202,777 \
#       --out results/motor/audit_erm_w16384ovl_n20.npz
#   Prefix S8 dan S12 keduanya sudah terdaftar di METHODS, jadi satu perintah
#   mengambil kedua tahap sekaligus.
#
# CATATAN INTEGRITAS YANG WAJIB MASUK METODOLOGI
#   1. Tumpang tindih 75% membuat window latih SALING BERKORELASI. GroupKFold
#      per-rekaman tetap mencegah kebocoran ke himpunan uji — torsi uji tidak pernah
#      menyumbang window latih — sehingga angkanya sah. Yang berkurang adalah ukuran
#      sampel EFEKTIF, bukan validitasnya.
#   2. w4096 penuh memakai stride 4096 (0% tumpang tindih). Jadi perbandingan E-P1
#      mengubah resolusi DAN tumpang tindih sekaligus. Itu sebabnya E-P3 ada: ia
#      memisahkan "resolusi" dari "sekadar lebih banyak window berkorelasi".
#
# PRASYARAT:
#   ${PYTHON:-python} -u scripts/run_preprocessing.py --motor --window 16384 --stride 4096
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w16384_ovl.sh
#
#   Ulang : rm -rf logs/.done_w16384ovl
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s4096_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_w16384ovl
LOG=logs/w16384ovl_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

if [ ! -f "data/processed/motor/$NPZ" ]; then
    echo "[ABORT] $NPZ belum ada. Jalankan preprocessing dulu (lihat kepala berkas)."
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

# ERM saja — alasan sama dengan Uji C dan kontrolnya: yang ditanya apakah SINYALNYA
# ada dan bisa dipanen, bukan metode mana yang terbaik. Seed dan flag lain identik
# dengan ketiga lengan pembanding.
stage erm_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384OVL_baseline

# ---- TAHAP 2 (U1, ditambahkan 3 Agu 2026 03:40): perluasan ke n=20 ----
# E-P1 lolos di n=8 dengan d = 0,982 -> n=20 berdaya > 0,99. Perluasan DIBENARKAN, berbeda
# dengan Uji D yang sengaja dilewati (d = 0,397; daya n=20 hanya 0,393).
#
# Kriteria TIDAK berubah: E-P1..E-P4 di kepala berkas ini berlaku apa adanya, dinilai ulang
# pada n=20 gabungan. Seed S12 tidak beririsan dengan S8, jadi penggabungannya sah.
#
# Envelope SENGAJA tetap yang lama. Klaim jendela adalah selisih w4096 vs w16384-ovl dengan
# envelope identik di kedua lengan; mengganti envelope di tengah merusak basis E-P1.
# (U3 dibatalkan 3 Agu — premisnya gugur di tingkat window.)
S12="1,3,5,11,13,23,31,57,99,101,202,777"

stage erm_w16384ovl_s12 $PY scripts/train_motor_dann.py $BASE --seeds $S12 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384OVL_baseline_s12

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku dengan dataset yang COCOK:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py --methods 'UJI E: ERM w16384-ovl' \\" | tee -a "$LOG"
echo "      --npz $NPZ --seeds $S8 --out results/motor/audit_erm_w16384ovl_s8.npz" | tee -a "$LOG"
