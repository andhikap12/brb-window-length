#!/usr/bin/env bash
# _run_window16384.sh — UJI C: jendela 16384 menguji batas fisik (M14).
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_bt_dann.sh.
#
# Rujukan: Sec. IV-B dan V-B makalah CENIM 2026.
#
# =====================================================================================
# HIPOTESIS (M14)
# =====================================================================================
# Seluruh kesulitan terkonsentrasi di torsi terendah: F1 78,64% pada 0,5 Nm versus
# 99,04% pada 4,0 Nm. Slip diukur langsung dari kanal getaran mentah untuk kedelapan
# level beban (audit_slip.py). Pada beban terendah:
#
#   jendela  4096 = 0,539 s | resolusi 1,855 Hz | 2sf = 1,78 Hz -> DI BAWAH RESOLUSI
#                                                | jendela memuat 0,96 periode
#   jendela 16384 = 2,156 s | resolusi 0,464 Hz | 2sf = 1,78 Hz -> TERESOLUSI
#                                                | jendela memuat 3,84 periode
#
# Bila terkonfirmasi, ini menjelaskan seluruh temuan dalam satu kalimat: TIDAK ADA
# metode invariansi yang dapat menciptakan sinyal yang tidak terekam. Itu sebabnya
# sumbu invariansi datar (+0,53 poin, p = 0,218, sepanjang probe 96,17% -> 20,00%).
#
# =====================================================================================
# YANG DIBANDINGKAN — DAN MENGAPA HANYA INI
# =====================================================================================
# Jumlah window turun 4x (13.200 -> ~3.300), sehingga basis perbandingannya berubah
# total. Angka w16384 TIDAK sebanding dengan tabel n=20 mana pun. Satu-satunya
# perbandingan yang sah adalah ERM w16384 versus ERM w4096 pada 8 seed yang SAMA.
#
# Pembanding ERM w4096 (n=8, seed identik): 89,40 +/- 4,18
#   per-torsi dari audit_ckpt_ext.py: torsi-0 74,18 | torsi-7 98,77 (angka n=20;
#   audit n=8 dijalankan ulang saat membaca hasil agar benar-benar berpasangan).
#
# ONGKOS TERUKUR (smoke 2 epoch, 2026-08-02 20:12): ~2-3 detik/epoch, jadi ~2,5 menit
# per seed dan ~20 menit untuk 8 seed. Perkiraan analitik sebelumnya (4x lebih mahal,
# ~3 jam) KELIRU: ia menghitung attention O(token^2) tetapi mengabaikan bahwa jumlah
# batch per epoch ikut turun 4x dan GPU menyerap biaya attention pada token sekecil
# 256. Angka terukur yang berlaku, bukan estimasinya.
#
# CATATAN CONFOUND YANG HARUS DILAPORKAN. token_size tetap 64, sehingga jendela 4x
# lebih panjang menghasilkan 4x lebih banyak token (64 -> 256).
# token_size TIDAK dinaikkan ke 256 dengan sengaja: token 256 sampel = 33,7 ms,
# sedangkan pembawa 60 Hz berperioda 16,7 ms — tokenizer sekasar itu meng-alias
# pembawa yang justru menjadi tempat sideband kerusakan menumpang. Jadi jumlah token
# ikut berubah, dan itu bagian dari perlakuan, bukan variabel yang dikendalikan.
# Harus dinyatakan eksplisit di metodologi.
#
# =====================================================================================
# KRITERIA GUGUR — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# =====================================================================================
# PELAJARAN DARI UJI D DAN UJI A: kriteria harus MEMBELAH HABIS ruang hasil. D-P2 hanya
# menyebut rerata sehingga efek nyata di varians nyaris lolos; A-P1 memberi dua ambang
# berjauhan (<93,57 berhasil, >=95 gugur) dan hasilnya jatuh di zona mati 94,02.
# Karena itu C-P1 di bawah ditulis sebagai SATU ambang tunggal tanpa celah.
#
#   C-P1 (utama, ambang tunggal — tidak ada zona mati)
#       Ukuran: reduksi galat RELATIF, bukan poin absolut.
#           R(torsi) = (F1_16384 - F1_4096) / (100 - F1_4096)
#       Alasan wajib relatif: torsi-7 pada 98,77 hanya punya 1,23 poin ruang naik,
#       torsi-0 pada 74,18 punya 25,82. Membandingkan kenaikan poin absolut di antara
#       keduanya menyesatkan secara struktural.
#
#       LOLOS  bila  R(torsi-0) - R(torsi-7) > 0  pada >= 6 dari 8 seed berpasangan.
#       GUGUR  bila  tidak. (Dua kemungkinan itu habis membelah ruang hasil.)
#
#       Bila gugur, penjelasan resolusi-jendela (M14) gugur sebagai penyebab defisit
#       torsi rendah, dan klaim resolusi-jendela makalah (Sec. V-B) harus ditulis ulang dari "keyakinan
#       sedang" menjadi ditolak.
#
#   C-P2 (deskriptif, sudah terverifikasi secara aritmetika sebelum run)
#       Torsi-0 berpindah kategori: 0,96 -> 3,84 periode per jendela; resolusi
#       1,855 -> 0,464 Hz sehingga 2sf = 1,78 Hz berpindah dari DI BAWAH resolusi
#       menjadi teresolusi. Ini bukan prediksi empiris melainkan konsekuensi
#       aritmetika; dicatat agar angkanya masuk laporan, bukan sebagai uji.
#
#   C-P3 (varians — sumbu yang C-P1 tidak minta; ditambahkan setelah pelajaran Uji D)
#       Laporkan sd antar-seed dan uji-F terhadap ERM w4096 TERLEPAS dari hasil C-P1.
#       Uji A memburuk signifikan di sumbu ini (sd 1,71 -> 4,20; p = 0,0303) dan itu
#       hanya ketahuan karena diperiksa terpisah.
#
# ATURAN DAYA: n=8 PENAPIS. Sebelum memutuskan perluasan, hitung dayanya dulu —
# pelajaran Uji D: n=20 hanya berdaya 0,393 untuk kontras d = 0,397, jadi perluasan
# tidak otomatis menyelesaikan apa pun.
#
# PRASYARAT: data/processed/motor/motor_multiclass_w16384_s16384_ss_env.npz harus ada
#   ${PYTHON:-python} -u scripts/run_preprocessing.py --motor --window 16384 --stride 16384
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_window16384.sh
#
#   Ulang satu tahap : rm logs/.done_w16384/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_w16384
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s16384_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"

DONE=logs/.done_w16384
LOG=logs/w16384_$(date +%Y%m%d_%H%M%S).log
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
echo "penanda selesai: $DONE/"

# ---- TAHAP 1: ERM pada jendela 16384 ----
# HANYA ERM. Uji ini menanyakan apakah SINYALNYA ada, bukan metode mana yang terbaik.
# Menambah varian metode di sini mencampur dua pertanyaan dan melipatgandakan ongkos
# yang sudah 4x lebih mahal per epoch.
stage erm_w16384_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --lambda_max 0.0 --name W16384_baseline

echo "" | tee -a "$LOG"
echo "=== TAHAP n=8 SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku, bukan dari log:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py  (tambahkan entri W16384_baseline)" | tee -a "$LOG"
echo "Baca terhadap C-P1..C-P3 di kepala berkas ini SEBELUM menyimpulkan." | tee -a "$LOG"
