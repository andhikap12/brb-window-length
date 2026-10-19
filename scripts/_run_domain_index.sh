#!/usr/bin/env bash
# _run_domain_index.sh — UJI A: regresi indeks domain di atas BT-DANN-SupCon.
# TAHAN-MATI: tahap yang sudah selesai dilewati. Pola sama dengan _run_bt_dann.sh.
#
# Rujukan: lengan LODO "BT + SupCon regresi-indeks" (Sec. VI makalah CENIM 2026).
# PRASYARAT: scripts/audit/sanity_domain_index.py lolos 32/32 (sudah, 2026-08-02).
#
# =====================================================================================
# HIPOTESIS
# =====================================================================================
# Kepala domain KATEGORIS memberi backbone 6! = 720 permutasi label sebagai jalan kabur
# yang MEMPERTAHANKAN informasi (M3): GRL memenuhi "maksimalkan loss diskriminator"
# dengan memetakan ulang fitur ke wilayah berlabel salah, bukan menghapus. Bukti M3:
# domain_loss melampaui plafon ln(6) = 1,7918 pada 32/32 kombinasi varian x seed
# (Wilcoxon p = 0,0078) — "salah dengan percaya diri" mustahil tanpa informasi.
#
# Target SKALAR tidak punya label untuk dipermutasi. Ruang kaburnya menyusut jadi pada
# dasarnya satu: pembalikan monoton (u -> 1-u), yang terdeteksi langsung sebagai
# korelasi(D(z), u) negatif.
#
# =====================================================================================
# TEMUAN YANG MENGUBAH RANCANGAN ABLASI (diukur 2026-08-02, sebelum run)
# =====================================================================================
# Ada tiga nilai indeks yang mungkin: torque, slip, f2s. Setelah normalisasi [0,1]:
#
#     u[f2s] == u[slip]  PERSIS   (2sf = 120*slip -> penskalaan murni; beda 1,1e-16)
#     u[torque] beda maks 0,0581 dari u[slip]  (r = 0,9978)
#
# Jadi "tiga varian indeks" sesungguhnya SATU target, plus satu varian yang beda tipis.
# Menjalankan f2s dan slip sebagai dua ablasi terpisah adalah pemborosan GPU dan TIDAK
# boleh dilaporkan sebagai ablasi — hasilnya identik secara aritmetika, bukan empiris.
# Karena itu hanya `f2s` yang dijalankan (variabel kausal fisiknya, default).
# Ablasi `torque` disiapkan tetapi TIDAK otomatis — lihat TAHAP 2.
#
# =====================================================================================
# KRITERIA GUGUR — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# (aturan pelaporan no.5: tidak boleh diubah)
# =====================================================================================
# Pembanding probe domain (beku, 6-arah, GroupKFold, audit_ckpt_ext.py):
#     ERM 96,16  |  DANN lambda=1,0 93,57  |  BT-DANN-SupCon 93,81
# Pembanding akurasi n=8: BT-DANN-SupCon 89,81 +/- 1,71
#
#   A-P1 (utama)     Probe turun DI BAWAH 93,57% — lebih baik daripada varian GRL mana
#                    pun sejauh ini.
#                    GUGUR bila probe >= 95%, yaitu praktis tidak ada penghapusan.
#
#   A-P2 (mekanisme) korelasi(D(z), u) tetap POSITIF sampai akhir pelatihan.
#                    GUGUR bila korelasi < -0,5 -> jalan kabur pembalikan monoton
#                    terjadi; analisis permutasi tetap berlaku tetapi perbaikan ini
#                    tidak bekerja.
#
#   A-P3 (keamanan)  Win-F1 dan F1 per-kelas TIDAK turun; kelas r1b dipantau khusus
#                    (sideband terlemah, kelas yang paling rentan bila GRL menggosok
#                    residu kerusakan).
#                    GUGUR bila r1b turun > 5 poin sementara kelas lain tetap.
#
#   A-P4 (stabilitas) MSE domain tidak divergen sepanjang 50 epoch.
#                    GUGUR bila tumbuh tak terbatas -> butuh pembatasan (mis. Huber).
#
# BILA A-P1 GUGUR SEMENTARA A-P2 LOLOS: target skalar tidak memberi diskriminator daya
# cukup untuk menekan backbone. Seluruh analisis permutasi (M3) — salah satu pilar
# diagnosis Sec. IV-D makalah — perlu ditinjau ulang, dan itu memengaruhi cara makalah ditulis.
# Itu sebabnya Uji A didahulukan meski kodenya paling kecil.
#
# ATURAN DAYA: n=8 PENAPIS, n=20 KLAIM (aturan pelaporan no.2).
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_domain_index.sh
#
#   Ulang satu tahap : rm logs/.done_domain_index/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_domain_index
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_domain_index
LOG=logs/domain_index_$(date +%Y%m%d_%H%M%S).log
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

# ---- TAHAP 1: n=8, indeks f2s, DI ATAS BT-DANN-SupCon ----
# Dijalankan di atas BT (bukan menggantikannya) karena normalisasi gradien sudah
# terbukti dan tetap diperlukan.
stage didx_f2s_bt_supcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --method dannsupcon $BT $SC --domain_index f2s --name DIDX_f2s_BT_dannsupcon

# ---- TAHAP 2: ablasi `torque` — TIDAK otomatis ----
# u[torque] hanya beda maks 0,058 dari u[f2s] (r = 0,998), jadi ablasi ini menguji
# perbedaan yang sangat tipis. Jalankan HANYA bila TAHAP 1 memberi efek terukur pada
# A-P1; kalau f2s saja tidak menggerakkan probe, ablasi variabelnya tidak informatif.
# stage didx_torque_bt_supcon_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
#     --method dannsupcon $BT $SC --domain_index torque --name DIDX_torque_BT_dannsupcon

echo "" | tee -a "$LOG"
echo "=== TAHAP n=8 SELESAI ===" | tee -a "$LOG"
echo "A-P2/A-P4 dibaca dari log: grep 'GRL-REG-EPOCH\\|corr(D(z),u)' $LOG | tail -60" | tee -a "$LOG"
echo "A-P1/A-P3 butuh audit: $PY scripts/audit/audit_ckpt_ext.py (arahkan ke ckpt DIDX)" | tee -a "$LOG"
echo "Baca terhadap A-P1..A-P4 di kepala berkas ini SEBELUM memutuskan n=20." | tee -a "$LOG"
