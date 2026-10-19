#!/usr/bin/env bash
# _run_w16384_method.sh — UJI F: apakah lever METODE masih menyumbang setelah lever
# DATA dipakai? SupCon dan BT-DANN+SupCon dijalankan ulang di jendela w16384-ovl.
# TAHAN-MATI.
#
# =====================================================================================
# MENGAPA UJI INI ADA
# =====================================================================================
# Uji E (2026-08-02 23:18) membuktikan satu keputusan preprocessing — jendela 16384
# dengan tumpang tindih 75% — memberi +12,51 poin torsi-0 (p = 0,0274; 7/8) dan
# +5,70 poin F1 total (p = 0,0154; 6/8) memakai ERM POLOS, tanpa domain adaptation
# sama sekali. Itu lebih jauh daripada seluruh lever metode di proyek ini, dan
# signifikan — yang tak pernah dicapai lever metode mana pun.
#
# Batasan Uji E no. 2 menyatakan secara eksplisit: "Hanya ERM yang diuji di jendela
# baru. Klaim 'mengalahkan semua metode' BELUM sah sampai metode-metode itu
# dijalankan ulang di w16384-ovl." Uji F menutup lubang itu.
#
# Pertanyaan yang dijawab: apakah keuntungan metode selama ini generalisasi sejati,
# atau tambalan atas keterbatasan jendela yang kini sudah diperbaiki?
#
#   TIDAK ADA CABANG YANG SIA-SIA
#   - Bila metode TIDAK lagi unggul di jendela yang benar: temuan jauh lebih kuat
#     daripada +3,02 yang sekarang — keuntungan metode terbukti kompensasi atas cacat
#     preprocessing, bukan sumbangan generalisasi. Ini menentukan bingkai makalah.
#   - Bila metode MASIH unggul: metode usulan justru menguat, karena berdiri di atas
#     baseline yang jauh lebih tinggi (95,10 vs 89,40 F1).
#
# =====================================================================================
# PEMBANDING — ERM w16384-ovl, 8 seed BERPASANGAN yang SAMA (audit_erm_w16384ovl_s8.npz)
# =====================================================================================
#   torsi-0  91,15 +/- 11,1   |  torsi-7  97,85 +/- 2,9
#   F1 total 95,10 +/-  4,4   |  probe    98,78%   <- TERTINGGI di seluruh proyek
#
# Angka w4096 mana pun DILARANG dipakai sebagai pembanding di sini (batasan Uji E
# no. 5: basis window berbeda, 12.000 vs 13.200, tidak sebanding).
#
# Acuan dari jendela LAMA, dicatat hanya sebagai latar — BUKAN pembanding:
#   SupCon             +2,21 vs ERM w4096 (p = 0,162; tidak signifikan, n=20)
#   BT-DANN+SupCon r0.3 +3,02 vs DANN+SupCon w4096 (p = 0,019, n=8)
#
# =====================================================================================
# KRITERIA — DIDAFTARKAN DI MUKA, SEBELUM ANGKA TERLIHAT
# Ambang tunggal tanpa zona mati (pelajaran A-P1); sumbu varians dilaporkan terpisah
# (pelajaran D-P2); daya dihitung sebelum mengusulkan perluasan (pelajaran Uji D);
# pembanding selalu dalam jendela yang sama pada seed yang sama (batasan Uji E no. 5).
# =====================================================================================
#
#   F-P1 (utama) — dinilai TERPISAH untuk masing-masing metode, satu vonis per metode.
#
#       LOLOS bila  F1 total(metode) > F1 total(ERM w16384-ovl) pada >= 6 dari 8 seed
#                   berpasangan.
#       GUGUR bila  tidak.
#
#       Ambang 6/8 identik dengan E-P1 dan CTRL-P1 — dipakai ulang justru agar tidak
#       bisa disetel setelah melihat hasil. Ambang ini membelah habis ruang hasil:
#       tidak ada keluaran yang jatuh di zona "tak berkesimpulan".
#
#       Metrik utamanya F1 TOTAL, bukan torsi-0. Alasannya: yang ditanya adalah apakah
#       lever metode menyumbang secara umum. torsi-0 adalah sumbu tempat lever DATA
#       bekerja; memakainya sebagai sumbu utama akan menilai metode di lapangan yang
#       sudah dipanen lever lain.
#
#   F-P2 (atribusi) — aturan penafsiran, didaftarkan SEBELUM hasil, tanpa ambang.
#
#       Bila F-P1 GUGUR untuk BT-DANN+SupCon, letak hilangnya keunggulan menentukan
#       penafsiran, dan penafsirannya sudah dikunci di sini:
#         - hilang TERPUSAT di torsi-0  -> keunggulan lama adalah KOMPENSASI atas
#           keterbatasan resolusi; begitu resolusi diperbaiki, tak ada lagi yang
#           ditambal. Ini yang diprediksi.
#         - hilang MERATA di kedua torsi -> efek LANGIT-LANGIT (ERM sudah 95,10;
#           ruang tersisa 4,90 poin). Bukan bukti metode tidak berguna.
#       Yang dilaporkan: Delta per-torsi, bukan vonis lulus/gagal.
#
#       Catatan ruang-kepala yang wajib disebut: pada w4096 ruang tersisa 10,60 poin,
#       di sini 4,90 poin. Delta absolut TIDAK sebanding lintas jendela. Rasio
#       Delta/ruang-tersisa dilaporkan sebagai angka pendamping — dan hanya itu, karena
#       rumusan R pernah cacat sekali (pelajaran C-P1) dan tidak dipakai sebagai ambang.
#
#   F-P3 (mekanisme — ini yang bisa MENCABUT dugaan korelasi terbalik) — probe backbone-beku.
#
#       Uji E menemukan probe naik 96,16% -> 98,78% justru pada konfigurasi yang
#       memperbaiki masalahnya, sehingga muncul dugaan bahwa keterbacaan beban
#       berkorelasi TERBALIK dengan hasil. Dugaan itu berdiri di atas SATU titik data.
#       (Sec. IV-D makalah akhirnya menguji dan menolak mediasinya.)
#       Uji F memberi titik kedua, dan bisa merobohkannya:
#         - probe TURUN dan F1 IKUT TURUN  -> klaim korelasi terbalik dapat dukungan
#           kedua yang independen.
#         - probe TURUN tetapi F1 NAIK     -> klaim korelasi terbalik GUGUR dan WAJIB
#           dicabut, berapa pun hasil F-P1.
#         - probe TIDAK turun              -> metode adversarial terbukti tidak
#           menghapus info domain di jendela ini; catat sebagai kegagalan mekanisme,
#           terpisah dari kegagalan hasil.
#       Probe WAJIB dari audit backbone-beku, tidak pernah domain_acc daring (aturan
#       pelaporan no. 1; metrik daring terbukti meleset 0,79 pada M1).
#
#   F-P4 (varians) — dilaporkan TERLEPAS dari F-P1, tidak boleh dilebur ke dalamnya.
#
#       Klaim utama BT-DANN sejak awal adalah penurunan varians, bukan kenaikan rerata
#       (sd 5,80 -> 1,71 di w4096). sd F1 ERM w16384-ovl = 4,4.
#       LOLOS-varians bila sd(metode) < sd(ERM w16384-ovl) dengan uji-F p < 0,05.
#       Vonis ini berdiri sendiri: metode boleh GUGUR di F-P1 tetapi LOLOS di F-P4,
#       dan itu tetap temuan yang dilaporkan (pelajaran D-P2 — Uji D memperbaiki
#       varians tanpa memperbaiki rerata).
#
#   DAYA — dihitung SEBELUM hasil, agar perluasan ke n=20 tidak diusulkan reaktif.
#       Bila |d| yang terukur >= 0,95, n=11 sudah berdaya 0,80 dan n=8 hampir memadai
#       (seperti Uji E, d = 0,982) -> perluasan S12 dibenarkan.
#       Bila |d| < 0,50, n=20 hanya berdaya < 0,45 -> perluasan DILARANG, seperti
#       Uji D yang sengaja dilewati (d = 0,397; daya n=20 = 0,393).
#
# PRASYARAT
#   Dataset sudah ada (dipakai Uji E):
#     data/processed/motor/motor_multiclass_w16384_s4096_ss_env.npz
#   Sanity BT-DANN sudah lolos 11/11 (2026-08-02) — arsitektur tidak berubah, hanya
#   dataset. Tidak ada implementasi adversarial BARU di sini, jadi aturan pelaporan
#   no. 4 terpenuhi tanpa sanity ulang.
#
# MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_w16384_method.sh
#
#   Ulang satu tahap : rm logs/.done_w16384method/<nama_tahap>
#   Ulang semuanya   : rm -rf logs/.done_w16384method
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
NPZ="motor_multiclass_w16384_s4096_ss_env.npz"
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"
BT="--bt_ratio 0.3"

DONE=logs/.done_w16384method
LOG=logs/w16384method_$(date +%Y%m%d_%H%M%S).log
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

# ---- TAHAP 1: SupCon ----
# Didahulukan karena non-adversarial dan lever metode terbaik sejauh ini. Bila SupCon
# saja sudah tidak menyumbang di jendela yang benar, seluruh cabang adversarial di
# atasnya kehilangan dasar — dan itu terbaca lebih dulu.
stage supcon_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --method supcon $SC --name W16384OVL_supcon

# ---- TAHAP 2: BT-DANN+SupCon r0.3 — lengan BT, Sec. IV-A makalah ----
stage bt_dannsupcon_w16384ovl_s8 $PY scripts/train_motor_dann.py $BASE --seeds $S8 \
    --npz "$NPZ" --method dannsupcon $BT $SC --name W16384OVL_BT_dannsupcon_r03

echo "" | tee -a "$LOG"
echo "=== SEMUA TAHAP SELESAI ===" | tee -a "$LOG"
echo "Per-torsi + probe WAJIB dari audit backbone-beku dengan dataset yang COCOK:" | tee -a "$LOG"
echo "  $PY scripts/audit/audit_ckpt_ext.py \\" | tee -a "$LOG"
echo "      --methods 'UJI E: ERM w16384-ovl;UJI F: SupCon w16384-ovl;UJI F: BT+SupCon w16384-ovl' \\" | tee -a "$LOG"
echo "      --npz $NPZ --seeds $S8 --out results/motor/audit_method_w16384ovl_s8.npz" | tee -a "$LOG"
echo "Periksa rasio BT: grep 'BT-DANN.*-BAL ep' $LOG | tail -20" | tee -a "$LOG"
