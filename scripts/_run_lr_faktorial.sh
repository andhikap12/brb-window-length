#!/usr/bin/env bash
# _run_lr_faktorial.sh — faktorial 2x2 R3 diulang pada LAJU BELAJAR KEDUA.
#
# MENGAPA
# =======
# Seluruh 15 lengan Tabel I, faktorial R3, dan replikasi TSSequencerPlus berjalan di
# SATU titik hiperparameter: lr 1e-3, batch 64, 50 epoch (src/config.py TRAIN_CONFIG).
# Sapuan atas seluruh skrip run mengonfirmasi --lr tidak pernah dilewatkan sekali pun;
# satu-satunya --batch_size 32 pada riset ini dipakai di lengan eksplorasi w65536, yang
# tidak dilaporkan makalah dan tidak dirilis: jendela 65.536 sampel tidak muat di kartu
# 6,4 GB, jadi itu paksaan memori dan bukan variasi terancang.
#
# Menahan hiperparameter tetap adalah KEKUATAN untuk validitas internal: itulah yang
# membuat Sec. III-G makalah boleh bilang beda antar-baris berasal dari objektif dan bukan
# dari penyetelan. Yang belum diuji adalah validitas eksternalnya. Jendela panjang punya
# urutan 4x lebih panjang dengan anggaran latih yang sama, jadi kalau 1e-3 kebetulan lebih
# cocok untuk urutan panjang, sebagian "efek durasi" sebenarnya efek penyetelan.
#
# Kampanye ini menutup lubang itu dengan bentuk yang SENGAJA dijiplak dari
# _run_tsseq_faktorial.sh: seed sama, flag sama, epoch sama, npz sama, backbone kembali ke
# TSiTPlus. Yang berubah HANYA --lr. Dengan begitu hasilnya sebanding baris-per-baris
# dengan faktorial asli dan dapat dilaporkan dengan pola kalimat yang sama.
#
# YANG DIPERTARUHKAN
# ==================
# Ini satu-satunya eksperimen dalam antrean yang BISA MEMBATALKAN KLAIM UTAMA makalah.
# Pembacaan dikunci di muka, 8 Agu 2026 sebelum satu angka pun ada, atas efek utama
# PANJANG (rata-rata w16384 dikurangi rata-rata w4096 pada kedua tingkat volume):
#
#     >= +10        bertahan; klaim bukan artefak penyetelan
#     +3 sampai +10 arah bertahan, besaran bergantung konfigurasi -> wajib masuk batasan
#     <  +3         klaim panjang TERIKAT pada 1e-3 -> temuan serius, makalah harus berubah
#
# Pembanding pada lr 1e-3: efek utama panjang +15,88 dan volume +7,62 (TSiTPlus),
# +14,75 dan +0,50 (TSSequencerPlus).
#
# CACAT DESAIN YANG DINYATAKAN DI MUKA, BUKAN DISEMBUNYIKAN
# =========================================================
# Menurunkan lr pada anggaran epoch TETAP mencampur "laju" dengan "jumlah latihan": kalau
# efeknya menyusut, lr dan kurang-latih tidak dapat dipisahkan dari kampanye ini sendiri.
# Tidak ada knob tunggal yang bersih pada anggaran epoch tetap, jadi cacat ini melekat.
# Ia harus ikut tertulis ke mana pun angka ini dibawa. Kalau hasilnya jatuh di bawah +3,
# langkah BERIKUTNYA adalah memisahkan keduanya lewat lr 3e-4 dengan epoch dinaikkan,
# bukan langsung menyimpulkan lr yang menyebabkannya.
#
# 3e-4 dipilih TURUN, bukan naik: 3e-3 berisiko divergen dan kegagalan divergen tidak
# informatif untuk pertanyaan ini.
#
# NAMA TIDAK BOLEH BENTROK. Checkpoint disimpan sebagai {name}_{backbone}_seed{s}_fold0.pth
# dan audit menurunkan backbone dari akhiran nama itu. Awalan LR3E4_ menjaga jalannya
# terpisah dari W4096SUB_baseline dkk milik lr 1e-3, yang TIDAK BOLEH tertimpa.

set -u
set -o pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
S8="42,123,2025,7,17,2024,2026,88"
LR=3e-4
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --lr $LR"

DONE=logs/.done_lr_faktorial
LOG=logs/lr_faktorial_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

# lengan -> berkas npz. Keempatnya sel faktorial R3, identik dengan kampanye TSSequencerPlus.
declare -A NPZ=(
  [w4096sub]="motor_multiclass_w4096_s4096_ss_env_sub3200.npz"
  [w16384]="motor_multiclass_w16384_s16384_ss_env.npz"
  [w4096full]="motor_multiclass_w4096_s4096_ss_env.npz"
  [w16384ovl]="motor_multiclass_w16384_s4096_ss_env.npz"
)
declare -A NAMA=(
  [w4096sub]="LR3E4_W4096SUB_baseline"
  [w16384]="LR3E4_W16384_baseline"
  [w4096full]="LR3E4_W4096FULL_baseline"
  [w16384ovl]="LR3E4_W16384OVL_baseline"
)

for k in "${!NPZ[@]}"; do
  if [ ! -f "data/processed/motor/${NPZ[$k]}" ]; then
    echo "[ABORT] ${NPZ[$k]} belum ada."
    exit 1
  fi
done

# Gerbang tabrakan: kalau checkpoint LR3E4 sudah ada dari jalan sebelumnya, penanda
# selesai yang mengurusnya. Yang dijaga di sini adalah checkpoint lr 1e-3 milik kampanye
# asli; kalau salah satu namanya bocor ke daftar di atas, hentikan sebelum menimpa.
for k in "${!NAMA[@]}"; do
  case "${NAMA[$k]}" in
    LR3E4_*) ;;
    *) echo "[ABORT] nama '${NAMA[$k]}' tanpa awalan LR3E4_ akan menimpa kampanye lr 1e-3."; exit 1 ;;
  esac
done

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
echo "lr = $LR (pembanding: 1e-3), backbone tsitplus, seed $S8" | tee -a "$LOG"

# Urutan dari termurah ke termahal, supaya salah konfigurasi terlihat dalam menit pertama
# alih-alih setelah satu jam.
for k in w4096sub w16384 w4096full w16384ovl; do
  stage "erm_${k}_lr3e4_s8" $PY scripts/train_motor_dann.py $BASE --seeds "$S8" \
      --npz "${NPZ[$k]}" --lambda_max 0.0 --name "${NAMA[$k]}"
done

echo "" | tee -a "$LOG"
echo "=== SELESAI ===" | tee -a "$LOG"
echo "Per-torsi WAJIB dari audit backbone-beku, satu perintah per lengan:" | tee -a "$LOG"
for k in w4096sub w16384 w4096full w16384ovl; do
  echo "  $PY scripts/audit/audit_ckpt_ext.py --npz ${NPZ[$k]} --seeds $S8 \\" | tee -a "$LOG"
  echo "      --out results/motor/audit_lr3e4_${k}_s8.npz" | tee -a "$LOG"
done
