#!/usr/bin/env bash
# _run_byvar_seeds.sh — TAHAN-MATI. Aman dijalankan ulang: tahap yang sudah selesai dilewati.
#
# TAHAP 1 — uji by_var=True (prasyarat; ~1 jam)
#   Standardisasi per-sampel DAN per-kanal. Default berbagi satu mean/std lintas keenam
#   kanal, menekan Vib_axial ke 3.9% dan envelope ke 5.7% (Sec. IV-C makalah CENIM 2026).
#   Pembanding: ERM 89.40 +/- 4.18, DANN lam=1.0 86.62 +/- 3.13.
#
# TAHAP 2 — perluasan seed 8 -> 20 (~3.3 jam)
#   12 seed BARU + seed 42 & 123 sebagai kontrol reproduksibilitas. Bila ERM seed-42
#   mereproduksi 97.73, penggabungan dengan 8 seed lama sah. Daya uji 0.40 -> ~0.84.
#
# CARA MELANJUTKAN SETELAH LAPTOP DIMATIKAN:
#   cd <repository root>
#   bash scripts/_run_byvar_seeds.sh          # tahap selesai dilewati otomatis
#
#   Ulang satu tahap saja  : rm logs/.done_byvar_seeds/<nama_tahap>
#   Ulang semuanya         : rm -rf logs/.done_byvar_seeds
set -u
set -o pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

S8="42,123,2025,7,17,2024,2026,88"
S14="42,123,1,3,5,11,13,23,31,57,99,101,202,777"   # 2 kontrol reproduksibilitas + 12 baru
BASE="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50"
SC="--supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128"

DONE=logs/.done_byvar_seeds
LOG=logs/byvar_seeds_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$DONE" logs

# jalankan satu tahap; lewati bila penandanya sudah ada
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

# ---- TAHAP 1: by_var ----
stage byvar_erm  $PY scripts/train_motor_dann.py $BASE --seeds $S8 --by_var \
    --lambda_max 0.0 --name BYVAR_baseline

stage byvar_dann $PY scripts/train_motor_dann.py $BASE --seeds $S8 --by_var \
    --lambda_max 1.0 --name BYVAR_dann_lam10

# ---- TAHAP 2: perluasan seed (standardisasi DEFAULT) ----
stage s20_erm        $PY scripts/train_motor_dann.py $BASE --seeds $S14 \
    --lambda_max 0.0 --name S20_baseline

stage s20_dann       $PY scripts/train_motor_dann.py $BASE --seeds $S14 \
    --lambda_max 1.0 --name S20_dann_lam10

stage s20_supcon     $PY scripts/train_motor_dann.py $BASE --seeds $S14 --method supcon $SC \
    --name S20_supcon

stage s20_dannsupcon $PY scripts/train_motor_dann.py $BASE --seeds $S14 --method dannsupcon \
    --lambda_max 1.0 $SC --name S20_dannsupcon

echo "" | tee -a "$LOG"
echo "=== SEMUA TAHAP SELESAI ===" | tee -a "$LOG"
