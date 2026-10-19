#!/usr/bin/env bash
# CDAN (conditional adversarial) vs CDAN+SupCon — protokol IDENTIK dengan
# 8-seed EXT lain (bandingkan langsung ke: ERM 89.40, DANN 86.62, DANNSupCon 88.79).
set -u
set -o pipefail
PY=${PYTHON:-python}
SEEDS="42,123,2025,7,17,2024,2026,88"
COMMON="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --seeds $SEEDS"

echo "=== [1/2] CDAN+E: lambda_max=1.0, entropy conditioning ON ==="
$PY scripts/train_motor_dann.py $COMMON --method cdan --lambda_max 1.0 \
    --name CDAN_l10_ent

echo ""
echo "=== [2/2] CDAN+SupCon: lambda_max=1.0 lambda_sc=1.0 tau=0.07 ==="
$PY scripts/train_motor_dann.py $COMMON --method cdansupcon --lambda_max 1.0 \
    --supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128 --name CDANSupCon_l10_sc10

echo "=== DONE ==="
