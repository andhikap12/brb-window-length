#!/usr/bin/env bash
# (1) DANN+SupCon (GRL core + SupCon stabilizer)  (2) DSRAN post-fix (A+C+B1)
set -u
set -o pipefail
PY=${PYTHON:-python}
SEEDS="42,123,2025,7,17,2024,2026,88"
COMMON="--backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --seeds $SEEDS"

echo "=== [1/2] DANN+SupCon: lambda_max=1.0 lambda_sc=1.0 tau=0.07 ==="
$PY scripts/train_motor_dann.py $COMMON --method dannsupcon --lambda_max 1.0 \
    --supcon_lambda 1.0 --supcon_temp 0.07 --supcon_proj_dim 128 --name DANNSupCon_l10_sc10

echo ""
echo "=== [2/2] DSRAN post-fix (sep+recon+adv): lambda_max=1.0 ortho1/dom1/recon1 ==="
$PY scripts/train_motor_dann.py $COMMON --method dsran --lambda_max 1.0 \
    --dsn_lambda_domain 1.0 --dsn_lambda_ortho 1.0 --dsn_lambda_recon 1.0 --name DSRANfix_l10_o1d1r1

echo "=== DONE ==="
