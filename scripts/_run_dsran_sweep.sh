#!/usr/bin/env bash
# DSRAN-combo sweep: separation (best DSN o3/d05) + adversarial GRL on z_task.
set -u
set -o pipefail
PY=${PYTHON:-python}
SEEDS="42,123,2025,7,17,2024,2026,88"
COMMON="--method dsran --backbone tsitplus --extremes_out --no_augment --fixed_budget --fp32 --epochs 50 --seeds $SEEDS --dsn_lambda_ortho 3.0 --dsn_lambda_domain 0.5"

echo "=== [1/2] DSRAN-combo: lambda_max=1.0 ortho=3.0 domain=0.5 ==="
$PY scripts/train_motor_dann.py $COMMON --lambda_max 1.0 --name DSRAN_l10_o3_d05

echo ""
echo "=== [2/2] DSRAN-combo: lambda_max=0.5 ortho=3.0 domain=0.5 ==="
$PY scripts/train_motor_dann.py $COMMON --lambda_max 0.5 --name DSRAN_l05_o3_d05

echo "=== DONE ==="
