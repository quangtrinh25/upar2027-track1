#!/usr/bin/env bash
# DINOv2-L Arm A, attempt 2 (attempt 1 was cut by a host power loss after epoch 4; preserved as failed).
set -u
cd "$(dirname "$0")/.."
PY=.venv-upgrade/bin/python
export OMP_NUM_THREADS=8
$PY -u scripts/dg_v1.py train --arm dinov2_l --data A --attempt 2 || { echo "TRAIN dinov2_l r2 FAILED"; exit 1; }
$PY -u scripts/dg_v1.py predict --data A --arms dinov2_l
