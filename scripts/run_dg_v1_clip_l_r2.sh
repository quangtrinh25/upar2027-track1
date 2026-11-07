#!/usr/bin/env bash
# CLIP ViT-L/14 Arm A, attempt 2 (attempt 1 stopped by the user for a reboot after 25 min; preserved as failed).
set -u
cd "$(dirname "$0")/.."
PY=.venv-upgrade/bin/python
export OMP_NUM_THREADS=8
$PY -u scripts/dg_v1.py train --arm clip_l --data A --attempt 2 || { echo "TRAIN clip_l r2 FAILED"; exit 1; }
$PY -u scripts/dg_v1.py predict --data A --arms clip_l
