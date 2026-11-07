#!/usr/bin/env bash
# dg_v1 Arm A (USER-RUN, one GPU job at a time): GPU smoke of the base models, independent leakage check, then
# Tiny bce -> Tiny weighted -> DINOv2-B -> ConvNeXtV2-B, then predictions.
# A failed arm is preserved (journal + folder) and the chain continues; no retry.
set -u
cd "$(dirname "$0")/.."
PY=.venv-upgrade/bin/python
export OMP_NUM_THREADS=8
for arm in convnextv2_b dinov2_b; do
  $PY -u scripts/dg_v1.py smoke --arm "$arm" || { echo "GPU SMOKE $arm FAILED: stop"; exit 1; }
done
$PY -u scripts/dg_v1.py leakage || { echo "LEAKAGE STEP FAILED OR NEEDS A DECISION: stop"; exit 1; }
for arm in tiny_bce tiny_weighted dinov2_b convnextv2_b; do
  $PY -u scripts/dg_v1.py train --arm "$arm" --data A || echo "ARM $arm FAILED (preserved, continuing)"
done
$PY -u scripts/dg_v1.py predict --data A
$PY -u scripts/dg_v1.py predict-incumbent
