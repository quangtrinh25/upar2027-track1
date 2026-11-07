#!/usr/bin/env bash
# Round 4 large models with the gate-passing recipe (pilot llrd: tune CA +.0019, mADM +.0047 vs flat DINOv2-B).
# v5 pilot skipped (user decision 2026-10-19, train cap 40h -> 50h).
set -u
cd "$(dirname "$0")/.."
PY=.venv-upgrade/bin/python
export OMP_NUM_THREADS=8
$PY -u scripts/dg_v1.py train --arm clip_l_llrd --data A && $PY -u scripts/dg_v1.py predict --data A --arms clip_l_llrd || echo "CLIP-L llrd FAILED (continuing)"
$PY -u scripts/dg_v1.py train --arm dinov2_l_llrd --data A && $PY -u scripts/dg_v1.py predict --data A --arms dinov2_l_llrd || echo "DINOv2-L llrd FAILED"
