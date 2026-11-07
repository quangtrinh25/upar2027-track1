#!/usr/bin/env bash
# USER-RUN, 2026-10-18: two more large foundation members, Arm A only (train+val adds nothing on the hidden test).
# SigLIP-L/16-256 (smoke first), then CLIP-L seed 43. Predictions after each model so selection can start early.
set -u
cd "$(dirname "$0")/.."
PY=.venv-upgrade/bin/python
export OMP_NUM_THREADS=8
$PY -u scripts/dg_v1.py smoke --arm siglip_l && {
  $PY -u scripts/dg_v1.py train --arm siglip_l --data A && $PY -u scripts/dg_v1.py predict --data A --arms siglip_l || echo "SIGLIP-L FAILED (continuing)"
} || echo "SIGLIP-L SMOKE FAILED (skipping SigLIP-L)"
$PY -u scripts/dg_v1.py train --arm clip_l_s2 --data A && $PY -u scripts/dg_v1.py predict --data A --arms clip_l_s2 || echo "CLIP-L s2 FAILED"
