#!/bin/bash
# DPRNN: 2-source → 3-source → 4-source (sequential within model)
set -e
cd "$(dirname "$0")"
echo "=== DPRNN start ===" && date
for NSRC in 2 3 4; do
    echo "--- DPRNN ${NSRC}-source ---"
    uv run python -u src/train.py \
      --model dprnn --n-sources ${NSRC} \
      --epochs 30 --batch-size 8 --lr 1e-3 --train-length 7680 \
      --device auto --num-workers 0 \
      --log-dir runs/dprnn_${NSRC}src \
      --checkpoint-dir checkpoints/dprnn_${NSRC}src
    echo ""
done
echo "=== DPRNN DONE ===" && date
