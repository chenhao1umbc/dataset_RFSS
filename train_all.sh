#!/bin/bash
# Full Phase 5 training script for Mac Mini (MPS).
# Trains 3 models × 3 source counts = 9 experiments, ~40h total.
# Order: ConvTasNet (18.5h) → DPRNN (8.5h) → CNN-LSTM (22h)
# ConvTasNet and DPRNN are the primary paper results; CNN-LSTM can be aborted
# if not needed without losing the first two.
#
# Run from project root:
#   nohup bash train_all.sh >> runs/train_all_v2.log 2>&1 &
#
# Monitor:
#   tail -f runs/train_all_v2.log

set -e
cd "$(dirname "$0")"

uv run python -c "
import torch
assert torch.backends.mps.is_available(), 'MPS not available'
print(f'MPS available: True | torch: {torch.__version__}')
"

echo ""
echo "=== ConvTasNet (30 epochs, CosineAnnealingLR) ==="

for NSRC in 2 3 4; do
    echo "--- ConvTasNet ${NSRC}-source ---"
    uv run python -u src/train.py \
      --model conv_tasnet --n-sources ${NSRC} \
      --epochs 30 --batch-size 8 --lr 1e-3 --train-length 7680 \
      --device auto --num-workers 0 \
      --log-dir runs/conv_tasnet_${NSRC}src \
      --checkpoint-dir checkpoints/conv_tasnet_${NSRC}src
    echo ""
done

echo "=== DPRNN (30 epochs, CosineAnnealingLR) ==="

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

echo "=== CNN-LSTM (30 epochs, CosineAnnealingLR) ==="

for NSRC in 2 3 4; do
    echo "--- CNN-LSTM ${NSRC}-source ---"
    uv run python -u src/train.py \
      --model cnn_lstm --n-sources ${NSRC} \
      --epochs 30 --batch-size 8 --lr 1e-3 --train-length 7680 \
      --device auto --num-workers 0 \
      --log-dir runs/cnn_lstm_${NSRC}src \
      --checkpoint-dir checkpoints/cnn_lstm_${NSRC}src
    echo ""
done

echo "=== ALL DONE ==="
