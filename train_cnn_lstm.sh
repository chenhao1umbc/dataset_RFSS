#!/bin/bash
# CNN-LSTM: 2-source → 3-source → 4-source (sequential within model)
set -e
cd "$(dirname "$0")"
echo "=== CNN-LSTM start ===" && date
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
echo "=== CNN-LSTM DONE ===" && date
