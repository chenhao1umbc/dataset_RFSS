#!/bin/bash
# Final training runs for the RFSS benchmark: one line per run in the lane lists below.
#
# Usage (from the project root):
#   bash train_all.sh LANE EPOCHS            LANE is cpu, mps or all
#   SEEDS_2SRC="1 2" bash train_all.sh cpu 10    2-source seeds to train (seed 0 exists as a pilot at 10 epochs)
#   SEEDS_2SRC="0 1 2" bash train_all.sh mps 20  all three seeds
#   SECONDARY=1 bash train_all.sh mps 20         also Conv-TasNet-L256 and CNN-LSTM-tconv (2-source, seed 0)
#   DRY_RUN=1 bash train_all.sh all 10           print the commands only
#
# The cpu lane (STFT-BLSTM) and the mps lane (the other models) can run side by side.
# Each run writes final/<model>_<n>src_seed<seed>/{ckpt,tb,log.txt}; a finished run leaves a DONE file
# and is skipped on a restart. An interrupted run restarts from scratch.
# 3-source and 4-source runs use seed 0 only. Learning rates are the validation-chosen ones.
#
# Evaluate with, for example:
#   uv run python check/eval_all.py --dl stft_blstm --skip-classical --tag _seed1 \
#       --ckpt-dir-format final/{name}_{n}src_seed1/ckpt

set -e
cd "$(dirname "$0")"

LANE=${1:?lane: cpu, mps or all}
EPOCHS=${2:?epochs}
SEEDS_2SRC=${SEEDS_2SRC:-"0 1 2"}

run() {
    local model=$1 lr=$2 nsrc=$3 seed=$4
    local dir=final/${model}_${nsrc}src_seed${seed}
    if [ -f "${dir}/DONE" ]; then
        echo "skip ${dir} (DONE)"
        return
    fi
    local cmd="uv run python -u src/train.py --model ${model} --n-sources ${nsrc} --epochs ${EPOCHS} \
--lr ${lr} --seed ${seed} --batch-size 8 --train-length 7680 --num-workers 0 \
--checkpoint-dir ${dir}/ckpt --log-dir ${dir}/tb"
    if [ -n "${DRY_RUN}" ]; then
        echo "${cmd}"
        return
    fi
    mkdir -p "${dir}"
    echo "start ${dir} $(date -u +%FT%TZ)"
    ${cmd} > "${dir}/log.txt" 2>&1
    touch "${dir}/DONE"
    echo "done ${dir} $(date -u +%FT%TZ)"
}

lane_cpu() {
    for seed in ${SEEDS_2SRC}; do run stft_blstm 3e-4 2 "${seed}"; done
    run stft_blstm 3e-4 3 0
    run stft_blstm 3e-4 4 0
}

lane_mps() {
    for seed in ${SEEDS_2SRC}; do run dprnn 1e-3 2 "${seed}"; done
    for seed in ${SEEDS_2SRC}; do run conv_tasnet 3e-4 2 "${seed}"; done
    run dprnn 1e-3 3 0
    run dprnn 1e-3 4 0
    run conv_tasnet 3e-4 3 0
    run conv_tasnet 3e-4 4 0
    if [ -n "${SECONDARY}" ]; then
        run conv_tasnet_l256 1e-4 2 0
        run cnn_lstm_tconv 3e-4 2 0
    fi
}

case "${LANE}" in
    cpu) lane_cpu ;;
    mps) lane_mps ;;
    all) lane_cpu; lane_mps ;;
    *) echo "unknown lane ${LANE}"; exit 1 ;;
esac
