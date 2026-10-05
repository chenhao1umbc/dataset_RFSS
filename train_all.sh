#!/bin/bash
# Final training runs for the RFSS benchmark: one line per run in the lane lists below.
#
# Usage (from the project root):
#   bash train_all.sh LANE EPOCHS            LANE is cpu, mps, mps2 (the mps list in reverse order) or all
#   SEEDS_2SRC="1 2" bash train_all.sh cpu 10    2-source seeds to train (seed 0 exists as a pilot at 10 epochs)
#   SEEDS_2SRC="0 1 2" bash train_all.sh mps 20  all three seeds
#   SECONDARY=1 bash train_all.sh mps 20         also Conv-TasNet-L256 and CNN-LSTM-tconv (2-source, seed 0)
#   DRY_RUN=1 bash train_all.sh all 10           print the commands only
#   bash train_all.sh link 10                    link the finished 10-epoch pilots as seed 0 of the 2-source runs
#   bash train_all.sh link 20                    link the 20-epoch STFT-BLSTM probe as its seed 0
#
# The cpu lane (STFT-BLSTM) and the mps lane (the other models) can run side by side; mps2 walks the mps list from the
# back, so a second process shortens the mps list. A run is skipped when it has a DONE file or its last-epoch checkpoint,
# or when a process with the same --checkpoint-dir is already running, so two lanes never train the same run.
# Each run writes final/<model>_<n>src_seed<seed>/{ckpt,tb,log.txt}. An interrupted run restarts from scratch.
# 3-source and 4-source runs use seed 0 only. Learning rates are the validation-chosen ones.
#
# Evaluate with, for example:
#   uv run python check/eval_all.py --dl stft_blstm --skip-classical --tag _seed1 \
#       --ckpt-dir-format final/{name}_{n}src_seed1/ckpt

set -e
cd "$(dirname "$0")"

LANE=${1:?lane: cpu, mps, all or link}
EPOCHS=${2:?epochs}
SEEDS_2SRC=${SEEDS_2SRC:-"0 1 2"}

run() {
    local model=$1 lr=$2 nsrc=$3 seed=$4
    local dir=final/${model}_${nsrc}src_seed${seed}
    if [ -f "${dir}/DONE" ] || [ -f "${dir}/ckpt/keep_epoch_$(printf '%03d' $((EPOCHS - 1))).pt" ]; then
        echo "skip ${dir} (done)"
        return
    fi
    if pgrep -f -- "--checkpoint-dir ${dir}/ckpt" > /dev/null; then
        echo "skip ${dir} (running)"
        return
    fi
    local cmd="uv run python -u src/train.py --model ${model} --n-sources ${nsrc} --epochs ${EPOCHS} \
--lr ${lr} --seed ${seed} --batch-size 8 --train-length 7680 --num-workers 0 --keep-epochs ${EPOCHS} \
--checkpoint-dir ${dir}/ckpt --log-dir ${dir}/tb"
    if [ -n "${DRY_RUN}" ]; then
        echo "${cmd}"
        return
    fi
    mkdir -p "${dir}"
    echo "start ${dir} $(date -u +%FT%TZ)"
    if ! ${cmd} > "${dir}/log.txt" 2>&1; then
        echo "FAILED ${dir}"
        tail -n 20 "${dir}/log.txt"
        exit 1
    fi
    touch "${dir}/DONE"
    echo "done ${dir} $(date -u +%FT%TZ)"
}

lane_cpu() {
    for seed in ${SEEDS_2SRC}; do run stft_blstm 3e-4 2 "${seed}"; done
    run stft_blstm 3e-4 3 0
    run stft_blstm 3e-4 4 0
}

mps_runs() {
    for seed in ${SEEDS_2SRC}; do echo "dprnn 1e-3 2 ${seed}"; done
    for seed in ${SEEDS_2SRC}; do echo "conv_tasnet 3e-4 2 ${seed}"; done
    echo "dprnn 1e-3 3 0"
    echo "dprnn 1e-3 4 0"
    echo "conv_tasnet 3e-4 3 0"
    echo "conv_tasnet 3e-4 4 0"
    if [ -n "${SECONDARY}" ]; then
        echo "conv_tasnet_l256 1e-4 2 0"
        echo "cnn_lstm_tconv 3e-4 2 0"
    fi
}

lane_mps() {
    mps_runs | while read -r model lr nsrc seed; do run "${model}" "${lr}" "${nsrc}" "${seed}" < /dev/null; done
}

lane_mps2() {
    mps_runs | tac | while read -r model lr nsrc seed; do run "${model}" "${lr}" "${nsrc}" "${seed}" < /dev/null; done
}

# The pilots used the same arguments as a final run with seed 0 (train.py --seed 0), so they are linked into the
# final/ layout instead of being trained again.
link_pilot() {
    local pilot=$1 model=$2 last_epoch=$3
    local ckpt=pilots/${pilot}/ckpt/keep_epoch_$(printf '%03d' $((last_epoch - 1))).pt
    if [ ! -f "${ckpt}" ]; then
        echo "missing ${ckpt}: pilot ${pilot} is not finished, not linked"
        return
    fi
    touch "pilots/${pilot}/DONE"
    mkdir -p final
    ln -sfn "../pilots/${pilot}" "final/${model}_2src_seed0"
    echo "linked final/${model}_2src_seed0 -> pilots/${pilot}"
}

link_all() {
    if [ "${EPOCHS}" = "10" ]; then
        link_pilot stft_lr3e-4 stft_blstm 10
        link_pilot dprnn_lr1e-3 dprnn 10
        link_pilot l16_lr3e-4 conv_tasnet 10
        link_pilot l256_lr1e-4 conv_tasnet_l256 10
        link_pilot tconv_lr3e-4 cnn_lstm_tconv 10
    elif [ "${EPOCHS}" = "20" ]; then
        link_pilot stft_lr3e-4_20ep stft_blstm 20
    fi
}

case "${LANE}" in
    link) link_all ;;
    cpu) lane_cpu ;;
    mps) lane_mps ;;
    mps2) lane_mps2 ;;
    all) lane_cpu; lane_mps ;;
    *) echo "unknown lane ${LANE}"; exit 1 ;;
esac
