#!/bin/bash
# Frozen evaluation passes of the final runs, each checkpoint exactly as scored in the validation tables.
#
# Usage (from the project root):
#   bash check/run_test_passes.sh 2                    2-source pass: 9 checkpoints, ICA, NMF, oracles, all test samples
#   bash check/run_test_passes.sh 34                   3- and 4-source pass: 6 checkpoints, ICA, NMF, oracles
#   SPLIT=val bash check/run_test_passes.sh 2 --n 30 --tag _smoke    smoke test on 30 validation samples per source count
# Extra arguments are passed to check/eval_all.py. Output: check/eval_all_src<list>_frozen_results.json (test split).
# Labels are <family>_s<seed>: stft = STFT-BLSTM, dprnn = DPRNN, conv = Conv-TasNet L=16. Seed 0 of the 2-source runs is the
# pilot (scored at its last epoch); every other run is scored at its best-validation epoch.

set -e
cd "$(dirname "$0")/.."

PASS=${1:?pass: 2 or 34}
SPLIT=${SPLIT:-test}

case "${PASS}" in
    2)
        SOURCES="2"
        RUNS=(
            stft_s0=stft_blstm:2:final/stft_blstm_2src_seed0/ckpt/keep_epoch_009.pt
            stft_s1=stft_blstm:2:final/stft_blstm_2src_seed1/ckpt/epoch_009_loss_-1.7889.pt
            stft_s2=stft_blstm:2:final/stft_blstm_2src_seed2/ckpt/epoch_008_loss_-1.8602.pt
            dprnn_s0=dprnn:2:final/dprnn_2src_seed0/ckpt/keep_epoch_009.pt
            dprnn_s1=dprnn:2:final/dprnn_2src_seed1/ckpt/epoch_009_loss_-1.5551.pt
            dprnn_s2=dprnn:2:final/dprnn_2src_seed2/ckpt/epoch_009_loss_-1.4313.pt
            conv_s0=conv_tasnet:2:final/conv_tasnet_2src_seed0/ckpt/keep_epoch_009.pt
            conv_s1=conv_tasnet:2:final/conv_tasnet_2src_seed1/ckpt/epoch_008_loss_-1.3647.pt
            conv_s2=conv_tasnet:2:final/conv_tasnet_2src_seed2/ckpt/epoch_008_loss_-1.1304.pt
        )
        ;;
    34)
        SOURCES="3 4"
        RUNS=(
            stft_s0=stft_blstm:3:final/stft_blstm_3src_seed0/ckpt/epoch_009_loss_4.2458.pt
            dprnn_s0=dprnn:3:final/dprnn_3src_seed0/ckpt/epoch_009_loss_4.2910.pt
            conv_s0=conv_tasnet:3:final/conv_tasnet_3src_seed0/ckpt/epoch_008_loss_4.5104.pt
            stft_s0=stft_blstm:4:final/stft_blstm_4src_seed0/ckpt/epoch_008_loss_7.4714.pt
            dprnn_s0=dprnn:4:final/dprnn_4src_seed0/ckpt/epoch_006_loss_8.4403.pt
            conv_s0=conv_tasnet:4:final/conv_tasnet_4src_seed0/ckpt/epoch_009_loss_8.3755.pt
        )
        ;;
    *) echo "unknown pass ${PASS}"; exit 1 ;;
esac

uv run python -u check/eval_all.py --split "${SPLIT}" --sources ${SOURCES} --tag _frozen --runs "${RUNS[@]}" "${@:2}"
