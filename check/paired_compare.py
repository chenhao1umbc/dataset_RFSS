"""
Paired bootstrap comparison of two runs scored on the same validation crops of check/encoder_sweep.py.

Both runs store the per-sample gain over the input mixture for the same 800 crops, so the difference of two runs
is resampled over crops (paired), which is much tighter than comparing two marginal intervals. A run is a key of
check/encoder_sweep_results.json, optionally followed by :EPOCH (1-based) for runs that store several epochs.

Usage:
    uv run python check/paired_compare.py stft_pilot_lr3e-4_ep5 dprnn_pilot_lr1e-3_ep5
    uv run python check/paired_compare.py stft_lr3e-4:4 l16_lr3e-4:3

Prints A minus B (dB) per bin with a 95 percent paired bootstrap interval.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from encoder_sweep import BINS, OUTPUT, load_items  # noqa: E402

N_RESAMPLES = 5000
SEED = 0


def per_sample_gain(results: dict, spec: str) -> np.ndarray:
    key, _, epoch = spec.partition(":")
    run = results[key]
    val = run["epochs"][int(epoch) - 1]["val"] if epoch else run["val"]
    return np.array(val["per_sample_gain_db"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-sources", type=int, default=2)
    args = ap.parse_args()

    results = json.loads(OUTPUT.read_text())
    gain_a, gain_b = per_sample_gain(results, args.run_a), per_sample_gain(results, args.run_b)
    _, _, info = load_items("val", args.n_val, args.n_sources)
    if not len(gain_a) == len(gain_b) == len(info):
        raise ValueError(f"crop counts differ: {len(gain_a)}, {len(gain_b)}, {len(info)}")
    diff = gain_a - gain_b
    rng = np.random.RandomState(SEED)
    print(f"{args.run_a} minus {args.run_b} (gain over input, dB; paired bootstrap, {N_RESAMPLES} resamples)")
    for name, keep in BINS.items():
        idx = np.array([i for i, (mode, snr) in enumerate(info) if keep(mode, snr)])
        boot = rng.choice(diff[idx], size=(N_RESAMPLES, len(idx))).mean(axis=1)
        print(f"  {name:20s} n={len(idx):3d}  {diff[idx].mean():+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]")


if __name__ == "__main__":
    main()
