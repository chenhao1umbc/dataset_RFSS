"""
Tables of the frozen test pass, read from the per-sample rows of check/eval_all_src<list>_frozen_results.json.

Prints, per source count, the gain over the input mixture (method score minus input score, dB) of every method with a
95 percent bootstrap interval in the bins all, adjacent-channel SNR>20 and co-channel SNR>20, and the matched-seed paired
differences of the three families (label prefixes stft, dprnn, conv, same seed number) with paired bootstrap intervals.

Usage:
    uv run python check/test_summary.py 2          # 2-source pass, first 7,680 samples of every signal
    uv run python check/test_summary.py 34 0       # 3- and 4-source pass, crop seed 0 (primary pass)
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
N_RESAMPLES = 5000
SEED = 0
BINS = {
    "all": lambda r: True,
    "adjacent_snr_gt_20": lambda r: r["mode"] == "adjacent-channel" and r["snr_db"] > 20,
    "co_snr_gt_20": lambda r: r["mode"] == "co-channel" and r["snr_db"] > 20,
}
PAIRS = [("stft", "dprnn"), ("dprnn", "conv"), ("stft", "conv")]


def interval(values: np.ndarray, rng: np.random.RandomState) -> str:
    boot = values[rng.randint(0, len(values), size=(N_RESAMPLES, len(values)))].mean(axis=1)
    return f"{values.mean():+.2f} [{np.percentile(boot, 2.5):+.2f}, {np.percentile(boot, 97.5):+.2f}]"


def main():
    crop = f"_crop{sys.argv[2]}" if len(sys.argv) > 2 else ""
    result = json.loads((ROOT / "check" / f"eval_all_src{sys.argv[1]}{crop}_frozen_results.json").read_text())
    print(f"commit {result['git_commit']}, code modified {result['git_code_modified']}, samples {len(result['samples'])}")
    rng = np.random.RandomState(SEED)
    for n_sources in sorted({r["num_sources"] for r in result["samples"]}):
        rows = [r for r in result["samples"] if r["num_sources"] == n_sources]
        methods = [m for m in rows[0] if m not in ("idx", "num_sources", "mode", "snr_db", "length", "input")]
        for bin_name, keep in BINS.items():
            sel = [r for r in rows if keep(r)]
            print(f"\n{n_sources}-source, bin {bin_name}, n={len(sel)}: gain over input (dB), 95% bootstrap interval")
            for m in methods:
                print(f"  {m:12s} {interval(np.array([r[m] - r['input'] for r in sel]), rng)}")
            seeds = sorted({m.rsplit('_s', 1)[1] for m in methods if '_s' in m})
            for a, b in PAIRS:
                for s in seeds:
                    if f"{a}_s{s}" in methods and f"{b}_s{s}" in methods:
                        print(f"  {a} minus {b}, seed {s}: {interval(np.array([r[f'{a}_s{s}'] - r[f'{b}_s{s}'] for r in sel]), rng)}")


if __name__ == "__main__":
    main()
