"""
SNR-stratified PI-SI-SINR analysis.

Reads per-sample results from baseline_results.json and breakdown_results.json,
looks up snr_db from HDF5 metadata, then bins results by SNR to show how each
method performs across the 0-30 dB SNR range.

Outputs: check/snr_stratified_results.json
"""

import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent
BASELINE_JSON = ROOT / "check" / "baseline_results.json"
BREAKDOWN_JSON = ROOT / "check" / "breakdown_results.json"
DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "snr_stratified_results.json"

SNR_BINS = [(0, 10), (10, 20), (20, 30)]
SNR_LABELS = ["0-10 dB", "10-20 dB", "20-30 dB"]

METHODS = ["ica", "nmf", "cnn_lstm", "dprnn", "conv_tasnet"]
N_SOURCES = [2, 3, 4]


def load_snr_map(indices: list[int]) -> dict[int, float]:
    """Read snr_db from HDF5 metadata for a list of dataset indices."""
    snr_map = {}
    with h5py.File(str(DATASET_PATH), "r") as f:
        meta_ds = f["metadata"]
        for idx in indices:
            raw = meta_ds[idx]
            meta = json.loads(raw if isinstance(raw, str) else raw.decode())
            snr_map[idx] = meta["snr_db"]
    return snr_map


def load_all_samples() -> dict[str, list[dict]]:
    """Return {config_key: [{'idx', 'mixing_mode', 'si_sinr_db', 'snr_db'}]}."""
    all_samples: dict[str, list[dict]] = {}

    # Baselines: per-sample data in baseline_results.json
    if BASELINE_JSON.exists():
        raw = json.loads(BASELINE_JSON.read_text())
        for key, entry in raw.items():
            samples = entry.get("samples", [])
            indices = [s["dataset_idx"] for s in samples]
            snr_map = load_snr_map(indices)
            all_samples[key] = [
                {
                    "idx": s["dataset_idx"],
                    "mixing_mode": s["mixing_mode"],
                    "si_sinr_db": s["si_sinr_db"],
                    "snr_db": snr_map[s["dataset_idx"]],
                }
                for s in samples
            ]
        print(f"  Loaded baselines: {list(raw.keys())}")

    # DL models: per-sample data in breakdown_results.json
    if BREAKDOWN_JSON.exists():
        raw = json.loads(BREAKDOWN_JSON.read_text())
        for key, entry in raw.items():
            if "samples" not in entry:
                continue
            samples = entry["samples"]
            indices = [s["global_idx"] for s in samples]
            snr_map = load_snr_map(indices)
            all_samples[key] = [
                {
                    "idx": s["global_idx"],
                    "mixing_mode": s["mixing_mode"],
                    "si_sinr_db": s["si_sinr_db"],
                    "snr_db": snr_map[s["global_idx"]],
                }
                for s in samples
            ]
        dl_keys = [k for k in raw if "samples" in raw[k]]
        print(f"  Loaded DL models: {dl_keys}")

    return all_samples


def bin_by_snr(samples: list[dict]) -> dict[str, dict]:
    """Bin samples by SNR and compute mean SI-SINR per bin and mixing mode."""
    result = {}
    for (lo, hi), label in zip(SNR_BINS, SNR_LABELS):
        binned = [s for s in samples if lo <= s["snr_db"] < hi]
        if not binned:
            result[label] = {"n": 0, "mean_si_sinr_db": None, "std_si_sinr_db": None,
                             "co_channel": None, "adjacent_channel": None}
            continue
        vals = [s["si_sinr_db"] for s in binned]
        co_vals = [s["si_sinr_db"] for s in binned if s["mixing_mode"] == "co-channel"]
        adj_vals = [s["si_sinr_db"] for s in binned if s["mixing_mode"] == "adjacent-channel"]
        result[label] = {
            "n": len(vals),
            "mean_si_sinr_db": round(float(np.mean(vals)), 3),
            "std_si_sinr_db": round(float(np.std(vals)), 3),
            "co_channel": round(float(np.mean(co_vals)), 3) if co_vals else None,
            "n_co": len(co_vals),
            "adjacent_channel": round(float(np.mean(adj_vals)), 3) if adj_vals else None,
            "n_adj": len(adj_vals),
        }
    return result


def print_table(stratified: dict):
    """Print a readable SNR-stratified table."""
    print(f"\n{'Config':<25} {'SNR Bin':<12} {'N':>5} {'Overall':>10} {'Co-ch':>10} {'Adj-ch':>10}")
    print("-" * 76)
    for key in sorted(stratified.keys()):
        bins = stratified[key]
        for label in SNR_LABELS:
            b = bins[label]
            if b["n"] == 0:
                continue
            overall = f"{b['mean_si_sinr_db']:.2f}" if b["mean_si_sinr_db"] is not None else "---"
            co = f"{b['co_channel']:.2f}" if b["co_channel"] is not None else "---"
            adj = f"{b['adjacent_channel']:.2f}" if b["adjacent_channel"] is not None else "---"
            print(f"{key:<25} {label:<12} {b['n']:>5} {overall:>9} dB {co:>9} dB {adj:>9} dB")
        print()


def main():
    print("Loading per-sample results and fetching SNR from HDF5 metadata...")
    all_samples = load_all_samples()

    print("\nComputing SNR-stratified PI-SI-SINR...")
    stratified = {}
    for key, samples in all_samples.items():
        stratified[key] = bin_by_snr(samples)

    print_table(stratified)

    OUTPUT.write_text(json.dumps(stratified, indent=2))
    print(f"\nResults saved to {OUTPUT}")

    # Also print a compact summary table for each n_sources / co-channel
    print("\n=== Co-channel PI-SI-SINR by SNR bin (mean dB) ===")
    method_display = {
        "ica": "ICA", "nmf": "NMF", "cnn_lstm": "CNN-LSTM",
        "dprnn": "DPRNN", "conv_tasnet": "Conv-TasNet"
    }
    for n in N_SOURCES:
        print(f"\n--- {n}-source ---")
        print(f"{'Method':<14}", end="")
        for label in SNR_LABELS:
            print(f"  {label:<12}", end="")
        print()
        for method in METHODS:
            key = f"{n}src_{method}"
            if key not in stratified:
                continue
            print(f"{method_display[method]:<14}", end="")
            for label in SNR_LABELS:
                v = stratified[key][label].get("co_channel")
                print(f"  {v:>10.2f} dB" if v is not None else f"  {'---':>10}   ", end="")
            print()


if __name__ == "__main__":
    main()
