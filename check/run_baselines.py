"""Baseline experiment runner for Phase 3/5 of the RFSS project.

Evaluates ICA and NMF source separation baselines on the RFSS dataset:
every test-split sample for each source count (2, 3, 4 sources), or
--n-per-group indices sampled uniformly at random (seed 42) per source count.
References come from build_aligned_references (exact terms of the mixture).

Results are saved to check/baseline_results.json.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baseline_algorithms import (
    ICASourceSeparation,
    NMFSourceSeparation,
    permutation_invariant_si_sinr,
)
from src.utils_mixing import build_aligned_references

DATASET_PATH = Path("data/rfss_dataset.h5")
RESULTS_PATH = Path("check/baseline_results.json")

# Test split: 85% to 100% of 100k = samples 85000–99999
TEST_START = 85000
TEST_END = 100000

# Fixed seed for reproducible random sampling
SAMPLE_SEED = 42


def load_sample(h5file: h5py.File, idx: int) -> dict:
    """Load a single sample from the HDF5 file."""
    signal_len = int(h5file["signal_lengths"][idx])
    mixed = h5file["mixed_signals"][idx, :signal_len].astype(np.complex128)
    metadata = json.loads(h5file["metadata"][idx])
    num_sources = metadata["num_sources"]

    references = list(
        build_aligned_references(h5file["source_signals"][idx, :num_sources], metadata, signal_len)
    )

    return {
        "mixed": mixed,
        "references": references,
        "metadata": metadata,
        "signal_len": signal_len,
        "num_sources": num_sources,
    }


def collect_test_indices(h5file: h5py.File, n_per_group: int) -> dict:
    """Collect all test indices by source count, then sample n_per_group uniformly at random."""
    rng = np.random.RandomState(SAMPLE_SEED)

    # First pass: collect all valid indices per source count
    all_indices = {2: [], 3: [], 4: []}
    for idx in range(TEST_START, TEST_END):
        meta = json.loads(h5file["metadata"][idx])
        ns = meta["num_sources"]
        if ns in all_indices:
            all_indices[ns].append(idx)

    # Random sample n_per_group from each group
    sampled = {}
    for ns, idxs in all_indices.items():
        n = min(n_per_group, len(idxs))
        chosen = rng.choice(idxs, size=n, replace=False)
        sampled[ns] = sorted(chosen.tolist())

    return sampled


def evaluate_algorithm(
    h5file: h5py.File,
    indices: list,
    algorithm: str,
    n_sources: int,
) -> list:
    """Run separation algorithm on all samples, return list of per-sample results."""
    results = []

    for idx in indices:
        sample = load_sample(h5file, idx)
        mixed = sample["mixed"]
        refs = sample["references"]
        actual_n = sample["num_sources"]

        if algorithm == "ica":
            sep = ICASourceSeparation(n_components=actual_n)
            estimates = sep.separate(mixed)
        else:
            sep = NMFSourceSeparation(n_components=actual_n)
            estimates = sep.separate(mixed)

        si_sinr, perm = permutation_invariant_si_sinr(estimates, refs)
        results.append(
            {
                "dataset_idx": int(idx),
                "num_sources": actual_n,
                "mixing_mode": sample["metadata"]["mixing_params"]["mixing_mode"],
                "signal_len": sample["signal_len"],
                "si_sinr_db": round(si_sinr, 4),
                "permutation": perm,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-group", type=int, default=0, help="samples per source count (0 = all test samples)")
    n_per_group = parser.parse_args().n_per_group or TEST_END - TEST_START

    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH}")
        sys.exit(1)

    print(f"Loading dataset: {DATASET_PATH}")
    h5file = h5py.File(DATASET_PATH, "r")

    print(f"Collecting {n_per_group} test samples per source count (random seed {SAMPLE_SEED})...")
    test_indices = collect_test_indices(h5file, n_per_group)
    for ns, idxs in sorted(test_indices.items()):
        print(f"  {ns}-source: {len(idxs)} samples")

    all_results = {}

    for n_sources in [2, 3, 4]:
        idxs = test_indices[n_sources]
        if not idxs:
            print(f"  No {n_sources}-source samples found in test split.")
            continue

        for alg_name in ["ica", "nmf"]:
            key = f"{n_sources}src_{alg_name}"
            print(f"\n[{key}] Running {alg_name.upper()} on {len(idxs)} samples...")
            results = evaluate_algorithm(h5file, idxs, alg_name, n_sources)
            si_sinrs = [r["si_sinr_db"] for r in results]
            print(
                f"  mean={np.mean(si_sinrs):.2f} dB  "
                f"std={np.std(si_sinrs):.2f} dB  "
                f"min={np.min(si_sinrs):.2f}  max={np.max(si_sinrs):.2f}"
            )
            all_results[key] = {
                "algorithm": alg_name.upper(),
                "n_sources": n_sources,
                "n_samples": len(results),
                "mean_si_sinr_db": round(float(np.mean(si_sinrs)), 4),
                "std_si_sinr_db": round(float(np.std(si_sinrs)), 4),
                "ci95_si_sinr_db": round(float(1.96 * np.std(si_sinrs, ddof=1) / np.sqrt(len(si_sinrs))), 4),
                "min_si_sinr_db": round(float(np.min(si_sinrs)), 4),
                "max_si_sinr_db": round(float(np.max(si_sinrs)), 4),
                "samples": results,
            }

    h5file.close()

    RESULTS_PATH.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n=== Summary ===")
    print(f"{'Config':<20} {'Mean SI-SINR':>14} {'Std':>8}")
    print("-" * 44)
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{key:<20} {r['mean_si_sinr_db']:>12.2f} dB  {r['std_si_sinr_db']:>6.2f}")


if __name__ == "__main__":
    main()
