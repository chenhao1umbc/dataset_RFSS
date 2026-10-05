"""
Spectral overlap and Nyquist check of the adjacent-channel mixtures, from the stored metadata only.

Each source of an adjacent-channel sample is shifted by frequency_offsets_hz (fixed 2 MHz spacing) from its baseband position, so its
band is centred at the offset. Two sources overlap when |offset_i - offset_j| < (B_i + B_j) / 2. Two bandwidth definitions are used:
nominal (the bandwidth_mhz field: channel bandwidth of the standard, 5 MHz for UMTS, 0.2 MHz for GSM) and occupied (UMTS 3.84 MHz chip rate,
LTE and NR 90 percent of the channel bandwidth, GSM 0.2 MHz). The mixture is sampled at the maximum source sample rate; a shifted band that
reaches beyond half of that rate wraps around (the shift is a complex exponential without filtering), which is counted per sample.

Usage:
    uv run python check/adjacent_overlap.py

Prints the shares for all 100,000 samples and for the two-source test subset (indices 85,000 to 99,999).
"""

import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
TEST_START = 85000


def occupied_mhz(source: dict) -> float:
    params = source["signal_params"]
    if source["standard"] == "UMTS":
        return 3.84
    if source["standard"] in ("LTE", "5G_NR"):
        return 0.9 * params["bandwidth_mhz"]
    return params["bandwidth_mhz"]


def describe(meta: dict) -> dict:
    offsets = np.array(meta["mixing_params"]["frequency_offsets_hz"]) / 1e6
    nominal = np.array([s["signal_params"]["bandwidth_mhz"] for s in meta["sources"]])
    occupied = np.array([occupied_mhz(s) for s in meta["sources"]])
    half_rate = max(s["signal_params"]["sample_rate"] for s in meta["sources"]) / 2e6
    out = {"n": len(offsets), "standards": sorted(s["standard"] for s in meta["sources"]), "half_rate_mhz": half_rate}
    for name, width in (("nominal", nominal), ("occupied", occupied)):
        pairs = [abs(offsets[i] - offsets[j]) < (width[i] + width[j]) / 2 for i, j in combinations(range(len(offsets)), 2)]
        out[f"{name}_any_overlap"] = any(pairs)
        out[f"{name}_pair_share"] = float(np.mean(pairs))
        out[f"{name}_edge_beyond_nyquist"] = bool(np.max(np.abs(offsets) + width / 2) > half_rate)
    out["centre_beyond_nyquist"] = bool(np.max(np.abs(offsets)) >= half_rate)
    out["rate_below_4mhz"] = half_rate < 2.0
    return out


def report(label: str, rows: list):
    print(f"\n{label}: {len(rows)} adjacent-channel samples")
    for key in ("nominal_any_overlap", "occupied_any_overlap", "nominal_edge_beyond_nyquist", "centre_beyond_nyquist", "rate_below_4mhz"):
        print(f"  {key:30s} {np.mean([r[key] for r in rows]):6.1%}")
    print(f"  {'nominal pairs overlapping':30s} {np.mean([r['nominal_pair_share'] for r in rows]):6.1%}")
    print(f"  {'occupied pairs overlapping':30s} {np.mean([r['occupied_pair_share'] for r in rows]):6.1%}")


def main():
    rows = []
    with h5py.File(DATASET_PATH, "r") as f:
        for idx in range(len(f["metadata"])):
            meta = json.loads(f["metadata"][idx])
            if meta["mixing_params"]["mixing_mode"] == "adjacent-channel":
                rows.append({"idx": idx, **describe(meta)})
    report("all samples", rows)
    for n in (2, 3, 4):
        report(f"{n}-source, all splits", [r for r in rows if r["n"] == n])
    report("2-source test subset", [r for r in rows if r["n"] == 2 and r["idx"] >= TEST_START])
    gsm_only = [r for r in rows if set(r["standards"]) == {"GSM"}]
    report("GSM-only mixtures, all splits", gsm_only)
    below = [r for r in rows if r["rate_below_4mhz"]]
    print(f"\nadjacent samples with mixture rate below 4 MHz: {len(below)}; compositions: "
          f"{ {tuple(r['standards']) for r in below} }; wrapped (centre beyond Nyquist): {sum(r['centre_beyond_nyquist'] for r in below)}")


if __name__ == "__main__":
    main()
