"""
Reference-alignment check (action plan A1/A2).

For test-split multi-source samples, rebuild the noiseless mixture from the stored
`source_signals` with the repo's own SignalMixer (resample -> pad -> frequency shift
-> power normalise -> sum) and compare it with the stored `mixed_signals`.
The residual should be AWGN at the stored snr_db. Then score, per source:
  - ref_current : reference used so far (nominal 1 ms native length, scipy Fourier resample, no shift)
  - ref_linear  : mixer's own resample from the true (non-zero) length, no shift
  - ref_aligned : exactly the term the mixer added to the mixture (shifted, scaled)
with SI-SINR of the mixture against each reference, and re-score ICA with
ref_current vs ref_aligned.

Outputs: check/reference_alignment_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.baseline_algorithms import (  # noqa: E402
    ICASourceSeparation,
    compute_si_sinr,
    permutation_invariant_si_sinr,
    resample_to_length,
)
from src.utils_mixing import SignalMixer, build_aligned_references  # noqa: E402

DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "reference_alignment_results.json"

TEST_START, TEST_END = 85000, 100000
SEED = 42


def pick_indices(f: h5py.File, n_per_cell: int) -> list[tuple[int, str, int]]:
    """Sample n_per_cell test indices per (mixing_mode, num_sources in 2..4)."""
    cells: dict[tuple[str, int], list[int]] = {}
    for idx in range(TEST_START, TEST_END):
        m = json.loads(f["metadata"][idx])
        ns = m["num_sources"]
        if ns in (2, 3, 4):
            cells.setdefault((m["mixing_params"]["mixing_mode"], ns), []).append(idx)
    rng = np.random.RandomState(SEED)
    out = []
    for (mode, ns), idxs in sorted(cells.items()):
        chosen = rng.choice(idxs, size=min(n_per_cell, len(idxs)), replace=False)
        out += [(int(i), mode, ns) for i in sorted(chosen)]
    return out


def rebuild(f: h5py.File, idx: int) -> dict:
    """Run the repo's SignalMixer on the stored sources; compare with the stored mixture."""
    meta = json.loads(f["metadata"][idx])
    mp = meta["mixing_params"]
    sig_len = int(f["signal_lengths"][idx])
    mixed = f["mixed_signals"][idx, :sig_len].astype(np.complex128)

    rates = [s["signal_params"]["sample_rate"] for s in meta["sources"]]
    mix_rate = max(rates)
    mixer = SignalMixer(sample_rate=mix_rate)
    flags = {"extent_differs_from_nominal": False}
    true_lens = []
    for i, (src, rate) in enumerate(zip(meta["sources"], rates)):
        nominal_len = int(round(rate * 0.001))
        raw = f["source_signals"][idx, i]
        nz = np.nonzero(raw)[0]
        native_len = int(nz[-1]) + 1
        true_lens.append(native_len)
        if native_len != nominal_len:
            flags["extent_differs_from_nominal"] = True
        mixer.add_source(
            torch.from_numpy(raw[:native_len]),
            label=src["standard"],
            power_db=mp["power_ratios_db"][i],
            freq_offset_hz=mp["frequency_offsets_hz"][i],
            timing_offset_samples=0,
            source_sample_rate=rate if rate != mix_rate else None,
        )
    res = mixer.mix(mode=mp["mixing_mode"])
    recon = res["mixed_signal"].numpy().astype(np.complex128)

    length = min(len(recon), sig_len)
    resid = mixed[:length] - recon[:length]
    resid_db = 10 * np.log10(np.mean(np.abs(resid) ** 2) / np.mean(np.abs(recon[:length]) ** 2))
    return {
        "meta": meta,
        "sig_len": sig_len,
        "recon_len": len(recon),
        "mixed": mixed,
        "rates": rates,
        "mix_rate": mix_rate,
        "len_match": len(recon) == sig_len,
        "resid_to_signal_db": float(resid_db),
        "expected_db": -float(meta["snr_db"]),
        "flags": flags,
        "aligned": [a.numpy().astype(np.complex128)[:sig_len] for a in res["source_signals_aligned"]],
        "linear": [c.numpy().astype(np.complex128)[:sig_len] for c in res["source_signals_clean"]],
        "true_lens": true_lens,
        "raw_native": [
            f["source_signals"][idx, i, : int(round(r * 0.001))].astype(np.complex128)
            for i, r in enumerate(rates)
        ],
    }


def score_refs(rb: dict) -> dict:
    """SI-SINR of the stored mixture vs each reference variant, per source (mean over sources).

    ref_current uses the nominal 1 ms native length, as check/run_baselines.py does.
    """
    cur = [resample_to_length(r, rb["sig_len"]) for r in rb["raw_native"]]
    out = {}
    for name, refs in (("current", cur), ("linear", rb["linear"]), ("aligned", rb["aligned"])):
        vals = [compute_si_sinr(rb["mixed"], r) for r in refs]
        out[name] = float(np.mean(vals))
    return out, cur


def check_builder(f: h5py.File, n: int) -> dict:
    """Forward-model proof of build_aligned_references on n random multi-source test samples."""
    rng = np.random.RandomState(SEED + 1)
    idxs = []
    while len(idxs) < n:
        idx = int(rng.randint(TEST_START, TEST_END))
        if idx not in idxs and json.loads(f["metadata"][idx])["num_sources"] in (2, 3, 4):
            idxs.append(idx)
    gaps = []
    for idx in idxs:
        meta = json.loads(f["metadata"][idx])
        length = int(f["signal_lengths"][idx])
        refs = build_aligned_references(f["source_signals"][idx, : meta["num_sources"]], meta, length)
        total = refs.sum(axis=0)
        mixed = f["mixed_signals"][idx, :length].astype(np.complex128)
        resid_db = 10 * np.log10(np.mean(np.abs(mixed - total) ** 2) / np.mean(np.abs(total) ** 2))
        gaps.append(resid_db + meta["snr_db"])
    gaps = np.array(gaps)
    return {
        "n": n,
        "abs_gap_db_median": float(np.median(np.abs(gaps))),
        "abs_gap_db_p99": float(np.percentile(np.abs(gaps), 99)),
        "abs_gap_db_max": float(np.max(np.abs(gaps))),
        "frac_within_0p5_db": float(np.mean(np.abs(gaps) < 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder-n", type=int, default=1000, help="samples for the build_aligned_references proof (0 = skip)")
    ap.add_argument("--n-per-cell", type=int, default=34, help="samples per (mode, num_sources) cell")
    ap.add_argument("--ica-n", type=int, default=13, help="ICA samples per (mode, num_sources) cell (0 = skip)")
    args = ap.parse_args()

    results = []
    with h5py.File(DATASET_PATH, "r") as f:
        picks = pick_indices(f, args.n_per_cell)
        print(f"{len(picks)} samples")
        ica_budget = {(m, n): args.ica_n for m in ("co-channel", "adjacent-channel") for n in (2, 3, 4)}
        for k, (idx, mode, ns) in enumerate(picks):
            rb = rebuild(f, idx)
            sc, cur = score_refs(rb)
            row = {
                "idx": idx,
                "mode": mode,
                "num_sources": ns,
                "snr_db": rb["meta"]["snr_db"],
                "standards": [s["standard"] for s in rb["meta"]["sources"]],
                "len_match": rb["len_match"],
                "resid_to_signal_db": round(rb["resid_to_signal_db"], 3),
                "expected_resid_db": round(rb["expected_db"], 3),
                "extent_differs_from_nominal": rb["flags"]["extent_differs_from_nominal"],
                "mix_rate_vs_len": [rb["mix_rate"], rb["sig_len"] * 1000.0],
                "mixture_vs_ref_si_sinr_db": {k2: round(v, 3) for k2, v in sc.items()},
            }
            if args.ica_n and ica_budget[(mode, ns)] > 0:
                ica_budget[(mode, ns)] -= 1
                est = ICASourceSeparation(n_components=ns).separate(rb["mixed"])
                row["ica_pi_si_sinr_db"] = {
                    "current": round(permutation_invariant_si_sinr(est, cur)[0], 3),
                    "aligned": round(permutation_invariant_si_sinr(est, rb["aligned"])[0], 3),
                }
            results.append(row)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(picks)}")

    summary = {}
    for mode in ("co-channel", "adjacent-channel"):
        rows = [r for r in results if r["mode"] == mode]
        if not rows:
            continue
        gap = np.array([r["resid_to_signal_db"] - r["expected_resid_db"] for r in rows])
        s = {
            "n": len(rows),
            "len_match_frac": float(np.mean([r["len_match"] for r in rows])),
            "recon_residual_minus_expected_db": {
                "median": float(np.median(gap)),
                "p5": float(np.percentile(gap, 5)),
                "p95": float(np.percentile(gap, 95)),
            },
            "extent_differs_from_nominal_frac": float(np.mean([r["extent_differs_from_nominal"] for r in rows])),
        }
        for v in ("current", "linear", "aligned"):
            s[f"median_mixture_vs_ref_{v}_db"] = float(
                np.median([r["mixture_vs_ref_si_sinr_db"][v] for r in rows])
            )
        ica_rows = [r for r in rows if "ica_pi_si_sinr_db" in r]
        if ica_rows:
            s["ica_n"] = len(ica_rows)
            for v in ("current", "aligned"):
                s[f"ica_mean_pi_si_sinr_{v}_db"] = float(
                    np.mean([r["ica_pi_si_sinr_db"][v] for r in ica_rows])
                )
        summary[mode] = s

    if args.builder_n:
        with h5py.File(DATASET_PATH, "r") as f:
            summary["build_aligned_references_check"] = check_builder(f, args.builder_n)

    OUTPUT.write_text(json.dumps({"summary": summary, "samples": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
