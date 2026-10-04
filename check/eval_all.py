"""
Unified evaluation of all methods on the same samples and the same segment.

Every method (input mixture, ICA, NMF, trained DL models, noise-limited oracle) is scored on the
first SEGMENT_LEN samples of each test-split sample (shorter signals are used in full), against the
exact references from build_aligned_references, with the same complex-valued permutation-invariant
SI-SINR. Reports per-sample results and, per source count and mixing mode (and per SNR bin), the mean,
median, standard deviation, 95 percent bootstrap interval and (for improvements) the fraction of
samples above zero, for the absolute score and for the improvement over the input mixture.

  input  : mean over sources of SI-SINR(mixture, reference_i)
  oracle : estimate_i = reference_i + (stored mixture - sum of references), i.e. perfect separation
           that leaves all additive noise in every output

Usage:
    uv run python check/eval_all.py                       # input, oracle, ICA, NMF
    uv run python check/eval_all.py --dl conv_tasnet dprnn cnn_lstm   # also trained models
    uv run python check/eval_all.py --dl ... --crop-seed 0            # robustness pass, random window
    uv run python check/eval_all.py --dl conv_tasnet --sources 2      # only the finished source counts

Output: check/eval_all_results.json (check/eval_all_crop<seed>_results.json with --crop-seed)
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
    NMFSourceSeparation,
    compute_si_sinr,
    permutation_invariant_si_sinr,
)
from src.train import _checkpoint_loss, build_model  # noqa: E402
from src.utils_mixing import build_aligned_references  # noqa: E402

DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "eval_all_results.json"
CHECKPOINT_ROOT = ROOT / "checkpoints"

TEST_START, TEST_END = 85000, 100000
SEGMENT_LEN = 7680
SNR_BINS = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40.001)]
N_BOOT = 2000
BOOT_SEED = 0


def load_segment(f: h5py.File, idx: int, crop_seed: int | None) -> dict:
    """Mixture and exact references for the evaluation segment of one sample.

    The segment is the first SEGMENT_LEN samples, or with crop_seed a window starting at a
    per-sample random offset (fixed by crop_seed and idx) to avoid start-of-signal transients.
    """
    meta = json.loads(f["metadata"][idx])
    n_src = meta["num_sources"]
    length = int(f["signal_lengths"][idx])
    refs = build_aligned_references(f["source_signals"][idx, :n_src], meta, length)
    mixed = f["mixed_signals"][idx, :length].astype(np.complex128)
    seg = min(length, SEGMENT_LEN)
    start = 0
    if crop_seed is not None and length > SEGMENT_LEN:
        start = int(np.random.RandomState([crop_seed, idx]).randint(0, length - SEGMENT_LEN + 1))
    return {
        "idx": idx,
        "meta": meta,
        "mixed": mixed[start:start + seg],
        "refs": refs[:, start:start + seg],
    }


def pi_score(estimates, refs) -> float:
    return float(permutation_invariant_si_sinr(list(estimates), list(refs))[0])


def reference_scores(sample: dict) -> dict:
    """Input and oracle rows."""
    mixed, refs = sample["mixed"], sample["refs"]
    noise = mixed - refs.sum(axis=0)
    return {
        "input": float(np.mean([compute_si_sinr(mixed, r) for r in refs])),
        "oracle": float(np.mean([compute_si_sinr(r + noise, r) for r in refs])),
    }


def load_dl_model(name: str, n_sources: int, device: str):
    ckpt_dir = CHECKPOINT_ROOT / f"{name}_{n_sources}src"
    best = min(ckpt_dir.glob("epoch_*.pt"), key=_checkpoint_loss)
    model = build_model(name, n_sources)
    model.load_state_dict(torch.load(best, map_location=device)["model"])
    return model.to(device).eval(), best.name


@torch.no_grad()
def dl_scores(model, samples: list[dict], device: str, batch_size: int = 16) -> list[float]:
    """PI-SI-SINR of a trained model; inputs are zero-padded to SEGMENT_LEN and RMS-normalised as in training."""
    out = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        x = np.zeros((len(batch), 2, SEGMENT_LEN), dtype=np.float32)
        for b, s in enumerate(batch):
            m = np.zeros(SEGMENT_LEN, dtype=np.complex128)
            m[: len(s["mixed"])] = s["mixed"]
            m = m / (np.sqrt(np.mean(np.abs(m) ** 2)) + 1e-8)
            x[b, 0], x[b, 1] = m.real, m.imag
        est = model(torch.from_numpy(x).to(device)).cpu().numpy()
        for b, s in enumerate(batch):
            seg = len(s["mixed"])
            est_c = est[b, :, 0, :seg] + 1j * est[b, :, 1, :seg]
            out.append(pi_score(est_c, s["refs"]))
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.RandomState) -> list[float]:
    idx = rng.randint(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stats(values: list[float], rng: np.random.RandomState) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
        "ci95": bootstrap_ci(v, rng),
        "frac_positive": float(np.mean(v > 0)),
    }


def summarise(rows: list[dict], methods: list[str]) -> dict:
    rng = np.random.RandomState(BOOT_SEED)
    groups: dict[str, list[dict]] = {}
    for r in rows:
        ns = f"{r['num_sources']}src"
        groups.setdefault(ns, []).append(r)
        groups.setdefault(f"{ns}/{r['mode']}", []).append(r)
        for lo, hi in SNR_BINS:
            if lo <= r["snr_db"] < hi:
                groups.setdefault(f"{ns}/snr_{lo:g}_{min(hi, 40):g}", []).append(r)
    out = {}
    for key, g in sorted(groups.items()):
        entry = {}
        for m in methods:
            entry[m] = stats([r[m] for r in g], rng)
            if m != "input":
                entry[m + "_improvement"] = stats([r[m] - r["input"] for r in g], rng)
        out[key] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="random test samples per source count (0 = all)")
    ap.add_argument("--dl", nargs="*", default=[], choices=["conv_tasnet", "dprnn", "cnn_lstm"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sources", type=int, nargs="*", default=[2, 3, 4], choices=[2, 3, 4], help="source counts to evaluate")
    ap.add_argument("--crop-seed", type=int, default=None, help="random window per sample instead of the first SEGMENT_LEN samples")
    args = ap.parse_args()
    output = OUTPUT if args.crop_seed is None else OUTPUT.with_name(f"eval_all_crop{args.crop_seed}_results.json")

    device = args.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    classical = {"ica": ICASourceSeparation, "nmf": NMFSourceSeparation}
    methods = ["input", "oracle", "ica", "nmf"] + args.dl
    rows = []
    checkpoints = {}
    with h5py.File(DATASET_PATH, "r") as f:
        by_n: dict[int, list[int]] = {2: [], 3: [], 4: []}
        for idx in range(TEST_START, TEST_END):
            ns = json.loads(f["metadata"][idx])["num_sources"]
            if ns in by_n:
                by_n[ns].append(idx)
        rng = np.random.RandomState(42)
        for ns, idxs in by_n.items():
            if ns not in args.sources:
                continue
            if args.n:
                idxs = sorted(rng.choice(idxs, size=min(args.n, len(idxs)), replace=False).tolist())
            print(f"{ns}-source: {len(idxs)} samples", flush=True)
            samples = [load_segment(f, i, args.crop_seed) for i in idxs]
            part = []
            for s in samples:
                row = {
                    "idx": s["idx"],
                    "num_sources": ns,
                    "mode": s["meta"]["mixing_params"]["mixing_mode"],
                    "snr_db": s["meta"]["snr_db"],
                    "length": len(s["mixed"]),
                    **reference_scores(s),
                }
                for name, cls in classical.items():
                    est = cls(n_components=ns).separate(s["mixed"])
                    row[name] = pi_score(est, s["refs"])
                part.append(row)
            for name in args.dl:
                model, ckpt_name = load_dl_model(name, ns, device)
                checkpoints[f"{name}_{ns}src"] = ckpt_name
                for row, score in zip(part, dl_scores(model, samples, device)):
                    row[name] = score
            rows += part

    result = {
        "segment_len": SEGMENT_LEN,
        "crop_seed": args.crop_seed,
        "test_range": [TEST_START, TEST_END],
        "checkpoints": checkpoints,
        "summary": summarise(rows, methods),
        "samples": rows,
    }
    # compute_si_sinr returns +-100 as sentinels for zero-power reference or residual; none are expected
    n_sentinel = sum(abs(abs(r[m]) - 100.0) < 1e-9 for r in rows for m in methods)
    result["n_sentinel_scores"] = int(n_sentinel)
    output.write_text(json.dumps(result))
    if n_sentinel:
        raise RuntimeError(f"{n_sentinel} sample scores equal the +-100 dB sentinel; inspect {output}")
    for key, entry in result["summary"].items():
        if "/" not in key:
            print(key, {m: round(entry[m]["mean"], 2) for m in methods})


if __name__ == "__main__":
    main()
