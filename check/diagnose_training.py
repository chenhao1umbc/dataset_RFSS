"""
Training diagnostics for the plateau seen in the Conv-TasNet 2-source run.

  overfit : train a fresh Conv-TasNet on a handful of fixed training crops. A working data path and
            model drive the training SI-SINR far above 0 dB; a value near the plateau means a defect.
  tensors : sum of the target sources against the mixture on the exact tensors that reach the loss
            (after cropping, padding, RMS normalisation and the real/imag stacking); the residual
            relative to the source sum should sit at minus the sample's SNR.
  linear  : gain of the best fixed linear filter. One complex FIR filter per source slot is fitted by least
            squares on training crops (mixture -> that slot's reference), then scored with the same
            permutation-invariant SI-SINR on the 800 validation crops of check/encoder_sweep.py. It shows how
            much of a plateau level can be reached without any separation of the content.
  ceiling : best SI-SINR any output of the CNN-LSTM decoder can reach. Its output is a 1x1 convolution on features at 1/8 of
            the sample rate, linearly interpolated back, so only piecewise-linear signals with 8-sample knots are
            possible; the reference of each source is least-squares fitted inside that space and scored.
  crops   : fraction of random training crops in which a source's aligned reference carries under
            1 percent of its full-signal power (a nearly empty target makes the loss meaningless).

Usage:
    uv run python check/diagnose_training.py overfit --n 32 --steps 400
    uv run python check/diagnose_training.py ceiling --n 64
    uv run python check/diagnose_training.py crops --n 500
    uv run python check/diagnose_training.py tensors --n 64
    uv run python check/diagnose_training.py linear --n 1500 --taps 64

Output: check/diagnose_training_results.json (keys overfit_n<n>_<device>, crops, tensors, linear_fir<taps>, cnn_lstm_ceiling)
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import pit_si_sinr_loss, si_sinr  # noqa: E402
from src.train import SeparationDataset, build_model  # noqa: E402
from src.utils_mixing import build_aligned_references  # noqa: E402

DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "diagnose_training_results.json"
TRAIN_LENGTH = 7680
SEED = 0
CNN_LSTM_DOWNSAMPLE = 8  # three stride-2 convolutions in CNNLSTMSeparator


def update_results(key: str, value: dict):
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    results[key] = value
    OUTPUT.write_text(json.dumps(results, indent=1))


def overfit(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ds = SeparationDataset(DATASET_PATH, split="train", n_sources=args.n_sources, train_length=TRAIN_LENGTH)
    items = [ds[i] for i in range(args.n)]  # crops drawn once, then fixed
    mixed = torch.stack([it["mixed"] for it in items]).to(args.device)
    sources = torch.stack([it["sources"] for it in items]).to(args.device)

    model = build_model(args.model, args.n_sources).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(1, args.steps + 1):
        perm = torch.randperm(args.n)
        for start in range(0, args.n, args.batch_size):
            b = perm[start:start + args.batch_size]
            loss = pit_si_sinr_loss(model(mixed[b]), sources[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step == 1 or step % 25 == 0:
            with torch.no_grad():
                train_sisinr = -float(pit_si_sinr_loss(model(mixed), sources))
            history.append({"step": step, "train_si_sinr_db": round(train_sisinr, 3)})
            print(f"step {step}: train SI-SINR {train_sisinr:.2f} dB", flush=True)
    update_results(f"overfit_n{args.n}_{args.device}", {"model": args.model, "n_sources": args.n_sources, "n_samples": args.n,
                               "steps": args.steps, "lr": 1e-3, "history": history})


def tensors(args):
    ds = SeparationDataset(DATASET_PATH, split="train", n_sources=args.n_sources, train_length=TRAIN_LENGTH)
    np.random.seed(SEED)
    rel_db, gap_db = [], []
    with h5py.File(DATASET_PATH, "r") as f:
        for i in range(args.n):
            item = ds[i]
            snr = json.loads(f["metadata"][ds.indices[i]])["snr_db"]
            total = item["sources"].sum(dim=0)
            ratio = 10 * np.log10(float((item["mixed"] - total).pow(2).mean()) / float(total.pow(2).mean()))
            rel_db.append(ratio)
            gap_db.append(ratio + snr)
    result = {
        "n_samples": int(args.n),
        "residual_to_source_sum_db": {"median": float(np.median(rel_db)), "max": float(np.max(rel_db))},
        "gap_to_minus_snr_db": {"median_abs": float(np.median(np.abs(gap_db))), "max_abs": float(np.max(np.abs(gap_db)))},
    }
    print(json.dumps(result, indent=1))
    update_results("tensors", result)


def linear(args):
    from numpy.lib.stride_tricks import sliding_window_view
    from encoder_sweep import load_items, validate

    train_x, train_y, _ = load_items("train", args.n, args.n_sources)
    val_x, val_y, val_info = load_items("val", 800, args.n_sources)
    taps = args.taps
    gram = np.zeros((taps, taps), dtype=np.complex128)
    cross = np.zeros((args.n_sources, taps), dtype=np.complex128)
    for i in range(len(train_x)):
        x = torch.complex(train_x[i, 0].double(), train_x[i, 1].double()).numpy()
        design = sliding_window_view(np.pad(x, (taps - 1, 0)), taps)[:, ::-1]  # row t: x[t], x[t-1], ..., x[t-taps+1]
        gram += design.conj().T @ design
        for k in range(args.n_sources):
            target = torch.complex(train_y[i, k, 0].double(), train_y[i, k, 1].double()).numpy()
            cross[k] += design.conj().T @ target
    filters = np.linalg.solve(gram + 1e-6 * np.trace(gram).real / taps * np.eye(taps), cross.T).T  # (S, taps)

    def estimator(m, t):
        x = torch.complex(m[0, 0].double(), m[0, 1].double()).numpy()
        design = sliding_window_view(np.pad(x, (taps - 1, 0)), taps)[:, ::-1]
        est = np.stack([design @ filters[k] for k in range(args.n_sources)])
        return torch.from_numpy(np.stack([est.real, est.imag], axis=1)).float().unsqueeze(0)

    val = validate(estimator, val_x, val_y, val_info, "cpu")
    summary = {k: v for k, v in val.items() if k != "per_sample_gain_db"}
    print(json.dumps(summary, indent=1))
    update_results(f"linear_fir{taps}", {"fit_crops": int(args.n), "taps": taps, "val": summary})


def ceiling(args):
    ds = SeparationDataset(DATASET_PATH, split="train", n_sources=args.n_sources, train_length=TRAIN_LENGTH)
    np.random.seed(SEED)
    sources = torch.stack([ds[i]["sources"] for i in range(args.n)]).double()  # (n, S, 2, T)
    length = sources.shape[-1]
    knots = length // CNN_LSTM_DOWNSAMPLE
    basis = F.interpolate(torch.eye(knots, dtype=torch.double)[None], size=length, mode="linear", align_corners=False)[0].T
    fitted = (basis @ torch.linalg.lstsq(basis, sources.reshape(-1, length).T).solution).T.reshape(sources.shape)
    flat = (sources.shape[0] * sources.shape[1], 2 * length)
    ceiling_db = float(si_sinr(fitted.reshape(flat), sources.reshape(flat)).mean())
    result = {"n_samples": int(args.n), "downsample": CNN_LSTM_DOWNSAMPLE, "mean_ceiling_si_sinr_db": ceiling_db}
    print(json.dumps(result, indent=1))
    update_results("cnn_lstm_ceiling", result)


def crops(args):
    rng = np.random.RandomState(SEED)
    with h5py.File(DATASET_PATH, "r") as f:
        ds_indices = SeparationDataset(DATASET_PATH, split="train", n_sources=args.n_sources,
                                       train_length=TRAIN_LENGTH).indices
        chosen = rng.choice(ds_indices, size=args.n, replace=False)
        low_any, per_source = 0, []
        for idx in chosen:
            meta = json.loads(f["metadata"][idx])
            length = int(f["signal_lengths"][idx])
            refs = build_aligned_references(f["source_signals"][idx, :args.n_sources], meta, length)
            start = int(rng.randint(0, length - TRAIN_LENGTH + 1)) if length >= TRAIN_LENGTH else 0
            full_power = np.mean(np.abs(refs) ** 2, axis=1)
            crop_power = np.mean(np.abs(refs[:, start:start + TRAIN_LENGTH]) ** 2, axis=1)
            low = crop_power < 0.01 * full_power
            low_any += int(low.any())
            per_source.append(low)
    per_source = np.array(per_source)
    result = {
        "n_samples": int(args.n),
        "n_sources": args.n_sources,
        "frac_crops_with_any_source_below_1pct": low_any / args.n,
        "frac_below_1pct_per_source_slot": per_source.mean(axis=0).tolist(),
    }
    print(json.dumps(result, indent=1))
    update_results("crops", result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["overfit", "crops", "tensors", "linear", "ceiling"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--taps", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-sources", type=int, default=2)
    ap.add_argument("--model", default="conv_tasnet", choices=["conv_tasnet", "dprnn", "cnn_lstm"])
    ap.add_argument("--device", default="cpu", help="cpu by default so a running training job is not slowed")
    args = ap.parse_args()
    {"overfit": overfit, "crops": crops, "tensors": tensors, "linear": linear, "ceiling": ceiling}[args.task](args)


if __name__ == "__main__":
    main()
