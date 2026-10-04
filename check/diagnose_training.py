"""
Training diagnostics for the plateau seen in the Conv-TasNet 2-source run.

  overfit : train a fresh Conv-TasNet on a handful of fixed training crops. A working data path and
            model drive the training SI-SINR far above 0 dB; a value near the plateau means a defect.
  crops   : fraction of random training crops in which a source's aligned reference carries under
            1 percent of its full-signal power (a nearly empty target makes the loss meaningless).

Usage:
    uv run python check/diagnose_training.py overfit --n 32 --steps 400
    uv run python check/diagnose_training.py crops --n 500

Output: check/diagnose_training_results.json (keys overfit, crops)
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

from src.models import pit_si_sinr_loss  # noqa: E402
from src.train import SeparationDataset, build_model  # noqa: E402
from src.utils_mixing import build_aligned_references  # noqa: E402

DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "diagnose_training_results.json"
TRAIN_LENGTH = 7680
SEED = 0


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
    update_results("overfit", {"model": args.model, "n_sources": args.n_sources, "n_samples": args.n,
                               "steps": args.steps, "lr": 1e-3, "history": history})


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
    ap.add_argument("task", choices=["overfit", "crops"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-sources", type=int, default=2)
    ap.add_argument("--model", default="conv_tasnet", choices=["conv_tasnet", "dprnn", "cnn_lstm"])
    ap.add_argument("--device", default="cpu", help="cpu by default so a running training job is not slowed")
    args = ap.parse_args()
    overfit(args) if args.task == "overfit" else crops(args)


if __name__ == "__main__":
    main()
