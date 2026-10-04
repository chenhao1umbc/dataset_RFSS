"""
Front-end comparison for 2-source separation, decided on the validation split only.

Variants (same data crops, same seed, same batch size and data order, 2 epochs of the 2-source train split):
  l16     : Conv-TasNet, encoder L=16, stride 8 (the control; the recipe of train_all.sh)
  l256    : Conv-TasNet, encoder L=256, stride 64, rest unchanged
  stft    : STFT front end (n_fft 2048, hop 512), BLSTM on log-magnitude features, learned complex ratio
            mask on the real/imag STFT, inverse STFT back to the waveform

After each epoch the validation crops are scored: mean PI SI-SINR and the gain over the input mixture
(mean over sources of SI-SINR(mixture, source)), overall and in the bins adjacent-channel / co-channel
with SNR above 20 dB, with 95 percent bootstrap intervals of the gain. Gradient norm is clipped at 1.0 as in train.py.

Usage:
    uv run python check/encoder_sweep.py --variants l16 l256 stft

Output: check/encoder_sweep_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import ConvTasNet, pit_si_sinr_loss, si_sinr  # noqa: E402
from src.train import SeparationDataset  # noqa: E402

DATASET_PATH = ROOT / "data" / "rfss_dataset.h5"
OUTPUT = ROOT / "check" / "encoder_sweep_results.json"
TRAIN_LENGTH = 7680
SEED = 0
BINS = {"all": lambda mode, snr: True,
        "adjacent_snr_gt_20": lambda mode, snr: mode == "adjacent-channel" and snr > 20,
        "co_snr_gt_20": lambda mode, snr: mode == "co-channel" and snr > 20}


class STFTMaskNet(nn.Module):
    """STFT -> BLSTM on log-magnitude -> complex ratio masks (tanh on real and imaginary parts) -> inverse STFT."""

    def __init__(self, n_sources: int = 2, n_fft: int = 2048, hop: int = 512, hidden: int = 256):
        super().__init__()
        self.n_sources, self.n_fft, self.hop = n_sources, n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))
        self.inp = nn.Linear(n_fft, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, n_sources * n_fft * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t = x.shape
        z = torch.complex(x[:, 0], x[:, 1])
        spec = torch.stft(z, self.n_fft, self.hop, window=self.window, return_complex=True, onesided=False)  # (B, F, T')
        feat = torch.log1p(spec.abs()).transpose(1, 2)                                                       # (B, T', F)
        h, _ = self.rnn(torch.relu(self.inp(feat)))
        m = torch.tanh(self.out(h)).view(b, -1, self.n_sources, self.n_fft, 2)                               # (B, T', S, F, 2)
        mask = torch.complex(m[..., 0], m[..., 1]).permute(0, 2, 3, 1)                                       # (B, S, F, T')
        est = (mask * spec.unsqueeze(1)).reshape(b * self.n_sources, self.n_fft, -1)
        wav = torch.istft(est, self.n_fft, self.hop, window=self.window, length=t, onesided=False, return_complex=True)
        wav = wav.view(b, self.n_sources, t)
        return torch.stack([wav.real, wav.imag], dim=2)                                                      # (B, S, 2, T)


def build(variant: str, n_sources: int) -> nn.Module:
    if variant == "l16":
        return ConvTasNet(N=256, L=16, B=128, H=256, P=3, X=8, R=3, n_sources=n_sources)
    if variant == "l256":
        return ConvTasNet(N=256, L=256, B=128, H=256, P=3, X=8, R=3, n_sources=n_sources, stride=64)
    return STFTMaskNet(n_sources=n_sources)


def load_items(split: str, n: int, n_sources: int):
    """Items of a split with fixed crops; n = 0 means all. Returns tensors and (mode, snr) per item."""
    ds = SeparationDataset(DATASET_PATH, split=split, n_sources=n_sources, train_length=TRAIN_LENGTH)
    rng = np.random.RandomState(SEED)
    chosen = np.arange(len(ds)) if n == 0 else np.sort(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    np.random.seed(SEED)
    items = [ds[int(i)] for i in chosen]
    with h5py.File(DATASET_PATH, "r") as f:
        meta = [json.loads(f["metadata"][ds.indices[int(i)]]) for i in chosen]
    info = [(m["mixing_params"]["mixing_mode"], m["snr_db"]) for m in meta]
    return torch.stack([it["mixed"] for it in items]), torch.stack([it["sources"] for it in items]), info


@torch.no_grad()
def validate(model, mixed, sources, info, device) -> dict:
    """Per-sample PI SI-SINR and gain over the input, summarised overall and per bin."""
    model.eval()
    score, gain = [], []
    for i in range(len(mixed)):
        m, t = mixed[i:i + 1].to(device), sources[i:i + 1].to(device)
        s = -float(pit_si_sinr_loss(model(m), t))
        flat_mix = m.reshape(1, -1).expand(t.shape[1], -1)
        inp = float(si_sinr(flat_mix, t[0].reshape(t.shape[1], -1)).mean())
        score.append(s)
        gain.append(s - inp)
    model.train()
    rng = np.random.RandomState(SEED)
    out = {}
    for name, keep in BINS.items():
        idx = [i for i, (mode, snr) in enumerate(info) if keep(mode, snr)]
        g = np.array([gain[i] for i in idx])
        boot = rng.choice(g, size=(2000, len(g))).mean(axis=1)
        out[name] = {"n": len(idx), "val_si_sinr_db": float(np.mean([score[i] for i in idx])),
                     "gain_over_input_db": float(g.mean()),
                     "gain_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    out["per_sample_gain_db"] = [round(x, 3) for x in gain]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["l16", "l256", "stft"], choices=["l16", "l256", "stft"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=0, help="training crops (0 = the whole 2-source train split)")
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--n-sources", type=int, default=2)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    t0 = time.time()
    train_x, train_y, _ = load_items("train", args.n_train, args.n_sources)
    val_x, val_y, val_info = load_items("val", args.n_val, args.n_sources)
    steps_per_epoch = len(train_x) // args.batch_size
    print(f"loaded {len(train_x)} train and {len(val_x)} val crops in {time.time() - t0:.0f}s; {steps_per_epoch} steps per epoch", flush=True)

    for variant in args.variants:
        torch.manual_seed(SEED)
        device = "cpu" if variant == "stft" else args.device  # MPS lacks the istft backward (unfold_backward)
        model = build(variant, args.n_sources).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(SEED)
        record = {"variant": variant, "params": n_params, "batch_size": args.batch_size, "seed": SEED,
                  "n_train_crops": len(train_x), "n_val_crops": len(val_x), "epochs": []}
        t1 = time.time()
        for epoch in range(1, args.epochs + 1):
            losses = []
            for b in torch.randperm(len(train_x), generator=gen)[: steps_per_epoch * args.batch_size].split(args.batch_size):
                loss = pit_si_sinr_loss(model(train_x[b].to(device)), train_y[b].to(device))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # as in train.py
                opt.step()
                losses.append(float(loss.detach()))
            val = validate(model, val_x, val_y, val_info, device)
            record["epochs"].append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
            print(f"{variant} epoch {epoch}: train loss {np.mean(losses):.3f}; val "
                  + "; ".join(f"{k}: {v['val_si_sinr_db']:.2f} dB (gain {v['gain_over_input_db']:+.2f} [{v['gain_ci95'][0]:+.2f},{v['gain_ci95'][1]:+.2f}], n={v['n']})"
                              for k, v in val.items() if k != "per_sample_gain_db")
                  + f" [{time.time() - t1:.0f}s]", flush=True)
            results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}  # variants may run in parallel processes
            results[variant] = record
            OUTPUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
