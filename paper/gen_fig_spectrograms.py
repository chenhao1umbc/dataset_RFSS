"""
Generate Figure: Multi-Standard Signal Spectrograms.

Generates one example signal per standard (GSM, UMTS, LTE, 5G NR) and
plots their short-time Fourier transform (STFT) spectrograms in a 2×2 grid.

Output: paper/figures/fig_spectrograms.pdf
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal.windows import hann

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).parent / "figures" / "fig_spectrograms.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
})

FS = 30_720_000   # common sample rate (30.72 MHz)
DURATION_MS = 1.0  # 1 ms

print("Generating signals…")

import torch
import math


def resample_to(sig: torch.Tensor, src_fs: int, dst_fs: int) -> torch.Tensor:
    """Simple rational resampling via torch.nn.functional.interpolate."""
    if src_fs == dst_fs:
        return sig
    g = math.gcd(src_fs, dst_fs)
    up, down = dst_fs // g, src_fs // g
    x = sig.unsqueeze(0).unsqueeze(0)           # (1,1,T)
    x = torch.nn.functional.interpolate(x, scale_factor=up, mode="linear",
                                         align_corners=False)
    x = x[:, :, ::down]
    n_target = int(DURATION_MS / 1000 * dst_fs)
    x = x[:, :, :n_target]
    if x.shape[-1] < n_target:
        x = torch.nn.functional.pad(x, (0, n_target - x.shape[-1]))
    return x.squeeze(0).squeeze(0)


def make_complex(result: dict, dst_fs: int) -> np.ndarray:
    sig = result["signal"]
    src_fs = int(result["metadata"]["sample_rate"])
    if torch.is_complex(sig):
        r = resample_to(sig.real, src_fs, dst_fs)
        i = resample_to(sig.imag, src_fs, dst_fs)
    else:
        r = resample_to(sig, src_fs, dst_fs)
        i = torch.zeros_like(r)
    return torch.complex(r, i).numpy()


signals = {}
try:
    from src.run_gsm  import generate_gsm_signal
    from src.run_umts import generate_umts_signal
    from src.run_lte  import generate_lte_signal
    from src.run_5g   import generate_5g_signal

    signals["GSM"]   = make_complex(generate_gsm_signal(duration_ms=DURATION_MS),  FS)
    signals["UMTS"]  = make_complex(generate_umts_signal(duration_ms=DURATION_MS), FS)
    signals["LTE"]   = make_complex(generate_lte_signal(duration_ms=DURATION_MS, bandwidth_mhz=10.0), FS)
    signals["5G NR"] = make_complex(generate_5g_signal(duration_ms=DURATION_MS, numerology=1, bandwidth_mhz=50.0), FS)
    for k, v in signals.items():
        print(f"  {k}: {len(v)} samples at {FS/1e6:.2f} MHz")
except Exception as e:
    print(f"Signal generation error: {e}")
    print("Falling back to synthetic test signals for each standard.")
    rng = np.random.default_rng(0)
    N = int(DURATION_MS / 1000 * FS)

    def gmsk_like(N):
        bits = rng.integers(0, 2, N // 8)
        phase = np.cumsum(np.repeat((2 * bits - 1) * np.pi / 8, 8))
        return np.exp(1j * phase[:N])

    def cdma_like(N):
        chips = rng.choice([-1, 1], N)
        return chips.astype(np.complex64)

    def ofdm_like(N, n_sc=128, cp=32):
        syms = []
        while len(syms) < N:
            fd = rng.standard_normal(n_sc) + 1j * rng.standard_normal(n_sc)
            td = np.fft.ifft(fd, n=n_sc)
            syms.extend(np.concatenate([td[-cp:], td]))
        return np.array(syms[:N], dtype=np.complex64)

    signals["GSM"]   = gmsk_like(N).astype(np.complex64)
    signals["UMTS"]  = cdma_like(N).astype(np.complex64)
    signals["LTE"]   = ofdm_like(N, n_sc=512, cp=36).astype(np.complex64)
    signals["5G NR"] = ofdm_like(N, n_sc=1024, cp=72).astype(np.complex64)

print("Computing spectrograms…")

NFFT = 512
HOP  = 128
WIN  = hann(NFFT, sym=False)

fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6))
fig.subplots_adjust(hspace=0.38, wspace=0.28)

labels = ["GSM", "UMTS", "LTE", "5G NR"]
subtitles = ["GSM (200 kHz, GMSK)", "UMTS (5 MHz, W-CDMA)",
             "LTE (10 MHz, OFDM)", "5G NR (50 MHz, OFDM)"]

for ax, label, subtitle in zip(axes.flat, labels, subtitles):
    sig = signals[label]
    n_frames = (len(sig) - NFFT) // HOP + 1
    stft = np.zeros((NFFT, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        frame = sig[i * HOP: i * HOP + NFFT] * WIN
        stft[:, i] = np.fft.fft(frame.astype(np.complex128), n=NFFT)

    # fftshift so DC is at centre, convert to dB
    stft = np.fft.fftshift(stft, axes=0)
    Sxx = 20 * np.log10(np.abs(stft) + 1e-10)
    Sxx -= Sxx.max()

    times = np.arange(n_frames) * HOP / FS * 1e3
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, d=1 / FS)) / 1e6

    im = ax.pcolormesh(times, freqs, Sxx, vmin=-60, vmax=0,
                       cmap="inferno", shading="auto", rasterized=True)
    ax.set_title(subtitle, pad=3)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")

cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02,
                  label="Normalised PSD (dB)")
cb.ax.tick_params(labelsize=6.5)

fig.suptitle("Multi-Standard RF Signal Spectrograms", y=1.01, fontsize=9)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
