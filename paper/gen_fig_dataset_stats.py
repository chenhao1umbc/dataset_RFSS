"""
Generate Figure: Dataset Statistics.

Scans HDF5 metadata for a random subset and plots:
  (a) Source count distribution (bar)
  (b) Mixing mode distribution (bar)
  (c) Signal standard co-occurrence heatmap (2-source pairs)

Reads metadata only — does not load signal data.

Output: paper/figures/fig_dataset_stats.pdf
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
HDF5 = ROOT / "data" / "rfss_dataset.h5"
OUT = Path(__file__).parent / "figures" / "fig_dataset_stats.pdf"

SCAN_N = 5000   # scan this many samples for statistics (fast — metadata only)
STANDARDS = ["GSM", "UMTS", "LTE", "5G NR"]
STD_NORM = {"GSM": "GSM", "UMTS": "UMTS", "LTE": "LTE", "5G_NR": "5G NR"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

print(f"Scanning {SCAN_N} samples from HDF5 metadata…")

import h5py
import random

n_sources_count = Counter()
mixing_mode_count = Counter()
pair_count = defaultdict(int)   # for 2-source co-occurrence

random.seed(42)
indices = sorted(random.sample(range(100000), SCAN_N))

with h5py.File(HDF5, "r") as f:
    meta_ds = f["metadata"]
    for idx in indices:
        raw = meta_ds[idx]
        if isinstance(raw, bytes):
            raw = raw.decode()
        meta = json.loads(raw)

        n = meta["num_sources"]
        n_sources_count[n] += 1

        mode = meta["mixing_params"]["mixing_mode"]
        mixing_mode_count[mode] += 1

        if n == 2:
            stds = sorted(
                STD_NORM.get(s["standard"], s["standard"])
                for s in meta["sources"]
            )
            pair_count[tuple(stds)] += 1

print("Scan complete.")

# ── figure ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.9))
fig.subplots_adjust(wspace=0.42)

BLUE  = "#0072B2"
GREEN = "#009E73"
ORNG  = "#E69F00"

# ── (a) Source count distribution ─────────────────────────────────────────────
ax = axes[0]
src_labels = [f"{k}-source" for k in sorted(n_sources_count)]
src_vals   = [n_sources_count[k] / SCAN_N * 100 for k in sorted(n_sources_count)]
bars = ax.bar(src_labels, src_vals, color=[BLUE, GREEN, ORNG],
              edgecolor="white", linewidth=0.3, zorder=3)
ax.set_ylabel("Proportion (%)")
ax.set_title("(a) Source Count", pad=4)
ax.grid(axis="y", linewidth=0.4, linestyle=":", color="gray", zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, 65)
for bar, val in zip(bars, src_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── (b) Mixing mode distribution ──────────────────────────────────────────────
ax = axes[1]
mode_labels = ["Co-channel", "Adjacent-\nchannel"]
mode_keys   = ["co-channel", "adjacent-channel"]
mode_vals   = [mixing_mode_count.get(k, 0) / SCAN_N * 100 for k in mode_keys]
bars = ax.bar(mode_labels, mode_vals, color=[BLUE, ORNG],
              edgecolor="white", linewidth=0.3, zorder=3)
ax.set_ylabel("Proportion (%)")
ax.set_title("(b) Mixing Mode", pad=4)
ax.grid(axis="y", linewidth=0.4, linestyle=":", color="gray", zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, 80)
for bar, val in zip(bars, mode_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── (c) 2-source standard co-occurrence heatmap ───────────────────────────────
ax = axes[2]
matrix = np.zeros((4, 4))
for i, s1 in enumerate(STANDARDS):
    for j, s2 in enumerate(STANDARDS):
        key = tuple(sorted([s1, s2]))
        matrix[i, j] = pair_count.get(key, 0)

# normalise to percentage of 2-source samples
total_2src = n_sources_count.get(2, 1)
matrix = matrix / total_2src * 100

im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(STANDARDS, rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(STANDARDS, fontsize=9)
ax.set_title("(c) 2-Source Pairs (%)", pad=4)
for i in range(4):
    for j in range(4):
        val = matrix[i, j]
        if val > 0.5:
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8.5,
                    color="white" if val > matrix.max() * 0.6 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
