"""
Generate Figure: Benchmark Results.

Two panels:
  Left:  Overall PI-SI-SINR for all 5 methods × 3 source counts (grouped bar).
  Right: Co-channel PI-SI-SINR only (the meaningful separation metric).

Output: paper/figures/fig_benchmark.pdf
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "check" / "breakdown_results.json"
OUT = Path(__file__).parent / "figures" / "fig_benchmark.pdf"

# ── publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

# ── load results ───────────────────────────────────────────────────────────────
with open(RESULTS) as f:
    data = json.load(f)

METHODS = ["ICA", "NMF", "CNN-LSTM", "DPRNN", "Conv-TasNet"]
KEY_MAP = {
    "ICA": "ica", "NMF": "nmf",
    "CNN-LSTM": "cnn_lstm", "DPRNN": "dprnn", "Conv-TasNet": "conv_tasnet",
}
SRCS = [2, 3, 4]

def get_mean(n_src, method, channel="overall"):
    k = f"{n_src}src_{KEY_MAP[method]}"
    if k not in data:
        return None
    entry = data[k]
    if channel == "overall":
        return entry["overall"]["mean_si_sinr_db"]
    return entry[channel]["mean_si_sinr_db"]

def get_std(n_src, method):
    k = f"{n_src}src_{KEY_MAP[method]}"
    if k not in data:
        return 0.0
    return data[k]["overall"]["std_si_sinr_db"]

# ── colors: colorblind-friendly (Wong palette) ─────────────────────────────────
COLORS = {
    "ICA":         "#D55E00",   # vermillion
    "NMF":         "#E69F00",   # orange
    "CNN-LSTM":    "#009E73",   # bluish green
    "DPRNN":       "#0072B2",   # blue
    "Conv-TasNet": "#CC79A7",   # reddish purple
}
HATCHES = {
    "ICA": "//", "NMF": "\\\\", "CNN-LSTM": "", "DPRNN": "..", "Conv-TasNet": "",
}

# ── layout ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2), sharey=False)
fig.subplots_adjust(wspace=0.35)

n_methods = len(METHODS)
n_groups = len(SRCS)
bar_w = 0.14
group_w = n_methods * bar_w
offsets = np.arange(n_methods) * bar_w - (group_w / 2) + bar_w / 2

for ax_idx, (ax, channel, title) in enumerate(zip(
    axes,
    ["overall", "co_channel"],
    ["(a) Overall PI-SI-SINR", "(b) Co-channel PI-SI-SINR"],
)):
    x_centers = np.arange(n_groups)

    for mi, method in enumerate(METHODS):
        vals = [get_mean(s, method, channel) for s in SRCS]
        xs = x_centers + offsets[mi]
        bars = ax.bar(
            xs, vals,
            width=bar_w,
            color=COLORS[method],
            hatch=HATCHES[method],
            edgecolor="white",
            linewidth=0.3,
            label=method,
            zorder=3,
        )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(["2-source", "3-source", "4-source"])
    ax.set_ylabel("PI-SI-SINR (dB)")
    ax.set_title(title, pad=4)
    ax.axhline(0, color="black", linewidth=0.4, linestyle="--", zorder=2)
    ax.grid(axis="y", linewidth=0.4, linestyle=":", color="gray", zorder=0)
    ax.set_axisbelow(True)

    # y-axis range
    if channel == "overall":
        ax.set_ylim(-45, -10)
        ax.set_yticks(range(-45, -9, 5))
    else:
        ax.set_ylim(-35, -5)
        ax.set_yticks(range(-35, -4, 5))

    # value labels on DL bars only (top 3 methods)
    for mi, method in enumerate(METHODS[2:], start=2):
        for si, s in enumerate(SRCS):
            v = get_mean(s, method, channel)
            if v is not None:
                ax.text(
                    x_centers[si] + offsets[mi], v + 0.4,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, rotation=90, color="black",
                )

# legend on right axis
handles = [
    mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m],
                   edgecolor="gray", linewidth=0.3, label=m)
    for m in METHODS
]
axes[1].legend(handles=handles, loc="lower right", framealpha=0.9,
               ncol=1, handlelength=1.4, borderpad=0.4)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
