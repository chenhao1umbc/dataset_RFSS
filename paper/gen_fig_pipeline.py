"""
Generate Figure: Dataset Construction Pipeline Diagram.

Schematic block diagram showing the dataset generation pipeline:
  Signal Generators → Channel Model → Hardware Impairments → Mixing → HDF5

Output: paper/figures/fig_pipeline.pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).parent / "figures" / "fig_pipeline.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.16, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

# ── colour palette (Wong) ────────────────────────────────────────────────────
C_GEN  = "#0072B2"   # blue  – signal generators
C_CH   = "#E69F00"   # orange – channel model
C_HW   = "#D55E00"   # vermillion – hardware impairments
C_MIX  = "#009E73"   # green – mixing
C_DS   = "#CC79A7"   # purple – HDF5 output
C_TXT  = "white"
ALPHA  = 0.92

# ── block geometry ──────────────────────────────────────────────────────────
# Each block: (cx, cy, width, height, color, title, subtitle_lines)
BLOCKS = [
    (1.0,  2.0, 1.6, 2.8, C_GEN, "Signal\nGenerators",
     ["GSM (GMSK)", "UMTS (W-CDMA)", "LTE (OFDM)", "5G NR (OFDM)"]),
    (3.0,  2.0, 1.6, 2.8, C_CH,  "Channel\nModel",
     ["TDL-A/B/C/D/E", "AWGN 0–30 dB", "Doppler, delay", "Rician/Rayleigh"]),
    (5.0,  2.0, 1.6, 2.8, C_HW,  "Hardware\nImpairments",
     ["CFO, phase noise", "I/Q imbalance", "DC offset", "PA nonlinearity"]),
    (7.0,  2.0, 1.6, 2.8, C_MIX, "Mixing",
     ["Co-channel", "Adjacent-channel", "2–4 sources", "Freq. offsets"]),
    (9.0,  2.0, 1.6, 2.8, C_DS,  "HDF5\nDataset",
     ["100 k samples", "4 standards", "103 GB", "Train/Val/Test"]),
]

def draw_block(ax, cx, cy, w, h, color, title, lines):
    bx = cx - w / 2
    by = cy - h / 2
    rect = mpatches.FancyBboxPatch(
        (bx, by), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor="white",
        linewidth=0.8, alpha=ALPHA, zorder=3,
    )
    ax.add_patch(rect)
    # title
    ax.text(cx, cy + h / 2 - 0.38, title,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_TXT, zorder=4, multialignment="center")
    # separator line
    ax.plot([bx + 0.1, bx + w - 0.1], [cy + h / 2 - 0.65, cy + h / 2 - 0.65],
            color="white", linewidth=0.5, alpha=0.6, zorder=4)
    # bullet lines
    for k, line in enumerate(lines):
        ax.text(cx, cy + h / 2 - 0.90 - k * 0.42, line,
                ha="center", va="center", fontsize=8.5,
                color=C_TXT, zorder=4)

for (cx, cy, w, h, color, title, lines) in BLOCKS:
    draw_block(ax, cx, cy, w, h, color, title, lines)

# ── arrows ───────────────────────────────────────────────────────────────────
arrow_kw = dict(
    arrowstyle="-|>",
    color="#333333",
    linewidth=1.2,
    mutation_scale=10,
    zorder=5,
)
xs = [b[0] for b in BLOCKS]
for i in range(len(xs) - 1):
    x_start = xs[i] + BLOCKS[i][2] / 2 + 0.04
    x_end   = xs[i + 1] - BLOCKS[i + 1][2] / 2 - 0.04
    y       = BLOCKS[i][1]
    ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                arrowprops=arrow_kw)

fig.tight_layout(pad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
