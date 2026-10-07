"""Figure 2: RFSS dataset composition.

Data source: the dataset-level counts in
    /Users/hc/.hermes/cache/scratch/rfss/dataset_meta_counts.json
which were produced by a full scan of the 100,000-entry metadata array of
data/rfss_dataset.h5.  The released file is 110 GB, so the figure reads the
cached counts rather than re-opening the HDF5; verify_counts.py performs the
live scan separately as a bounded cross-check.

Panels:
  (a) sample counts by source count and mixing mode (stacked bars, counts
      printed); the legend sits ABOVE the axes, in the figure's top margin, so
      it can never cover a bar, a count label or the panel letter.
  (b) SNR distribution over all 100,000 samples, 2 dB bins on -10 .. 40 dB.
  (c) 2-source standard pair matrix; unordered pairs, upper triangle only
      (including the diagonal, i.e. the same-standard counts), shared colour
      scale from 0 to the largest cell (14,054).
Drawn at the final print size: 7.16 in wide, text 7 to 8 pt, vector PDF.

Output:  paper/figures/fig_dataset_stats.pdf
Preview: paper/figures/preview/fig_dataset_stats.png (150 dpi, not for the paper)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT = Path(__file__).parent.parent
COUNTS_JSON = Path(
    "/Users/hc/.hermes/cache/scratch/rfss/dataset_meta_counts.json"
)
OUT_DIR = Path(__file__).parent / "figures"
OUT = OUT_DIR / "fig_dataset_stats.pdf"
PREVIEW = OUT_DIR / "preview" / "fig_dataset_stats.png"

WIDTH_IN = 7.16
HEIGHT_IN = 2.62
FS_AXIS = 8.0
FS_TICK = 7.0
FS_ANNOT = 7.0

STANDARDS = ["GSM", "UMTS", "LTE", "5G_NR"]
STD_LABEL = {"GSM": "GSM", "UMTS": "UMTS", "LTE": "LTE", "5G_NR": "5G NR"}
PAIR_JSON_KEY = {"GSM": "GSM", "UMTS": "UMTS", "LTE": "LTE", "5G_NR": "5G NR"}
MODE_COLOR = {"co-channel": "#5c5c5c", "adjacent-channel": "#b0b0b0"}
SNR_EDGES = np.arange(-10.0, 42.0, 2.0)
LUMINANCE_CUT = 0.5

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_TICK,
    "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})


def load_counts():
    with open(COUNTS_JSON, "r", encoding="utf-8") as handle:
        counts = json.load(handle)
    return counts


def pair_matrix(counts):
    matrix = np.full((4, 4), np.nan, dtype=float)
    for row in range(4):
        for col in range(row, 4):
            first = PAIR_JSON_KEY[STANDARDS[row]]
            second = PAIR_JSON_KEY[STANDARDS[col]]
            matrix[row, col] = float(counts["pair_upper"][f"{first}|{second}"])
    return matrix


def style_axes(ax):
    ax.tick_params(direction="out", length=2.0, pad=1.5)
    ax.tick_params(axis="x", pad=3.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def cell_text_colour(cmap, shade):
    red, green, blue = cmap(shade)[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < LUMINANCE_CUT else "#1a1a1a"


def geometry_check(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    boxes = []
    seen = set()
    for artist in fig.findobj(match=mtext.Text):
        if id(artist) in seen or not artist.get_text().strip():
            continue
        seen.add(id(artist))
        if not artist.get_visible():
            continue
        bb = artist.get_window_extent(renderer=renderer)
        if bb.width <= 0.0 or bb.height <= 0.0:
            continue
        boxes.append((artist.get_text(), bb.x0 / dpi, bb.y0 / dpi,
                      bb.x1 / dpi, bb.y1 / dpi))
    problems = 0
    for text, x0, y0, x1, y1 in boxes:
        if x0 < -0.002 or x1 > WIDTH_IN + 0.002 or y0 < -0.002 or y1 > HEIGHT_IN + 0.002:
            print(f"OUT OF PAGE  {text!r} x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]")
            problems += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            dx = min(a[3], b[3]) - max(a[1], b[1])
            dy = min(a[4], b[4]) - max(a[2], b[2])
            if dx > 0.0 and dy > 0.0:
                if problems < 20:
                    print(f"TEXT OVERLAP {a[0]!r} <-> {b[0]!r} area={dx * dy:.5f} in^2")
                problems += 1
    print(f"geometry check: {len(boxes)} text boxes, {problems} problems")
    return problems


def main():
    counts = load_counts()
    joint = {tuple(key.split("|")): value for key, value in counts["joint"].items()}
    joint = {(int(n), mode): value for (n, mode), value in joint.items()}
    source_counts = sorted({key[0] for key in joint})
    modes = ["co-channel", "adjacent-channel"]
    mode_label = {"co-channel": "co-channel", "adjacent-channel": "adjacent"}
    matrix = pair_matrix(counts)
    vmax = float(np.nanmax(matrix))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    norm = Normalize(vmin=0.0, vmax=vmax)
    snr_bins = np.array([entry[2] for entry in counts["snr_bins"]], dtype=float)
    snr_edges = np.array([entry[0] for entry in counts["snr_bins"]] + [counts["snr_bins"][-1][1]])
    snr_centres = 0.5 * (snr_edges[:-1] + snr_edges[1:])

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    ax_a = fig.add_axes((0.46 / WIDTH_IN, 0.60 / HEIGHT_IN, 1.60 / WIDTH_IN, 1.54 / HEIGHT_IN))
    ax_b = fig.add_axes((2.46 / WIDTH_IN, 0.60 / HEIGHT_IN, 1.92 / WIDTH_IN, 1.54 / HEIGHT_IN))
    ax_c = fig.add_axes((4.74 / WIDTH_IN, 0.80 / HEIGHT_IN, 1.70 / WIDTH_IN, 1.34 / HEIGHT_IN))
    cax = fig.add_axes((6.52 / WIDTH_IN, 0.88 / HEIGHT_IN, 0.085 / WIDTH_IN, 1.18 / HEIGHT_IN))

    positions = np.arange(len(source_counts), dtype=float)
    bottom = np.zeros(len(source_counts))
    handles = []
    labels = []
    for mode in modes:
        heights = np.array([joint.get((n, mode), 0) for n in source_counts], dtype=float)
        bars = ax_a.bar(positions, heights, 0.62, bottom=bottom, color=MODE_COLOR[mode],
                        edgecolor="white", linewidth=0.3, label=mode_label[mode], zorder=3)
        handles.append(bars[0])
        labels.append(mode_label[mode])
        for xpos, height, base in zip(positions, heights, bottom):
            if height > 0:
                ax_a.text(xpos, base + height / 2.0, f"{int(height):,}", ha="center",
                          va="center", fontsize=FS_ANNOT, color="white", zorder=4)
        bottom += heights
    for xpos, height in zip(positions, bottom):
        ax_a.text(xpos, height + 900, f"{int(height):,}", ha="center", va="bottom",
                  fontsize=FS_TICK, color="#1a1a1a", zorder=4)
    ax_a.set_xticks(positions)
    ax_a.set_xticklabels([f"{n}" for n in source_counts])
    ax_a.set_xlabel("Sources per sample")
    ax_a.set_ylabel("Samples")
    ax_a.set_ylim(0, 66000)
    ax_a.set_yticks([0, 20000, 40000, 60000])
    ax_a.set_yticklabels(["0", "20k", "40k", "60k"])
    ax_a.set_xlim(-0.62, len(source_counts) - 0.38)
    legend = fig.legend(handles, labels, loc="lower left",
                        bbox_to_anchor=(0.46 / WIDTH_IN, 2.24 / HEIGHT_IN), ncol=2,
                        fontsize=FS_TICK, handlelength=0.9, handleheight=0.6,
                        handletextpad=0.4, columnspacing=1.0, borderpad=0.0,
                        labelspacing=0.2, borderaxespad=0.0, frameon=False)
    style_axes(ax_a)
    ax_a.text(0.02, 0.98, "(a)", transform=ax_a.transAxes, ha="left", va="top",
              fontsize=FS_TICK, color="#1a1a1a")

    ax_b.hist(snr_centres, bins=snr_edges, weights=snr_bins, color="#0072B2",
              edgecolor="white", linewidth=0.3, zorder=3)
    ax_b.set_xlabel("Mixture SNR (dB)")
    ax_b.set_ylabel("Samples")
    ax_b.set_xlim(SNR_EDGES[0], SNR_EDGES[-1])
    ax_b.set_xticks([-10, 0, 10, 20, 30, 40])
    ax_b.set_ylim(0, 8600)
    ax_b.set_yticks([0, 2500, 5000, 7500])
    ax_b.set_yticklabels(["0", "2.5k", "5k", "7.5k"])
    style_axes(ax_b)
    ax_b.text(0.02, 0.98, "(b)", transform=ax_b.transAxes, ha="left", va="top",
              fontsize=FS_TICK, color="#1a1a1a")

    image = ax_c.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax_c.set_xticks(range(4))
    ax_c.set_yticks(range(4))
    ax_c.set_xticklabels([STD_LABEL[s] for s in STANDARDS], rotation=40, ha="right")
    ax_c.set_yticklabels([STD_LABEL[s] for s in STANDARDS])
    ax_c.set_xticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax_c.set_yticks(np.arange(-0.5, 4.0, 1.0), minor=True)
    ax_c.grid(which="minor", color="white", linewidth=0.4)
    ax_c.tick_params(which="minor", length=0)
    ax_c.tick_params(which="major", length=0, pad=1.5)
    for row in range(4):
        for col in range(row, 4):
            shade = norm(matrix[row, col])
            ax_c.text(col, row, f"{int(matrix[row, col]):,}", ha="center", va="center",
                      fontsize=FS_ANNOT, color=cell_text_colour(cmap, shade))
    ax_c.text(0.0, 1.015, "(c)", transform=ax_c.transAxes, ha="left", va="bottom",
              fontsize=FS_TICK, color="#1a1a1a")
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=FS_ANNOT, length=1.6, pad=1.0)
    bar.set_label("2-source samples", fontsize=FS_ANNOT, labelpad=1.5)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer)
    axes_box = ax_a.get_window_extent()
    clear = legend_box.y0 >= axes_box.y1
    print("legend rect (in)  x[{:.3f},{:.3f}] y[{:.3f},{:.3f}]".format(
        legend_box.x0 / fig.dpi, legend_box.x1 / fig.dpi,
        legend_box.y0 / fig.dpi, legend_box.y1 / fig.dpi))
    print("ax_a rect   (in)  x[{:.3f},{:.3f}] y[{:.3f},{:.3f}]".format(
        axes_box.x0 / fig.dpi, axes_box.x1 / fig.dpi,
        axes_box.y0 / fig.dpi, axes_box.y1 / fig.dpi))
    print(f"legend strictly above ax_a data area: {clear}")
    problems = geometry_check(fig)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)

    print("counts by source count:",
          {str(n): int(sum(joint.get((n, m), 0) for m in modes)) for n in source_counts})
    print("counts by mode:",
          {m: int(sum(joint.get((n, m), 0) for n in source_counts)) for m in modes})
    print("joint:",
          {f"{n}/{m}": int(joint.get((n, m), 0)) for n in source_counts for m in modes})
    upper = [int(matrix[row, col]) for row in range(4) for col in range(row, 4)]
    print("pair matrix upper triangle (row major):", upper, "sum", sum(upper))
    print("same-standard (diagonal) total:", int(np.nansum(np.diag(matrix))))
    print("snr bins:", [int(v) for v in snr_bins], "total", int(np.sum(snr_bins)))
    print("lowest non-zero bin sum check:",
          int(np.sum(snr_bins[:5])), "vs", int(np.sum(snr_bins[5:10])))
    print(f"width {WIDTH_IN} in, height {HEIGHT_IN} in, smallest font {FS_ANNOT} pt")
    print(f"saved {OUT}")
    print(f"saved {PREVIEW}")
    if problems:
        raise SystemExit(f"geometry check failed with {problems} problem(s)")


if __name__ == "__main__":
    main()
