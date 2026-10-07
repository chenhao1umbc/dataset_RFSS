"""Figure 3: single-source spectrograms for the four cellular standards.

SIGNAL SOURCE (blocking fix).  Every panel is computed from the RELEASED
samples in data/rfss_single.h5 -- the file users download -- not from a fresh
call to the generators.  The previous version called the generators with their
default carrier_freq (GSM 900 MHz, UMTS 2.1 GHz, LTE 2.0 GHz, NR 3.5 GHz).
add_carrier_frequency multiplies by exp(j 2 pi fc t) at the native sample rate,
so the plotted signals aliased: 2.0e9 mod 15.36e6 = +3.2 MHz (LTE) and
2.1e9 mod 15.36e6 = -4.32 MHz (UMTS), and similarly for GSM and NR.  Reading the
released file removes the mismatch between script and generator for good.

The rows plotted are source_signals: baseband generator output after the TDL
channel and the hardware impairments (CFO, SFO, IQ imbalance, DC offset, phase
noise, PA nonlinearity), power-normalised and with NO AWGN.  We deliberately do
NOT plot mixed_signals, which carries AWGN at the sample SNR (up to 40 dB) and
would smear the spectral floor.

DETERMINISTIC SELECTIONS (first match, parsed from the metadata JSON strings,
never hardcoded):
  GSM    index 0     (fs 2.166 MHz,  1890 samples = 0.8726 ms)
  UMTS   index 1000  (fs 7.68 MHz,   7680 samples = 1.0000 ms)
  LTE    index 2005  (fs 15.36 MHz,  15360 samples, bandwidth 10 MHz)
  5G NR  index 3003  (fs 61.44 MHz,  61348 samples, numerology 1, 50 MHz)

TITLE LABELS.  The GSM panel title states the 200 kHz carrier spacing of the GSM
channel plan ("200 kHz spacing") next to the measured 99 percent occupied
bandwidth printed in the panel, because the released GSM record is shorter than
1 ms and the measured value is what the panel shows.

FREQUENCY SPAN (amended plan).  Each panel is plotted on its OWN native span,
i.e. the y-axis is exactly -fs/2 .. +fs/2 for that panel's sample rate, so the
four panels have different spans: GSM +-1.083 MHz, UMTS +-3.84 MHz,
LTE +-7.68 MHz, NR +-30.72 MHz.  There is one shared time axis (0 to 1 ms), one
shared colour scale and one colour bar for the whole figure.  Because the spans
differ, EVERY panel carries its own y tick labels (revision, reviewer 11:50:08
item C): the previous version dropped the labels on the right column, so the
reader saw the left column's numbers on the UMTS and NR panels.

Panels, left to right and top to bottom: (a) GSM, (b) UMTS, (c) LTE, (d) 5G NR.
Standard names appear in the panel titles only; the coloured in-axes name text
of the previous version (the unreadable pink "5G NR" label) is gone.  The
occupied bandwidth is marked by ONE thin bracket at the right edge of each
panel instead of the two cyan dotted lines.  The bracket label is white with a
thin black outline so it stays legible on any background.

Why the GSM trace ends near 0.87 ms.  generate_gsm_signal computes
num_bits = int(270833 * 1e-3) = 270 bits, then the GMSK baseband uses
samples_per_symbol = int(sample_rate / 270833) = int(2166000 / 270833) = 7
(integer truncation), so the signal is 270 * 7 = 1890 samples = 0.8726 ms
instead of 2166 samples = 1.000 ms.  This is NOT an artefact of the window, the
crop or the native-span change: every one of the 1000 released GSM samples has
length 1890.  It is reported and documented, not fixed.  The trace is plotted on
the shared 0..1 ms axis so the shortfall is visible.

Why the GSM panel is a narrow trace.  A 200 kHz GMSK signal fills only
0.2/2.166 = 9 percent of its native 2.166 MHz span, so it appears as a single
narrow horizontal trace; the rest of the panel is the colour-bar floor.  The NR
panel is at the other extreme: a 47.9 MHz signal on a 61.44 MHz span fills
78 percent of the panel.  The factor-of-240 difference in fractional occupancy
is the property the figure is meant to show.  No window or crop artefact remains
once the released baseband samples are used.

Output: paper/figures/fig_spectrograms.pdf
Preview: paper/figures/preview/fig_spectrograms.png (150 dpi, not for the paper)
"""

import json
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.text as mtext
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.signal.windows import hann

ROOT = Path(__file__).parent.parent
H5_PATH = ROOT / "data" / "rfss_single.h5"

OUT_DIR = Path(__file__).parent / "figures"
OUT = OUT_DIR / "fig_spectrograms.pdf"
PREVIEW = OUT_DIR / "preview" / "fig_spectrograms.png"

WIDTH_IN = 7.16
HEIGHT_IN = 3.50
FS_AXIS = 8.0
FS_TICK = 7.0
FS_ANNOT = 7.0
WINDOW_US = 25.0
HOP_US = 6.25
VMIN_DB = -60.0
VMAX_DB = 0.0
TARGET_BIN_HZ = 25000.0
CMAP = "inferno"

STD_COLOR = {"GSM": "#56B4E9", "UMTS": "#E69F00", "LTE": "#009E73", "5G NR": "#CC79A7"}

# name, metadata standard, bandwidth, numerology, nominal occupied BW (MHz),
# title, panel letter, y ticks (MHz)
PANELS = [
    ("GSM", "GSM", 0.2, None, 0.2, "GSM, 200 kHz spacing, GMSK", "(a)", [-1.0, 0.0, 1.0]),
    ("UMTS", "UMTS", 5.0, None, 5.0, "UMTS, 5 MHz, W-CDMA", "(b)", [-3.0, 0.0, 3.0]),
    ("LTE", "LTE", 10.0, None, 10.0, "LTE, 10 MHz, OFDM", "(c)", [-6.0, 0.0, 6.0]),
    ("5G NR", "5G_NR", 50.0, 1, 47.9, "5G NR, 50 MHz, CP-OFDM", "(d)", [-30.0, 0.0, 30.0]),
]

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
})


def select_samples(handle):
    """First deterministic match per standard, parsed from the metadata JSON."""
    wanted = {panel[0]: panel for panel in PANELS}
    found = {}
    for index, raw in enumerate(handle["metadata"]):
        if len(found) == len(PANELS):
            break
        record = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        source = record["sources"][0]
        params = source["signal_params"]
        for name, standard, bw, num, _occ, _title, _letter, _ticks in PANELS:
            if name in found or source["standard"] != standard:
                continue
            if float(params.get("bandwidth_mhz")) != bw:
                continue
            if num is not None and int(params.get("numerology")) != num:
                continue
            found[name] = (index, float(params["sample_rate"]))
    missing = [name for name in wanted if name not in found]
    if missing:
        raise RuntimeError(f"released samples not found for {missing}")
    return found


def load_signal(handle, index, length):
    """Read exactly one released row (~1 MB) as complex128."""
    row = handle["source_signals"][index, 0, :length]
    return np.asarray(row, dtype=np.complex128)


def spectrogram(signal, fs):
    window_len = int(round(WINDOW_US * 1e-6 * fs))
    hop_len = max(1, int(round(HOP_US * 1e-6 * fs)))
    n_fft = 1
    while n_fft < max(window_len, int(round(fs / TARGET_BIN_HZ))):
        n_fft *= 2
    window = hann(window_len, sym=False)
    n_frames = (len(signal) - window_len) // hop_len + 1
    frames = np.zeros((n_fft, n_frames), dtype=np.complex128)
    for index in range(n_frames):
        start = index * hop_len
        frames[:, index] = np.fft.fft(signal[start:start + window_len] * window, n=n_fft)
    frames = np.fft.fftshift(frames, axes=0)
    power_db = 20.0 * np.log10(np.abs(frames) + 1e-12)
    power_db -= power_db.max()
    times_ms = (np.arange(n_frames) * hop_len + window_len / 2.0) / fs * 1e3
    freqs_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs)) / 1e6
    return times_ms, freqs_mhz, power_db


def measured_99_bw(signal, fs):
    """Band containing 99 percent of the signal power, from a Welch PSD."""
    nperseg = min(1024, len(signal))
    freq, pxx = _welch(signal, fs, nperseg)
    total = pxx.sum()
    cumulative = np.cumsum(pxx)
    low = freq[np.searchsorted(cumulative, 0.005 * total)]
    high = freq[np.searchsorted(cumulative, 0.995 * total)]
    return (high - low) / 1e6


def _welch(signal, fs, nperseg):
    from scipy.signal import welch
    freq, pxx = welch(signal, fs=fs, window="hann", nperseg=nperseg,
                      noverlap=int(nperseg * 0.5), detrend=False, return_onesided=False)
    order = np.argsort(freq)
    return freq[order], pxx[order]


def outline():
    return [path_effects.withStroke(linewidth=1.0, foreground="black", alpha=0.9)]


def draw_bracket(ax, occ_hz, fs):
    """One thin bracket at the right edge marking the occupied band.

    The band occupies the centred fraction occ/fs of the native span, so in
    axes coordinates it runs from 0.5 - (occ/fs)/2 to 0.5 + (occ/fs)/2.
    """
    x = 0.955
    half_frac = 0.5 * (occ_hz / fs)
    y_lo = 0.5 - half_frac
    y_hi = 0.5 + half_frac
    cap = 0.030
    ax.plot([x, x], [y_lo, y_hi], transform=ax.transAxes, color="white", linewidth=0.7,
            path_effects=outline(), zorder=7, solid_capstyle="butt")
    for y in (y_lo, y_hi):
        ax.plot([x - cap, x], [y, y], transform=ax.transAxes, color="white", linewidth=0.7,
                path_effects=outline(), zorder=7, solid_capstyle="butt")


def overlap(a, b):
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[4], b[4]) - max(a[2], b[2])
    return dx, dy


def geometry_check(fig, panels):
    """Numeric audit in figure inches: text vs text, text vs foreign axes, and
    one check that every panel actually carries y tick labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = fig.dpi
    texts = []
    seen = set()
    for artist in fig.findobj(match=mtext.Text):
        if id(artist) in seen or not artist.get_text().strip() or not artist.get_visible():
            continue
        seen.add(id(artist))
        bb = artist.get_window_extent(renderer=renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        texts.append((artist.get_text(), bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi))
    axes_rects = []
    for index, ax in enumerate(panels):
        bb = ax.get_window_extent()
        axes_rects.append((f"ax{index}", bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi))
    problems = 0
    for item in texts:
        if item[1] < -0.002 or item[3] > WIDTH_IN + 0.002 or item[2] < -0.002 \
                or item[4] > HEIGHT_IN + 0.002:
            print(f"OUT OF PAGE  {item[0]!r} x[{item[1]:.3f},{item[3]:.3f}] "
                  f"y[{item[2]:.3f},{item[4]:.3f}]")
            problems += 1
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            dx, dy = overlap(texts[i], texts[j])
            if dx > 0 and dy > 0:
                print(f"TEXT OVERLAP {texts[i][0]!r} <-> {texts[j][0]!r} area={dx * dy:.5f} in^2")
                problems += 1
    for text in texts:
        centre = (0.5 * (text[1] + text[3]), 0.5 * (text[2] + text[4]))
        for rect in axes_rects:
            inside = rect[1] <= centre[0] <= rect[3] and rect[2] <= centre[1] <= rect[4]
            if inside:
                continue
            dx, dy = overlap(text, rect)
            if dx > 0 and dy > 0:
                print(f"TEXT/AXES OVERLAP {text[0]!r} <-> {rect[0]} area={dx * dy:.5f} in^2")
                problems += 1
    for index, ax in enumerate(panels):
        labels = [item.get_text() for item in ax.get_yticklabels() if item.get_text()]
        if not labels:
            print(f"PANEL {index} has no y tick labels")
            problems += 1
        else:
            print(f"  panel {index} y tick labels: {labels}")
    print(f"geometry check: {len(texts)} text boxes, {len(axes_rects)} axes, {problems} problems")
    return problems


def main():
    floor = plt.get_cmap(CMAP)(0.0)
    with h5py.File(H5_PATH, "r") as handle:
        found = select_samples(handle)
        lengths = handle["signal_lengths"]
        data = {}
        for name, standard, bw, num, occ, title, letter, ticks in PANELS:
            index, fs = found[name]
            length = int(lengths[index])
            signal = load_signal(handle, index, length)
            data[name] = (signal, fs, index, occ)

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    panel_w, panel_h, left, gap = 2.76, 1.20, 0.52, 0.42
    right = left + panel_w + gap
    top_y, bot_y = 2.10, 0.62
    rects = [(left, top_y), (right, top_y), (left, bot_y), (right, bot_y)]

    panels = []
    for (name, standard, bw, num, occ, title, letter, ticks), (x0, y0) in zip(PANELS, rects):
        signal, fs, index, _occ = data[name]
        times_ms, freqs_mhz, power_db = spectrogram(signal, fs)
        half = 0.5 * fs / 1e6
        keep = np.abs(freqs_mhz) <= half + 1e-9
        ax = fig.add_axes((x0 / WIDTH_IN, y0 / HEIGHT_IN, panel_w / WIDTH_IN, panel_h / HEIGHT_IN))
        ax.set_facecolor(floor)
        ax.pcolormesh(times_ms, freqs_mhz[keep], power_db[keep, :], vmin=VMIN_DB, vmax=VMAX_DB,
                      cmap=CMAP, shading="auto", rasterized=True)
        bw99 = measured_99_bw(signal, fs)
        draw_bracket(ax, occ * 1e6, fs)
        ax.text(0.030, 0.935, f"99% BW\n{bw99:.2f} MHz", transform=ax.transAxes,
                ha="left", va="top", fontsize=FS_ANNOT, color="#1a1a1a", zorder=8,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="0.4", linewidth=0.3, alpha=0.85))
        ax.set_title(title, fontsize=FS_TICK, pad=1.8, color=STD_COLOR[name])
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-half, half)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{value:g}" for value in ticks])
        ax.tick_params(direction="out", length=2.0, pad=1.5)
        ax.tick_params(axis="x", pad=3.2)
        panels.append(ax)
        print(f"  {name:6s} idx={index:5d} fs={fs / 1e6:8.3f} MHz n={len(signal):6d} "
              f"dur={len(signal) / fs * 1e3:7.4f} ms span=+-{half:6.3f} MHz "
              f"nominal={occ:5.2f} MHz measured99={bw99:5.2f} MHz "
              f"frames={len(times_ms)} grid={int(keep.sum())}")

    for ax in (panels[0], panels[2]):
        ax.set_ylabel("Frequency (MHz)")
    for ax, (name, standard, bw, num, occ, title, letter, ticks) in zip(panels, PANELS):
        ax.text(0.030, 0.055, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=FS_ANNOT, color="#1a1a1a", zorder=8,
                bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                          edgecolor="0.4", linewidth=0.3, alpha=0.85))
    for ax in (panels[0], panels[1]):
        ax.tick_params(labelbottom=False)
    fig.text(3.475 / WIDTH_IN, 0.14 / HEIGHT_IN, "Time (ms)", ha="center", va="center",
             fontsize=FS_AXIS)

    cax = fig.add_axes((6.62 / WIDTH_IN, 0.60 / HEIGHT_IN, 0.09 / WIDTH_IN, 2.70 / HEIGHT_IN))
    bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=VMIN_DB, vmax=VMAX_DB), cmap=CMAP), cax=cax)
    bar.outline.set_linewidth(0.4)
    bar.ax.tick_params(labelsize=FS_ANNOT, length=1.8, pad=1.2)
    bar.set_label("PSD (dB)\npeak-normalised", fontsize=FS_ANNOT, labelpad=1.2)

    problems = geometry_check(fig, panels)

    for ax in panels:
        rect = ax.get_window_extent()
        print(f"  panel rect (in) x[{rect.x0 / fig.dpi:.3f},{rect.x1 / fig.dpi:.3f}] "
              f"y[{rect.y0 / fig.dpi:.3f},{rect.y1 / fig.dpi:.3f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print(f"width {WIDTH_IN} in, smallest font {FS_ANNOT} pt")
    print(f"saved {OUT}")
    print(f"saved {PREVIEW}")
    if problems:
        raise SystemExit(f"geometry check failed with {problems} problem(s)")


if __name__ == "__main__":
    main()
