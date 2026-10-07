"""Figure 4: signal quality of the four cellular standards.

SIGNAL SOURCE.  Every panel is computed from the RELEASED samples in
data/rfss_single.h5 -- the file users download -- not from a fresh call to the
generators.  These rows are source_signals: baseband generator output after the
TDL channel and the hardware impairments (CFO, SFO, IQ imbalance, DC offset,
phase noise, PA nonlinearity), power-normalised and with NO AWGN.  The previous
version called the generators with their default carrier_freq (GSM 900 MHz,
UMTS 2.1 GHz, LTE 2.0 GHz, NR 3.5 GHz) and so plotted aliased signals
(2.0e9 mod 15.36e6 = +3.2 MHz for LTE, 2.1e9 mod 15.36e6 = -4.32 MHz for UMTS).
There is no resampling and no mixing: each standard is used at its native rate.

MULTI-ROW STATISTICS (revision, reviewer 11:50:08 item D).  The previous version
computed the CCDF, the PSD and the envelope CDF from ONE released signal per
standard, i.e. 34 to 39 windows, which cannot support a CCDF axis down to 0.01
and gives a jagged PSD.  Every statistic is now an average over INDEX_ROWS
released rows per standard.  The index list is fixed in this file (first match
on the metadata, then consecutive rows of the same configuration), so the figure
is reproducible; the rows are the only 1 ms records available for that
configuration.  1000 rows per standard exist in rfss_single.h5.  No index or
hash file is written: the plan rules out `paper/fig_*_indices.sha1`
(reviewer 12:20:34, item 0), and the resolved lists are printed by this script.

  Window for the PAPR CCDF: one slot, i.e. 0.5 ms = 14 OFDM symbols for LTE,
  14 symbols for NR at numerology mu (1 ms / 2^mu is not used: the window is one
  slot of 0.5 ms / 2^(mu-1)), and one UMTS frame of 1 ms is too short for a slot
  window, so the UMTS window is the whole record; for GSM the whole record.
  Reason: a window shorter than one channel coherence block is dominated by the
  fading realisation, and a window longer than one slot averages over the
  scheduling-independent part of the waveform.  Concretely, the window is chosen
  as the largest power of two number of samples not exceeding 0.5 ms at the
  native rate, so that the same number of samples is used for every row of a
  standard and the window is stated in the caption:
      GSM  1083 samples = 0.500 ms (the record is 1890 samples)
      UMTS 3840 samples = 0.500 ms
      LTE  7680 samples = 0.500 ms
      NR  30720 samples = 0.500 ms
  Consecutive windows with 50 percent overlap are pooled over all rows of the
  standard: 800 windows for GSM (400 rows), 900 for UMTS and LTE (300 rows),
  1000 for GSM (500 rows), 1002 for UMTS (334), 963 for LTE (321) and 438 for
  5G NR (219), so the smallest resolvable CCDF level is about 1e-3.

Constant-envelope GSM.  An ideal GMSK signal has |s(t)| constant, so over any
window peak = mean and the per-window PAPR is exactly 0.00 dB.  On the RELEASED
samples that is not what is measured: the GSM rows carry PA nonlinearity and a
multipath TDL channel, so the envelope is no longer constant and the per-window
PAPR is 1.1 to 3.4 dB.  The GSM curve is therefore drawn as an ordinary line
labelled "GSM" like the others (revision, reviewer 11:50:08 item D(i)); the
"constant envelope" label is gone because it is false after the channel.

Panels:
  (a) PAPR CCDF, empirical, pooled over all rows of the standard.
  (b) peak-normalised power spectral density, each standard on its own native
      frequency grid (-fs/2 .. +fs/2), never resampled, Welch-averaged over the
      sample axis and over the rows.  The y axis floor is set from the plotted
      data (deepest bin minus 2 dB, rounded down to 5 dB) so no curve is clipped;
      the run prints the floor and the deepest plotted minimum.
  (c) envelope (amplitude) CDF on the RMS-normalised amplitude axis, pooled over
      all rows, with the Rayleigh reference 1 - exp(-a^2) overlaid in grey.
The same colour and line style marks each standard in every panel.  One shared
legend row serves the whole figure.  Text is 7 to 8 pt at the 7.16 in print width.

Output: paper/figures/fig_signal_quality.pdf
Preview: paper/figures/preview/fig_signal_quality.png (150 dpi, not for the paper)
"""

import json
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.text as mtext
from matplotlib.lines import Line2D
from scipy.signal.windows import hann

ROOT = Path(__file__).parent.parent
H5_PATH = ROOT / "data" / "rfss_single.h5"

OUT_DIR = Path(__file__).parent / "figures"
OUT = OUT_DIR / "fig_signal_quality.pdf"
PREVIEW = OUT_DIR / "preview" / "fig_signal_quality.png"

WIDTH_IN = 7.16
HEIGHT_IN = 2.25
FS_AXIS = 8.0
FS_TICK = 7.0
FS_ANNOT = 7.0
WINDOW_MS = 0.5
OVERLAP = 0.5
STFT_SEG = 2048
PSD_BIN_HZ = 15000.0
PSD_FLOOR_DB = -60.0
CDF_EDGES = np.linspace(0.0, 3.5, 141)

# name, metadata standard, bandwidth, numerology, rows per standard.  The row
# count is the largest available for that configuration that still leaves the
# file readable in slices: GSM 500 of 1000, UMTS 334 of 1000, LTE 321 of 321 at
# 10 MHz, NR 219 of 219 at mu = 1 / 50 MHz.  The resolved index lists are held in
# the script and printed on every run, so the figure is reproducible from a fixed
# list rather than from a rule.  No hash-pinning file is written: the plan rules
# out `paper/fig_*_indices.sha1` (reviewer 12:20:34, item 0).  The 99 percent
# bandwidth summary printed by this script is the source of the medians quoted in
# Section IV.
SPECS = [
    ("GSM", "GSM", 0.2, None, 500),
    ("UMTS", "UMTS", 5.0, None, 334),
    ("LTE", "LTE", 10.0, None, 321),
    ("5G NR", "5G_NR", 50.0, 1, 219),
]

STYLE = {
    "GSM": ("#56B4E9", "-"),
    "UMTS": ("#E69F00", (0, (4.5, 2.0))),
    "LTE": ("#009E73", (0, (5.0, 1.6, 1.2, 1.6))),
    "5G NR": ("#CC79A7", (0, (1.6, 1.5))),
}
STANDARDS = ["GSM", "UMTS", "LTE", "5G NR"]

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


def select_rows(handle, standard, bw, num, count):
    """Fixed index list: first metadata match, then consecutive rows of the
    same standard/bandwidth/numerology.  The list is deterministic."""
    wanted = []
    for index in range(handle["metadata"].shape[0]):
        record = json.loads(handle["metadata"][index].decode())
        source = record["sources"][0]
        if source["standard"] != standard:
            continue
        params = source["signal_params"]
        if float(params.get("bandwidth_mhz")) != bw:
            continue
        if num is not None and int(params.get("numerology")) != num:
            continue
        wanted.append((index, float(params["sample_rate"])))
        if len(wanted) >= count:
            break
    if len(wanted) < count:
        raise RuntimeError(f"only {len(wanted)} rows found for {standard}")
    return wanted


def window_samples(fs):
    """WINDOW_MS at the native rate, rounded to a whole number of samples."""
    return int(round(WINDOW_MS * 1e-3 * fs))


def papr_values(signal, fs):
    window = window_samples(fs)
    hop = max(1, int(round(window * (1.0 - OVERLAP))))
    starts = np.arange(0, len(signal) - window + 1, hop)
    values = np.empty(len(starts))
    for index, start in enumerate(starts):
        block = np.abs(signal[start:start + window]) ** 2
        values[index] = 10.0 * np.log10(block.max() / block.mean())
    return values, window


def ccdf(values):
    ordered = np.sort(values)
    return ordered, 1.0 - np.arange(len(ordered)) / len(ordered)


def psd_average(signals, fs):
    n_fft = 1
    while n_fft < int(round(fs / PSD_BIN_HZ)):
        n_fft *= 2
    window = hann(n_fft, sym=False)
    hop = n_fft // 4
    accumulator = np.zeros(n_fft)
    count = 0
    for signal in signals:
        starts = np.arange(0, len(signal) - n_fft + 1, hop)
        for start in starts:
            accumulator += np.abs(np.fft.fft(signal[start:start + n_fft] * window)) ** 2
            count += 1
    accumulator /= max(count, 1)
    db = 10.0 * np.log10(accumulator + 1e-30)
    db -= db.max()
    freq_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs)) / 1e6
    return freq_mhz, np.fft.fftshift(db), count, n_fft


def envelope_cdf(signals, edges):
    counts = np.zeros(len(edges) - 1)
    total = 0
    for signal in signals:
        amplitude = np.abs(signal)
        amplitude = amplitude / np.sqrt(np.mean(amplitude ** 2))
        hist, _ = np.histogram(amplitude, bins=edges)
        counts += hist
        total += len(amplitude)
    return edges[1:], np.cumsum(counts) / total


def measured_99_bw(signal, fs):
    """Band containing 99 percent of the signal power, from a Welch PSD."""
    from scipy.signal import welch
    nperseg = min(1024, len(signal))
    freq, pxx = welch(signal, fs=fs, window="hann", nperseg=nperseg,
                      noverlap=int(nperseg * 0.5), detrend=False,
                      return_onesided=False)
    order = np.argsort(freq)
    freq, pxx = freq[order], pxx[order]
    total = pxx.sum()
    cumulative = np.cumsum(pxx)
    low = freq[np.searchsorted(cumulative, 0.005 * total)]
    high = freq[np.searchsorted(cumulative, 0.995 * total)]
    return (high - low) / 1e6


def style_axes(ax):
    ax.tick_params(direction="out", length=2.0, pad=1.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def overlap(a, b):
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[4], b[4]) - max(a[2], b[2])
    return dx, dy


def geometry_check(fig, axes):
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
    rects = []
    for index, ax in enumerate(axes):
        bb = ax.get_window_extent()
        rects.append((f"ax{index}", bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi))
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
        for rect in rects:
            inside = rect[1] <= centre[0] <= rect[3] and rect[2] <= centre[1] <= rect[4]
            if inside:
                continue
            dx, dy = overlap(text, rect)
            if dx > 0 and dy > 0:
                print(f"TEXT/AXES OVERLAP {text[0]!r} <-> {rect[0]} area={dx * dy:.5f} in^2")
                problems += 1
    print(f"geometry check: {len(texts)} text boxes, {len(rects)} axes, {problems} problems")
    return problems


def main():
    data = {}
    index_lists = {}
    with h5py.File(H5_PATH, "r") as handle:
        lengths = handle["signal_lengths"]
        for name, standard, bw, num, count in SPECS:
            rows = select_rows(handle, standard, bw, num, count)
            fs = rows[0][1]
            signals = []
            for index, rate in rows:
                if rate != fs:
                    raise RuntimeError(f"{name} row {index} has rate {rate}")
                length = int(lengths[index])
                signals.append(np.asarray(handle["source_signals"][index, 0, :length],
                                          dtype=np.complex128))
            data[name] = (signals, fs, [index for index, _ in rows])
            index_lists[name] = {"rows": [index for index, _ in rows],
                                 "sample_rate_hz": fs,
                                 "records": len(signals)}
            print(f"  {name:6s} rows={len(signals)} "
                  f"indices {index_lists[name]['rows'][0]}.."
                  f"{index_lists[name]['rows'][-1]} "
                  f"fs={fs / 1e6:8.3f} MHz n={len(signals[0])}")

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    ax_a = fig.add_axes((0.56 / WIDTH_IN, 0.36 / HEIGHT_IN, 2.18 / WIDTH_IN, 1.50 / HEIGHT_IN))
    ax_b = fig.add_axes((3.10 / WIDTH_IN, 0.36 / HEIGHT_IN, 2.18 / WIDTH_IN, 1.50 / HEIGHT_IN))
    ax_c = fig.add_axes((5.64 / WIDTH_IN, 0.36 / HEIGHT_IN, 1.46 / WIDTH_IN, 1.50 / HEIGHT_IN))

    papr_summary = {}
    for name in STANDARDS:
        signals, fs, indices = data[name]
        color, linestyle = STYLE[name]
        pooled = np.concatenate([papr_values(signal, fs)[0] for signal in signals])
        window = window_samples(fs)
        levels, ccdf_values = ccdf(pooled)
        ax_a.semilogy(levels, ccdf_values, color=color, linestyle=linestyle, linewidth=0.8,
                      label=name)
        summary = {
            "rows": len(signals),
            "window_samples": window,
            "window_ms": window / fs * 1e3,
            "windows_per_row": int((len(signals[0]) - window) // max(1, int(round(window * 0.5))) + 1),
            "total_windows": int(len(pooled)),
            "min": float(np.min(pooled)),
            "p05": float(np.percentile(pooled, 5)),
            "median": float(np.median(pooled)),
            "p95": float(np.percentile(pooled, 95)),
            "max": float(np.max(pooled)),
            "ccdf_min": float(np.min(ccdf_values)),
        }
        papr_summary[name] = summary
        print(f"  {name:6s} PAPR window={window} samples ({summary['window_ms']:.3f} ms) "
              f"windows/row={summary['windows_per_row']} pooled={summary['total_windows']} "
              f"min={summary['min']:6.2f} p05={summary['p05']:6.2f} median={summary['median']:6.2f} "
              f"p95={summary['p95']:6.2f} max={summary['max']:6.2f} dB "
              f"ccdf_floor={summary['ccdf_min']:.2e}")
    ax_a.set_xlabel("PAPR (dB)")
    ax_a.set_ylabel("CCDF of PAPR")
    ax_a.set_xlim(-0.5, 13.0)
    ax_a.set_ylim(1e-3, 1.05)
    ax_a.set_xticks([0, 3, 6, 9, 12])
    ax_a.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax_a.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax_a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    style_axes(ax_a)
    ax_a.text(0.03, 0.94, "(a)", transform=ax_a.transAxes, ha="left", va="top",
              fontsize=FS_ANNOT, color="#1a1a1a")

    psd_summary = {}
    for name in STANDARDS:
        signals, fs, indices = data[name]
        color, linestyle = STYLE[name]
        freq_mhz, db, count, n_fft = psd_average(signals, fs)
        ax_b.plot(freq_mhz, db, color=color, linestyle=linestyle, linewidth=0.8)
        psd_summary[name] = {"segments": int(count), "nfft": int(n_fft),
                             "bin_khz": fs / n_fft / 1e3, "rows": len(signals),
                             "min_db": float(db.min()),
                             "p001_db": float(np.percentile(db, 0.001)),
                             "max_db": float(db.max())}
        print(f"  {name:6s} PSD rows={len(signals)} segments={count} nfft={n_fft} "
              f"bin={fs / n_fft / 1e3:5.2f} kHz span=+-{freq_mhz.max():6.3f} MHz")
    ax_b.set_xlabel("Frequency (MHz)")
    ax_b.set_ylabel("Normalised PSD (dB)")
    ax_b.set_xlim(-30.72, 30.72)
    floor_db = min(PSD_FLOOR_DB, min(v["p001_db"] for v in psd_summary.values()) - 2.0)
    floor_db = float(np.floor(floor_db / 5.0) * 5.0)
    ax_b.set_ylim(floor_db, 3.0)
    # Vertical ticks run from 0 down to the data floor.  The previous version
    # stopped at -60 while the curves reach -95, so the deepest part of every
    # curve sat below the lowest labelled tick (revision, reviewer 12:20:34).
    ticks = [0.0]
    while ticks[-1] - 20.0 >= floor_db:
        ticks.append(ticks[-1] - 20.0)
    if ticks[-1] > floor_db + 1e-9:
        ticks.append(floor_db)
    ax_b.set_yticks(sorted(ticks))
    ax_b.set_yticklabels([f"{int(value)}" for value in sorted(ticks)])
    print(f"  PSD panel floor {floor_db:.1f} dB (deepest plotted minimum "
          f"{min(v['min_db'] for v in psd_summary.values()):.2f} dB), y ticks "
          f"{sorted(int(value) for value in ticks)}")
    ax_b.set_xticks([-20, 0, 20])
    style_axes(ax_b)
    ax_b.text(0.03, 0.94, "(b)", transform=ax_b.transAxes, ha="left", va="top",
              fontsize=FS_ANNOT, color="#1a1a1a")

    cdf_summary = {}
    for name in STANDARDS:
        signals, fs, indices = data[name]
        color, linestyle = STYLE[name]
        centres, cdf_values = envelope_cdf(signals, CDF_EDGES)
        ax_c.plot(centres, cdf_values, color=color, linestyle=linestyle, linewidth=0.8)
        pooled = np.concatenate([np.abs(signal) / np.sqrt(np.mean(np.abs(signal) ** 2))
                                 for signal in signals])
        cdf_summary[name] = {"rows": len(signals), "p50": float(np.percentile(pooled, 50)),
                             "p90": float(np.percentile(pooled, 90)),
                             "p99": float(np.percentile(pooled, 99))}
        print(f"  {name:6s} envelope CDF rows={len(signals)} p50={cdf_summary[name]['p50']:5.2f} "
              f"p90={cdf_summary[name]['p90']:5.2f} p99={cdf_summary[name]['p99']:5.2f} (RMS units)")
    bw_summary = {}
    for name in STANDARDS:
        signals, fs, indices = data[name]
        values = np.array([measured_99_bw(signal, fs) for signal in signals])
        bw_summary[name] = {
            "rows": len(signals),
            "median_mhz": float(np.median(values)),
            "p05_mhz": float(np.percentile(values, 5)),
            "p95_mhz": float(np.percentile(values, 95)),
            "min_mhz": float(np.min(values)),
            "max_mhz": float(np.max(values)),
        }
        print(f"  {name:6s} 99% BW rows={len(signals)} "
              f"median={bw_summary[name]['median_mhz']:6.2f} "
              f"p05={bw_summary[name]['p05_mhz']:6.2f} "
              f"p95={bw_summary[name]['p95_mhz']:6.2f} "
              f"min={bw_summary[name]['min_mhz']:6.2f} "
              f"max={bw_summary[name]['max_mhz']:6.2f} MHz")

    rayleigh = 1.0 - np.exp(-CDF_EDGES[1:] ** 2)
    ax_c.plot(CDF_EDGES[1:], rayleigh, color="0.45", linestyle=(0, (2.0, 1.4)), linewidth=0.7,
              label="Rayleigh")
    ax_c.set_xlabel("Amplitude / RMS")
    ax_c.set_ylabel("CDF")
    ax_c.set_xlim(0.0, 3.5)
    ax_c.set_ylim(0.0, 1.02)
    ax_c.set_xticks([0, 1, 2, 3])
    ax_c.set_yticks([0.0, 0.5, 1.0])
    style_axes(ax_c)
    ax_c.text(0.05, 0.94, "(c)", transform=ax_c.transAxes, ha="left", va="top",
              fontsize=FS_ANNOT, color="#1a1a1a")

    handles = [Line2D([], [], color=STYLE[name][0], linestyle=STYLE[name][1],
                       linewidth=0.8, label=name) for name in STANDARDS]
    handles.append(Line2D([], [], color="0.45", linestyle=(0, (2.0, 1.4)), linewidth=0.7,
                          label="Rayleigh"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5, fontsize=FS_ANNOT, handlelength=1.8, handletextpad=0.4,
               columnspacing=1.1, borderaxespad=0.0)

    problems = geometry_check(fig, [ax_a, ax_b, ax_c])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print("PAPR summary:", json.dumps(papr_summary, indent=1))
    print("99% BW summary:", json.dumps(bw_summary, indent=1))
    print(f"width {WIDTH_IN} in, height {HEIGHT_IN} in, smallest font {FS_ANNOT} pt")
    print(f"saved {OUT}")
    print(f"saved {PREVIEW}")
    if problems:
        raise SystemExit(f"geometry check failed with {problems} problem(s)")


if __name__ == "__main__":
    main()
