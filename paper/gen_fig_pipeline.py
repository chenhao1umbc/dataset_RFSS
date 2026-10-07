"""Figure 1: RFSS dataset construction pipeline block diagram.

One branch per drawn source.  The generator draws each source independently
WITH replacement, so K branches are drawn for K = 2, 3 or 4 sources and a
mixture may contain the same standard twice (e.g. two 5G NR sources).

Branch order, verified against the code:
    generator          src/generate_dataset.py lines 50-88 (carrier_freq=0.0)
    TDL channel        src/generate_dataset.py lines 99-105, TR 38.901
    impairments        CFO 111-119, SFO 122-127, I/Q 130-136,
                       DC 139-144, phase noise 147-153, PA 156-162
    SignalMixer        src/generate_dataset.py lines 206-221
    AWGN               src/generate_dataset.py line 226
    HDF5 write         src/generate_dataset.py lines 390-394

STORED REFERENCES ARE TAKEN BEFORE MIXING (revision, reviewer 12:20:34, item 1).
The HDF5 file stores the mixture, the per-source references and the metadata,
but the references are the channel- and impairment-distorted source waveforms at
their native rate, before resampling, frequency shift and power scaling:
src/generate_dataset.py line 231 keeps `source_signals` (the generator output)
and line 222 keeps `mixed_signal` (the mixer output).  The previous revision
drew the references and the metadata as outputs of the AWGN sum, which
contradicts the text.  This revision draws them as DASHED taps that leave each
impairment output, join a dashed rail, and reach the reference and metadata
boxes, while the mixture ALONE leaves the sum:
    impairment output -> bus -> mixer -> adder -> mixture box (solid)
    impairment output -> dashed tap -> dashed rail -> references, metadata

There is deliberately NO separate power-scaling box: each generator already
normalises its own output power internally (normalize_power, for example
src/run_gsm.py line 51), and the per-source power ratio is applied by the
SignalMixer itself (src/generate_dataset.py line 215, power_db=...).  The old
"Power scaling + CFO / SFO" box and the old ordering that put AWGN inside the
sum before the mixer have both been deleted.

The two header lines of the previous revision (the generator/spec list and the
"K identical branches" note) are NOT drawn in the image; they moved into the
LaTeX caption of Figure 1 (revision, reviewer 12:20:34, item 1).

LAYOUT IS MEASURED, NOT GUESSED.  Each block is sized from the rendered width
of its longest label plus a fixed pad, its box height is set from the number of
text lines it holds, and the vertical pitch is checked against the box height.
geometry_check reports, in inches, box-vs-box overlaps, text-vs-text overlaps,
text-vs-foreign-box overlaps and every connector segment against every box
rectangle, because the defect this revision had to remove was between box edges,
which a text-only audit cannot see.

Cleanup of an earlier revision (reviewer 11:50:08, item E):
  * The hardware-impairments text is compressed to three lines inside a box
    0.54 in high in a 0.90 in row pitch, so the three copies have a visible
    vertical gap.
  * The "AWGN at the sample SNR" label is placed below the summing junction.
  * The bottom-left colour key (GSM / UMTS / LTE / 5G NR) is deleted: the
    colours carry no meaning in this diagram.

Drawn at the final print size: 7.16 in wide, text 7.0 to 7.5 pt, vector PDF.
Output:  paper/figures/fig_pipeline.pdf
Preview: paper/figures/preview/fig_pipeline.png (150 dpi, not for the paper)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).parent / "figures"
OUT = OUT_DIR / "fig_pipeline.pdf"
PREVIEW = OUT_DIR / "preview" / "fig_pipeline.png"

W = 7.16
H = 2.72
FS_SMALL = 7.0
FS_TEXT = 7.5
PAD_X = 0.07
PAD_Y = 0.060
LEADING = 0.135
MARGIN = 0.12
RIGHT_MARGIN = 0.10
ROW_PITCH = 0.80
ROW_TOP = 2.145
TALLEST_LIMIT = 0.70
# Minimum clear space between the AWGN label and the SignalMixer box bottom edge
# (reviewer 13:27:07, item 3), in points.
CLEARANCE_PT = 4.0

BOX_FILL = "#f4f4f4"
BOX_EDGE = "#5c5c5c"
TEXT_DARK = "#1a1a1a"
TEXT_GREY = "#4d4d4d"
ARROW_COLOR = "#333333"
DASH_STYLE = (0, (2.4, 1.5))

GEN_TEXTS = ["Generator", "GSM / UMTS / LTE / NR"]
TDL_TEXTS = ["TDL channel", "TR 38.901"]
SYM_TEXTS = ["Symbols"]
IMP_TEXTS = ["Hardware impairments", "CFO, SFO, I/Q, DC,", "phase noise, PA"]
MIX_TEXTS = ["SignalMixer", "rate = max source rate", "power ratio per source",
             "offset 0 (co-channel), k x 2 MHz"]
OUT_TEXTS = [["Mixture"], ["Per-source", "references"], ["Metadata", "(JSON)"]]
AWGN_TEXT = "AWGN at the sample SNR"

# Widths of the zero-width columns that carry only a connector rail.
RAIL_W = 0.0
BUS_W = 0.0
TRUNK_W = 0.0
DASH_W = 0.0
JUNC_W = 0.20

GAP_KEYS = ("sym-gen", "gen-tdl", "tdl-imp", "imp-rail", "rail-bus", "bus-mix",
            "mix-junc", "junc-trunk", "trunk-dash", "dash-out")
GAP_BASE = (0.16, 0.16, 0.16, 0.10, 0.08, 0.14, 0.12, 0.10, 0.10, 0.09)

# The dashed reference rail runs to the RIGHT of the summing junction and BELOW
# the mixture row, and terminates only on the references and metadata boxes
# (reviewer 12:57:59, item 1).  RAIL_BOTTOM is the y of the horizontal run that
# carries the rail past the junction; it sits below the metadata box.
RAIL_BOTTOM = 0.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_TEXT,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

TEXTS = []
BOXES = []
SEGMENTS = []
RAIL_SEGMENTS = []
MIXTURE_BOX = []
MIXER_BOX = []
AWGN_ARTIST = []


def add_box(ax, cx, cy, w, h):
    patch = FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=BOX_FILL, edgecolor=BOX_EDGE, linewidth=0.6, zorder=2,
    )
    ax.add_patch(patch)
    BOXES.append((f"box@{cx:.2f},{cy:.2f}", cx - w / 2.0 - 0.02, cy - h / 2.0 - 0.02,
                  cx + w / 2.0 + 0.02, cy + h / 2.0 + 0.02))
    return patch


def add_arrow(ax, x0, y0, x1, y1, style="solid", rail=False):
    dashed = style == "dashed"
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5,
        color=ARROW_COLOR, linewidth=0.6, shrinkA=0.0, shrinkB=0.0, zorder=4,
        linestyle=DASH_STYLE if dashed else "solid",
    ))
    SEGMENTS.append((f"{style} arrow", x0, y0, x1, y1))
    if rail:
        RAIL_SEGMENTS.append((f"{style} arrow", x0, y0, x1, y1))


def add_line(ax, x0, y0, x1, y1, style="solid", rail=False):
    dashed = style == "dashed"
    ax.plot([x0, x1], [y0, y1], color=ARROW_COLOR, linewidth=0.6, zorder=4,
            linestyle=DASH_STYLE if dashed else "solid",
            solid_capstyle="butt", dash_capstyle="butt")
    SEGMENTS.append((f"{style} line", x0, y0, x1, y1))
    if rail:
        RAIL_SEGMENTS.append((f"{style} line", x0, y0, x1, y1))


def label(ax, x, y, text, size, color, style="normal", ha="center", va="center"):
    artist = ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color,
                     style=style, zorder=5)
    TEXTS.append((text, artist))
    return artist


def text_block(ax, cx, cy, lines):
    y0 = cy + (len(lines) - 1) * LEADING / 2.0
    for index, (text, size, color, style) in enumerate(lines):
        label(ax, cx, y0 - index * LEADING, text, size, color, style)


def measure(fig, text, size):
    artist = fig.text(0.0, 0.0, text, fontsize=size)
    width = artist.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    artist.remove()
    return width


def layout(fig):
    """Column centres from the rendered label widths, and the row geometry."""
    lines = {
        "sym": [(SYM_TEXTS[0], FS_TEXT)],
        "gen": [(t, FS_TEXT if i == 0 else FS_SMALL) for i, t in enumerate(GEN_TEXTS)],
        "tdl": [(t, FS_TEXT if i == 0 else FS_SMALL) for i, t in enumerate(TDL_TEXTS)],
        "imp": [(t, FS_TEXT if i == 0 else FS_SMALL) for i, t in enumerate(IMP_TEXTS)],
        "mix": [(t, FS_TEXT if i == 0 else FS_SMALL) for i, t in enumerate(MIX_TEXTS)],
    }
    out_lines = {"out": [(t, FS_TEXT) for group in OUT_TEXTS for t in group]}
    widths = {name: max(measure(fig, t, s) for t, s in block) + 2.0 * PAD_X
              for name, block in {**lines, **out_lines}.items()}
    heights = {name: len(block) * LEADING + 2.0 * PAD_Y for name, block in lines.items()}
    heights["junc"] = 2.0 * JUNC_W

    out_heights = [len(group) * LEADING + 2.0 * PAD_Y for group in OUT_TEXTS]

    fixed = (widths["sym"] + widths["gen"] + widths["tdl"] + widths["imp"]
             + widths["mix"] + JUNC_W + widths["out"]
             + RAIL_W + BUS_W + TRUNK_W + DASH_W)
    slack = (W - MARGIN - RIGHT_MARGIN) - fixed - sum(GAP_BASE)
    if slack < 0.0:
        raise SystemExit(f"layout does not fit: overflow {-slack:.3f} in")
    shares = [0.10, 0.10, 0.10, 0.06, 0.05, 0.12, 0.11, 0.10, 0.12, 0.14]
    gaps = {key: base + slack * share
            for key, base, share in zip(GAP_KEYS, GAP_BASE, shares)}

    centres = {}
    x = MARGIN
    ordered = ("sym", "gen", "tdl", "imp", "rail", "bus", "mix", "junc", "trunk",
               "dash", "out")
    zero = {"rail": RAIL_W, "bus": BUS_W, "trunk": TRUNK_W, "dash": DASH_W,
            "junc": JUNC_W}
    for index, name in enumerate(ordered):
        width = zero[name] if name in zero else widths[name]
        centres[name] = x + width / 2.0
        x += width
        if index < len(GAP_KEYS):
            x += gaps[GAP_KEYS[index]]

    tallest = max(heights["gen"], heights["tdl"], heights["imp"])
    if tallest >= ROW_PITCH:
        raise SystemExit(f"row pitch {ROW_PITCH} too small for box {tallest:.3f} in")
    if max(out_heights) >= ROW_PITCH:
        raise SystemExit("HDF5 boxes do not fit the row pitch")
    rows = [ROW_TOP - i * ROW_PITCH for i in range(3)]

    print("box widths (in):", {k: round(float(v), 3) for k, v in widths.items()})
    print("box heights (in):", {k: round(float(v), 3) for k, v in heights.items()})
    print("out box heights (in):", [round(v, 3) for v in out_heights])
    print("gaps (in):", {k: round(float(v), 3) for k, v in gaps.items()})
    print("centres (in):", {k: round(float(v), 3) for k, v in centres.items()})
    print(f"row pitch {ROW_PITCH:.3f} in, tallest branch box {tallest:.3f} in, "
          f"visible vertical gap {ROW_PITCH - tallest:.3f} in")
    print(f"page width used: {x + RIGHT_MARGIN:.3f} in of {W:.3f} in")
    if tallest > TALLEST_LIMIT:
        raise SystemExit(f"branch box {tallest:.3f} in exceeds {TALLEST_LIMIT} in")
    return widths, heights, out_heights, centres, rows


def _overlap(a, b):
    a0, a1, a2, a3 = a[-4:]
    b0, b1, b2, b3 = b[-4:]
    dx = min(a2, b2) - max(a0, b0)
    dy = min(a3, b3) - max(a1, b1)
    return dx, dy


def _contains(box, point):
    return box[1] <= point[0] <= box[3] and box[2] <= point[1] <= box[4]


def _shrink(box, margin):
    return (box[1], box[2], box[3], box[4] - margin)


def geometry_check(fig, ax, texts, boxes, segments):
    """Numeric audit in inches: box-vs-box, text-vs-text, text-vs-foreign-box,
    and every connector segment against every box rectangle."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    text_boxes = []
    for text, artist in texts:
        bb = artist.get_window_extent(renderer=renderer)
        xs, ys = [], []
        for corner in ((bb.x0, bb.y0), (bb.x1, bb.y1)):
            point = inverse.transform(corner)
            xs.append(point[0])
            ys.append(point[1])
        text_boxes.append((text, min(xs), min(ys), max(xs), max(ys)))
    problems = 0
    for item in text_boxes + list(boxes):
        text, x0, y0, x1, y1 = item
        if x0 < -0.002 or x1 > W + 0.002 or y0 < -0.002 or y1 > H + 0.002:
            print(f"OUT OF PAGE  {text!r} x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}]")
            problems += 1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dx, dy = _overlap(boxes[i], boxes[j])
            if dx > 0.0 and dy > 0.0:
                print(f"BOX OVERLAP {boxes[i][0]} <-> {boxes[j][0]} "
                      f"area={dx * dy:.5f} in^2")
                problems += 1
    for text in text_boxes:
        centre = (0.5 * (text[1] + text[3]), 0.5 * (text[2] + text[4]))
        for box in boxes:
            if _contains(box, centre):
                continue
            dx, dy = _overlap(text, box)
            if dx > 0.0 and dy > 0.0:
                print(f"TEXT/BOX OVERLAP {text[0]!r} <-> {box[0]} "
                      f"area={dx * dy:.5f} in^2")
                problems += 1
    for i in range(len(text_boxes)):
        for j in range(i + 1, len(text_boxes)):
            dx, dy = _overlap(text_boxes[i], text_boxes[j])
            if dx > 0.0 and dy > 0.0:
                print(f"TEXT OVERLAP {text_boxes[i][0]!r} <-> {text_boxes[j][0]!r} "
                      f"area={dx * dy:.5f} in^2")
                problems += 1
    for name, x0, y0, x1, y1 in segments:
        seg = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        for box in boxes:
            inner = _shrink(box, 0.02)
            if inner[2] <= inner[0] or inner[3] <= inner[1]:
                continue
            dx, dy = _overlap(seg, inner)
            if dx > 0.005 and dy > 0.005:
                print(f"SEGMENT/BOX OVERLAP {name} <-> {box[0]} "
                      f"area={dx * dy:.5f} in^2")
                problems += 1
    print(f"geometry check: {len(text_boxes)} text boxes, {len(boxes)} box "
          f"rectangles, {len(segments)} segments, {problems} problems")
    return problems


def _bbox_data(ax, artist):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = artist.get_window_extent(renderer=renderer)
    inverse = ax.transData.inverted()
    xs, ys = [], []
    for corner in ((bb.x0, bb.y0), (bb.x1, bb.y1)):
        point = inverse.transform(corner)
        xs.append(point[0])
        ys.append(point[1])
    return (min(xs), min(ys), max(xs), max(ys))


def rail_audit(fig, ax, rail_segments, awgn_artist, mixture_box, mixer_box):
    """Numeric check that the dashed reference rail crosses neither the AWGN
    label bbox nor the mixture box rectangle, and (defensively) no other text
    box or box rectangle.  This is the check a text-only audit cannot do
    (reviewer 12:57:59, item 1).  It also measures the clear space between the
    AWGN label and the bottom edge of the SignalMixer box (reviewer 13:27:07,
    item 3): the label must clear the box by at least CLEARANCE_PT points."""
    print(f"RAIL GEOMETRY (in): AWGN label bbox "
          f"x[{awgn_artist.get_window_extent().x0:.1f}px]")
    awgn = _bbox_data(ax, awgn_artist)
    mix = (mixture_box[1], mixture_box[2], mixture_box[3], mixture_box[4])
    mixer = (mixer_box[1], mixer_box[2], mixer_box[3], mixer_box[4])
    print(f"  AWGN label bbox  x[{awgn[0]:.3f},{awgn[2]:.3f}] y[{awgn[1]:.3f},{awgn[3]:.3f}]")
    print(f"  mixture box rect x[{mix[0]:.3f},{mix[2]:.3f}] y[{mix[1]:.3f},{mix[3]:.3f}]")
    print(f"  SignalMixer rect x[{mixer[0]:.3f},{mixer[2]:.3f}] y[{mixer[1]:.3f},{mixer[3]:.3f}]")
    clearance_in = mixer[1] - awgn[3]
    clearance_pt = 72.0 * clearance_in
    print(f"  AWGN label to SignalMixer bottom edge: {clearance_in:.4f} in "
          f"= {clearance_pt:.2f} pt clear space")
    problems = 0
    if clearance_pt < CLEARANCE_PT:
        print(f"  AWGN LABEL TOO CLOSE to the SignalMixer box: {clearance_pt:.2f} pt "
              f"< {CLEARANCE_PT} pt")
        problems += 1
    other_text = []
    for text, artist in TEXTS:
        if artist is awgn_artist:
            continue
        other_text.append((text, *_bbox_data(ax, artist)))
    other_boxes = [b for b in BOXES if b is not mixture_box]
    for name, x0, y0, x1, y1 in rail_segments:
        seg = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        for label, target in (("AWGN label", awgn), ("mixture box", mix)):
            dx, dy = _overlap(seg, target)
            if dx > 0.005 and dy > 0.005:
                print(f"  RAIL CROSSES {label}  segment {name} "
                      f"x[{x0:.3f},{x1:.3f}] y[{y0:.3f},{y1:.3f}] area={dx * dy:.5f} in^2")
                problems += 1
        for text, q0, q1, q2, q3 in other_text:
            dx, dy = _overlap(seg, (q0, q1, q2, q3))
            if dx > 0.005 and dy > 0.005:
                print(f"  RAIL CROSSES TEXT {text!r} segment {name} area={dx * dy:.5f} in^2")
                problems += 1
        for box in other_boxes:
            inner = _shrink(box, 0.02)
            if inner[2] <= inner[0] or inner[3] <= inner[1]:
                continue
            dx, dy = _overlap(seg, inner)
            if dx > 0.005 and dy > 0.005:
                print(f"  RAIL CROSSES BOX {box[0]} segment {name} area={dx * dy:.5f} in^2")
                problems += 1
    print(f"rail audit: {len(rail_segments)} rail segments, {problems} crossings")
    return problems


def main():
    for path in (OUT, PREVIEW):
        path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, W)
    ax.set_ylim(0.0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    widths, heights, out_heights, centres, rows = layout(fig)
    cx_sym, cx_gen, cx_tdl, cx_imp = (centres["sym"], centres["gen"],
                                      centres["tdl"], centres["imp"])
    rail_x, bus_x, cx_mix = centres["rail"], centres["bus"], centres["mix"]
    junc_x, trunk_x, dash_x, cx_out = (centres["junc"], centres["trunk"],
                                       centres["dash"], centres["out"])
    junc_r = JUNC_W / 2.0
    half_imp = widths["imp"] / 2.0

    for row in rows:
        add_box(ax, cx_sym, row, widths["sym"], heights["sym"])
        add_box(ax, cx_gen, row, widths["gen"], heights["gen"])
        add_box(ax, cx_tdl, row, widths["tdl"], heights["tdl"])
        add_box(ax, cx_imp, row, widths["imp"], heights["imp"])
        text_block(ax, cx_sym, row, [(SYM_TEXTS[0], FS_TEXT, TEXT_DARK, "normal")])
        text_block(ax, cx_gen, row, [
            (GEN_TEXTS[0], FS_TEXT, TEXT_DARK, "normal"),
            (GEN_TEXTS[1], FS_SMALL, TEXT_GREY, "normal"),
        ])
        text_block(ax, cx_tdl, row, [
            (TDL_TEXTS[0], FS_TEXT, TEXT_DARK, "normal"),
            (TDL_TEXTS[1], FS_SMALL, TEXT_GREY, "italic"),
        ])
        text_block(ax, cx_imp, row, [
            (IMP_TEXTS[0], FS_TEXT, TEXT_DARK, "normal"),
            (IMP_TEXTS[1], FS_SMALL, TEXT_DARK, "normal"),
            (IMP_TEXTS[2], FS_SMALL, TEXT_DARK, "normal"),
        ])
        add_arrow(ax, cx_sym + widths["sym"] / 2.0, row, cx_gen - widths["gen"] / 2.0, row)
        add_arrow(ax, cx_gen + widths["gen"] / 2.0, row, cx_tdl - widths["tdl"] / 2.0, row)
        add_arrow(ax, cx_tdl + widths["tdl"] / 2.0, row, cx_imp - widths["imp"] / 2.0, row)
        add_arrow(ax, cx_imp + half_imp, row, bus_x, row)

    # Dashed taps: the stored references and the metadata leave the impairment
    # outputs, not the sum (reviewer 12:20:34, item 1).
    tap_x = cx_imp + 0.30
    tap_ys = [row - 0.33 for row in rows]
    for row, tap_y in zip(rows, tap_ys):
        add_line(ax, tap_x, row - heights["imp"] / 2.0 - 0.02, tap_x, tap_y,
                 style="dashed", rail=True)
        add_line(ax, tap_x, tap_y, rail_x, tap_y, style="dashed", rail=True)
    # The dashed rail runs down the left, along a bottom corridor BELOW the
    # mixture row, and up on the far RIGHT of the summing junction, so it feeds
    # only the per-source-references and metadata boxes (reviewer 12:57:59,
    # item 1).  It never climbs to the mixture row: the mixture box is fed by the
    # solid path alone.
    add_line(ax, rail_x, tap_ys[0], rail_x, RAIL_BOTTOM, style="dashed", rail=True)
    add_line(ax, rail_x, RAIL_BOTTOM, dash_x, RAIL_BOTTOM, style="dashed", rail=True)
    add_line(ax, dash_x, RAIL_BOTTOM, dash_x, rows[1], style="dashed", rail=True)

    # Solid path: impaired sources collect on the bus, enter the mixer, and the
    # mixture alone leaves the summing junction.
    add_line(ax, bus_x, rows[-1], bus_x, rows[0])
    add_arrow(ax, bus_x, rows[1], cx_mix - widths["mix"] / 2.0, rows[1])
    add_box(ax, cx_mix, rows[1], widths["mix"], heights["mix"])
    text_block(ax, cx_mix, rows[1], [
        (MIX_TEXTS[0], FS_TEXT, TEXT_DARK, "normal"),
        (MIX_TEXTS[1], FS_SMALL, TEXT_DARK, "normal"),
        (MIX_TEXTS[2], FS_SMALL, TEXT_DARK, "normal"),
        (MIX_TEXTS[3], FS_SMALL, TEXT_DARK, "normal"),
    ])
    MIXER_BOX.append(BOXES[-1])
    add_arrow(ax, cx_mix + widths["mix"] / 2.0, rows[1], junc_x - junc_r, rows[1])
    ax.add_patch(Circle((junc_x, rows[1]), junc_r, facecolor="white",
                        edgecolor=ARROW_COLOR, linewidth=0.6, zorder=5))
    label(ax, junc_x, rows[1], "+", FS_TEXT, TEXT_DARK)
    add_arrow(ax, junc_x, rows[1] - 0.30, junc_x, rows[1] - junc_r - 0.01)
    AWGN_ARTIST.append(label(ax, junc_x - 0.04, rows[1] - 0.47, AWGN_TEXT, FS_SMALL,
                             TEXT_DARK, ha="right"))
    add_line(ax, junc_x + junc_r, rows[1], trunk_x, rows[1])
    add_line(ax, trunk_x, rows[1], trunk_x, rows[0])

    label(ax, cx_out, rows[0] + out_heights[0] / 2.0 + 0.16, "HDF5 file", FS_TEXT,
          TEXT_GREY, style="italic")
    for index, (cy, group, height) in enumerate(zip(rows, OUT_TEXTS, out_heights)):
        add_box(ax, cx_out, cy, widths["out"], height)
        if index == 0:
            MIXTURE_BOX.append(BOXES[-1])
        text_block(ax, cx_out, cy, [(t, FS_TEXT, TEXT_DARK, "normal") for t in group])
        left = cx_out - widths["out"] / 2.0
        if index == 0:
            add_arrow(ax, trunk_x, rows[0], left, rows[0])
        else:
            add_arrow(ax, dash_x, cy, left, cy, style="dashed", rail=True)

    problems = geometry_check(fig, ax, TEXTS, BOXES, SEGMENTS)
    problems += rail_audit(fig, ax, RAIL_SEGMENTS, AWGN_ARTIST[0], MIXTURE_BOX[0],
                           MIXER_BOX[0])

    fig.savefig(OUT)
    fig.savefig(PREVIEW, dpi=150)
    plt.close(fig)
    print(f"width {W} in, height {H} in, smallest font {FS_SMALL} pt")
    print(f"saved {OUT}")
    print(f"saved {PREVIEW}")
    if problems:
        raise SystemExit(f"geometry check failed with {problems} problem(s)")


if __name__ == "__main__":
    main()
