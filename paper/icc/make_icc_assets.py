"""Build the ICC paper's numbers, table and figure from the frozen primary test pass (crop seed 0).

Reads check/eval_all_src2_crop0_frozen_results.json; writes paper/icc/icc_numbers.json, paper/icc/table_main.tex
and paper/icc/figures/fig_snr.pdf. No number in the paper text is typed by hand: the text includes these files
or quotes icc_numbers.json.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "check" / "eval_all_src2_crop0_frozen_results.json"
OUT = Path(__file__).resolve().parent
N_BOOT = 2000
FAMILIES = {"stft": "STFT-BLSTM", "dprnn": "DPRNN", "conv": "Conv-TasNet"}
PARAMS = {"stft": "7.36", "dprnn": "1.11", "conv": "2.52"}
SNR_EDGES = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40.001)]

d = json.loads(SRC.read_text())
rows = d["samples"]
assert d["crop_seed"] == 0 and len(rows) == 7526
inp = np.array([r["input"] for r in rows])
mode = np.array([r["mode"] for r in rows])
snr = np.array([r["snr_db"] for r in rows])
adj = (mode == "adjacent-channel") & (snr > 20)
co = (mode == "co-channel") & (snr > 20)
gain = {m: np.array([r[m] for r in rows]) - inp for m in rows[0] if m not in ("idx", "num_sources", "mode", "snr_db", "length", "input")}
rng = np.random.RandomState(0)


def boot(x):
    n = len(x)
    b = [x[rng.randint(0, n, n)].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


num = {"n": len(rows), "n_adjacent_snr20": int(adj.sum()), "n_cochannel_snr20": int(co.sum()),
       "input_mean_db": float(inp.mean()), "git_commit": d["git_commit"], "crop_seed": d["crop_seed"]}
for m in ("irm_oracle", "oracle", "ica", "nmf"):
    num[m] = {"all": float(gain[m].mean()), "ci": boot(gain[m]), "adj": float(gain[m][adj].mean()), "co": float(gain[m][co].mean())}
fam = {}
for k in FAMILIES:
    seeds = [gain[f"{k}_s{i}"] for i in range(3)]
    fam[k] = {"seeds_all": [float(s.mean()) for s in seeds], "mean_all": float(np.mean([s.mean() for s in seeds])),
              "range_all": [float(min(s.mean() for s in seeds)), float(max(s.mean() for s in seeds))],
              "mean_adj": float(np.mean([s[adj].mean() for s in seeds])), "mean_co": float(np.mean([s[co].mean() for s in seeds])),
              "ci_seed1": boot(seeds[1]), "irm_fraction": float(np.mean([s.mean() for s in seeds]) / gain["irm_oracle"].mean())}
num["families"] = fam
pairs = {}
for a, b in (("stft", "dprnn"), ("dprnn", "conv"), ("stft", "conv")):
    per = []
    for i in range(3):
        x = gain[f"{a}_s{i}"] - gain[f"{b}_s{i}"]
        per.append({"all": float(x.mean()), "ci": boot(x), "adj": float(x[adj].mean()), "co": float(x[co].mean())})
    pairs[f"{a}-{b}"] = {"per_seed": per, "mean_all": float(np.mean([p["all"] for p in per])),
                         "range_all": [min(p["all"] for p in per), max(p["all"] for p in per)]}
num["pairs"] = pairs
bins = []
for lo, hi in SNR_EDGES:
    sel = (snr >= lo) & (snr < hi)
    row = {"lo": lo, "hi": hi, "n": int(sel.sum())}
    for k in FAMILIES:
        row[k] = float(np.mean([gain[f"{k}_s{i}"][sel].mean() for i in range(3)]))
    for m in ("irm_oracle", "nmf", "ica"):
        row[m] = float(gain[m][sel].mean())
    bins.append(row)
num["snr_bins"] = bins
(OUT / "icc_numbers.json").write_text(json.dumps(num, indent=1))

f = lambda v: f"{v:+.2f}"
lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Method & Params & All & Adjacent & Co-channel \\", r"\midrule"]
lines.append(f"ICA & -- & {f(num['ica']['all'])} & {f(num['ica']['adj'])} & {f(num['ica']['co'])} \\\\")
lines.append(f"NMF & -- & {f(num['nmf']['all'])} & {f(num['nmf']['adj'])} & {f(num['nmf']['co'])} \\\\")
for k, name in FAMILIES.items():
    s = fam[k]["seeds_all"]
    lines.append(f"{name} & {PARAMS[k]}M & {f(fam[k]['mean_all'])} ({s and min(s):+.2f} to {max(s):+.2f}) & {f(fam[k]['mean_adj'])} & {f(fam[k]['mean_co'])} \\\\")
lines += [r"\midrule", f"IRM oracle & -- & {f(num['irm_oracle']['all'])} & {f(num['irm_oracle']['adj'])} & {f(num['irm_oracle']['co'])} \\\\",
          f"Noise-limited oracle & -- & {f(num['oracle']['all'])} & {f(num['oracle']['adj'])} & {f(num['oracle']['co'])} \\\\",
          r"\bottomrule", r"\end{tabular}"]
(OUT / "table_main.tex").write_text("\n".join(lines) + "\n")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mid = [(b["lo"] + min(b["hi"], 40)) / 2 for b in bins]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
style = {"stft": ("o-", "STFT-BLSTM"), "dprnn": ("s-", "DPRNN"), "conv": ("^-", "Conv-TasNet"),
         "irm_oracle": ("k--", "IRM oracle"), "nmf": ("x:", "NMF"), "ica": ("v:", "ICA")}
for k, (st, lab) in style.items():
    ax.plot(mid, [b[k] for b in bins], st, label=lab, markersize=3.5, linewidth=1.1)
ax.axhline(0, color="0.6", linewidth=0.6)
ax.set_xlabel("Per-sample SNR bin centre (dB)")
ax.set_ylabel("Gain over input (dB)")
ax.legend(fontsize=6, ncol=2, frameon=False, loc="lower right")
ax.grid(alpha=0.3, linewidth=0.4)
fig.tight_layout()
fig.savefig(OUT / "figures" / "fig_snr.pdf")
print(json.dumps({k: num[k] for k in ("n", "n_adjacent_snr20", "n_cochannel_snr20", "input_mean_db")}))
print([(b["lo"], b["n"], round(b["stft"], 2), round(b["irm_oracle"], 2), round(b["nmf"], 2)) for b in bins])
print({k: (round(v["mean_all"], 2), round(v["irm_fraction"], 3)) for k, v in fam.items()})
