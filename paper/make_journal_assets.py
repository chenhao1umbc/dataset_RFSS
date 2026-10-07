"""Build the journal paper's result tables and numbers from the frozen test passes and the validation sweep records.

Reads check/eval_all_src2_crop0_frozen_results.json and check/eval_all_src34_crop0_frozen_results.json (the headline pass,
random window, crop seed 0) and check/encoder_sweep_results.json (learning-rate screening); writes paper/tables/table_main.tex,
table_modes.tex, table_snr.tex, table_lr.tex, table_cost.tex (from check/epoch_times.json) and paper/journal_numbers.json. Run from the project root:
    uv run python paper/make_journal_assets.py
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
N_BOOT = 2000
FAMILIES = {"conv": "Conv-TasNet", "dprnn": "DPRNN", "stft": "STFT-BLSTM"}
SNR_BINS = [("$-10$--$10$~dB", -10, 10), ("10--20~dB", 10, 20), ("20--30~dB", 20, 30), ("${>}30$~dB", 30, 40.001)]
SCREEN_EPOCH = 2
ESCAPE_DB = 4.0
LR_ROWS = [
    ("Conv-TasNet", [("1e-4", ["l16_lr1e-4"]), ("3e-4", ["l16_lr3e-4", "l16_lr3e-4_seed1", "l16_lr3e-4_seed2"]), ("1e-3", ["l16"])], "3e-4"),
    ("Conv-TasNet-L256", [("3e-5", ["l256_lr3e-5"]), ("1e-4", ["l256_lr1e-4"]), ("3e-4", ["l256_lr3e-4"])], "1e-4"),
    ("DPRNN", [("1e-4", ["dprnn_lr1e-4"]), ("3e-4", ["dprnn_lr3e-4", "dprnn_lr3e-4_seed1", "dprnn_lr3e-4_seed2"]),
               ("1e-3", ["dprnn_lr1e-3", "dprnn_lr1e-3_seed1", "dprnn_lr1e-3_seed2"])], "1e-3"),
    ("CNN-LSTM (transposed decoder)", [("1e-4", ["cnn_lstm_tconv_lr1e-4"]), ("3e-4", ["cnn_lstm_tconv_lr3e-4"]), ("1e-3", ["cnn_lstm_tconv_lr1e-3"])], "3e-4"),
    ("STFT-BLSTM", [("3e-4", ["stft_lr3e-4", "stft_lr3e-4_seed1", "stft_lr3e-4_seed2"]), ("1e-3", ["stft", "stft_seed1", "stft_seed2"])], "3e-4"),
]

rng = np.random.RandomState(0)


def signed(v: float) -> str:
    return f"+{v:.2f}" if v >= 0 else f"$-${abs(v):.2f}"


def boot(x: np.ndarray) -> list:
    means = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def interval_cell(x: np.ndarray) -> str:
    lo, hi = boot(x)
    return f"{signed(x.mean())} [{signed(lo)}, {signed(hi)}]"


def load_pass(name: str) -> dict:
    """Per-source-count arrays of one frozen pass: gains over input of every method and the sample descriptors."""
    d = json.loads((ROOT / "check" / name).read_text())
    assert d["crop_seed"] == 0 and d["split"] == "test" and not d["git_code_modified"] and d["n_sentinel_scores"] == 0
    out = {}
    for n_src in sorted({r["num_sources"] for r in d["samples"]}):
        rows = [r for r in d["samples"] if r["num_sources"] == n_src]
        skip = ("idx", "num_sources", "mode", "snr_db", "length", "input")
        inp = np.array([r["input"] for r in rows])
        out[n_src] = {"input": inp, "mode": np.array([r["mode"] for r in rows]), "snr": np.array([r["snr_db"] for r in rows]),
                      "gain": {m: np.array([r[m] for r in rows]) - inp for m in rows[0] if m not in skip}, "commit": d["git_commit"]}
    return out


def family_gain(p: dict, fam: str, mask: np.ndarray) -> float:
    """Mean over the available seeds of the mean gain of one family in the samples of mask."""
    labels = [m for m in p["gain"] if m.startswith(fam + "_s")]
    return float(np.mean([p["gain"][m][mask].mean() for m in labels]))


def main_table(passes: dict, numbers: dict) -> str:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & 2-source & 3-source & 4-source \\", r"\midrule"]
    for method, label in (("ica", "ICA"), ("nmf", "NMF")):
        lines.append(f"{label} & " + " & ".join(interval_cell(passes[n]["gain"][method]) for n in (2, 3, 4)) + r" \\")
    for fam, label in FAMILIES.items():
        cells = []
        for n in (2, 3, 4):
            labels = [m for m in passes[n]["gain"] if m.startswith(fam + "_s")]
            means = [passes[n]["gain"][m].mean() for m in labels]
            if n == 2:
                cells.append(f"{signed(np.mean(means))} ({signed(min(means))} to {signed(max(means))})")
            else:
                cells.append(interval_cell(passes[n]["gain"][labels[0]]))
            numbers.setdefault(fam, {})[f"{n}src"] = {"mean": float(np.mean(means)), "seeds": [float(v) for v in means]}
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for method, label in (("irm_oracle", "IRM oracle"), ("oracle", "Noise-limited oracle")):
        lines.append(f"{label} & " + " & ".join(interval_cell(passes[n]["gain"][method]) for n in (2, 3, 4)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def subset_table(p: dict, columns: list, header: str, count_row: bool) -> str:
    """Rows of ICA, NMF, the three families and the IRM oracle for the sample subsets given as (label, mask)."""
    lines = [r"\begin{tabular}{l" + "r" * len(columns) + "}", r"\toprule", header, r"\midrule"]
    for method, label in (("ica", "ICA"), ("nmf", "NMF")):
        lines.append(f"{label} & " + " & ".join(signed(p["gain"][method][m].mean()) for _, m in columns) + r" \\")
    for fam, label in FAMILIES.items():
        lines.append(f"{label} & " + " & ".join(signed(family_gain(p, fam, m)) for _, m in columns) + r" \\")
    lines += [r"\midrule", "IRM oracle & " + " & ".join(signed(p["gain"]["irm_oracle"][m].mean()) for _, m in columns) + r" \\"]
    if count_row:
        lines.append("$N$ & " + " & ".join(f"{int(m.sum()):,}".replace(",", "{,}") for _, m in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def lr_cell(curves: list, chosen: bool) -> tuple:
    """Worst-seed adjacent-channel SNR>20 gain at the screening epoch and the first epoch with the worst seed above ESCAPE_DB."""
    n_epochs = min(len(c) for c in curves)
    worst = [min(c[e] for c in curves) for e in range(n_epochs)]
    gain = worst[SCREEN_EPOCH - 1]
    escape = next((e + 1 for e, g in enumerate(worst) if g > ESCAPE_DB), None)
    g_txt, e_txt = signed(gain), (str(escape) if escape else "--")
    return (r"\textbf{" + g_txt + "}", r"\textbf{" + e_txt + "}") if chosen else (g_txt, e_txt)


def lr_table(sweep: dict) -> str:
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Learning rate & Gain & Escape epoch \\", r"\midrule"]
    for model, grid, chosen in LR_ROWS:
        rates, gains, escapes = [], [], []
        for lr, keys in grid:
            curves = [[e["val"]["adjacent_snr_gt_20"]["gain_over_input_db"] for e in sweep[k]["epochs"]] for k in keys]
            g, e = lr_cell(curves, lr == chosen)
            base, exp = lr.split("e")
            rate = f"10^{{{exp}}}" if base == "1" else f"{base}\\times10^{{{exp}}}"
            rates.append(f"$\\mathbf{{{rate}}}$" if lr == chosen else f"${rate}$")
            gains.append(g)
            escapes.append(e)
        lines.append(f"{model} & {', '.join(rates)} & {', '.join(gains)} & {', '.join(escapes)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def cost_table(runs: list) -> str:
    """Epoch time (min / median / max over the epochs of the runs with seeds 1 and 2 at 2 sources, seed 0 otherwise) and concurrency."""
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule", r"Model & Sources & Parameters & Device & Epoch time (s) & Other runs \\", r"\midrule"]
    for fam, label in FAMILIES.items():
        name = {"conv": "conv_tasnet", "dprnn": "dprnn", "stft": "stft_blstm"}[fam]
        for n in (2, 3, 4):
            group = [r for r in runs if r["run"].startswith(f"{name}_{n}src_seed") and "stuck" not in r["run"]]
            times = np.concatenate([r["epoch_seconds"] for r in group])
            others = np.concatenate([r["concurrent_others"] for r in group])
            lines.append(f"{label} & {n} & {group[0]['params'] / 1e6:.2f}~M & {group[0]['device']} & "
                         f"{times.min():,.0f} / {np.median(times):,.0f} / {times.max():,.0f} & {others.min()}--{others.max()} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def flagged_table(sweep: dict) -> str:
    """Flagged-cell record for the appendix: every 3- or 4-source cell the rule
    marked, its screen outcome, and the final validation gain (all-samples bin)."""
    rows = [
        ("STFT-BLSTM", 3, "2", "cleared at epoch 2",
         "stft_final_stft_blstm_3src_seed0_ep10"),
        ("Conv-TasNet", 3, "2, 4", "slow, not stuck (screen: none)",
         "l16_final_conv_tasnet_3src_seed0_ep9"),
        ("Conv-TasNet", 4, "4", "three rates, no change",
         "l16_final_conv_tasnet_4src_seed0_ep10"),
        ("DPRNN", 3, "2, 4", "restarted at $5\\times10^{-4}$",
         "dprnn_final_dprnn_3src_seed0_ep10"),
        ("DPRNN", 4, "4", "three rates, no winner",
         "dprnn_final_dprnn_4src_seed0_ep7"),
    ]
    lines = [r"\begin{tabular}{llllr}", r"\toprule",
             r"Model & Src. & Flagged at & Screen outcome & Final gain \\", r"\midrule"]
    for model, n, epochs, outcome, key in rows:
        final = signed(float(sweep[key]["val"]["all"]["gain_over_input_db"]))
        lines.append(f"{model} & {n} & {epochs} & {outcome} & {final} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    passes = {**load_pass("eval_all_src2_crop0_frozen_results.json"), **load_pass("eval_all_src34_crop0_frozen_results.json")}
    numbers = {"git_commit_2src": passes[2]["commit"], "git_commit_34src": passes[3]["commit"]}
    numbers["input_mean_db"] = {f"{n}src": float(passes[n]["input"].mean()) for n in (2, 3, 4)}
    numbers["n_test"] = {f"{n}src": int(len(passes[n]["input"])) for n in (2, 3, 4)}
    for method in ("ica", "nmf", "irm_oracle", "oracle"):
        numbers[method] = {f"{n}src": float(passes[n]["gain"][method].mean()) for n in (2, 3, 4)}
    (OUT / "tables").mkdir(exist_ok=True)
    (OUT / "tables" / "table_main.tex").write_text(main_table(passes, numbers))

    p2 = passes[2]
    everything = np.ones(len(p2["input"]), dtype=bool)
    co, adj = p2["mode"] == "co-channel", p2["mode"] == "adjacent-channel"
    adj20 = adj & (p2["snr"] > 20)
    modes = [("All", everything), ("Co-channel", co), ("Adjacent", adj), ("Adjacent, SNR${>}20$~dB", adj20)]
    header = "Method & " + " & ".join(label for label, _ in modes) + r" \\"
    (OUT / "tables" / "table_modes.tex").write_text(subset_table(p2, modes, header, True))
    snr_cols = [(label, (p2["snr"] >= lo) & (p2["snr"] < hi)) for label, lo, hi in SNR_BINS]
    header = "Method & " + " & ".join(label for label, _ in snr_cols) + r" \\"
    (OUT / "tables" / "table_snr.tex").write_text(subset_table(p2, snr_cols, header, False))
    numbers["mode_n"] = {label: int(m.sum()) for label, m in modes}
    numbers["snr_n"] = {label: int(m.sum()) for label, m in snr_cols}
    numbers["adjacent_snr20"] = {fam: family_gain(p2, fam, adj20) for fam in FAMILIES}
    numbers["mode_gain"] = {label: {**{fam: family_gain(p2, fam, m) for fam in FAMILIES}, "irm_oracle": float(p2["gain"]["irm_oracle"][m].mean())} for label, m in modes}

    runs = json.loads((ROOT / "check" / "epoch_times.json").read_text())
    (OUT / "tables" / "table_cost.tex").write_text(cost_table(runs))
    sweep = json.loads((ROOT / "check" / "encoder_sweep_results.json").read_text())
    (OUT / "tables" / "table_lr.tex").write_text(lr_table(sweep))
    (OUT / "tables" / "table_flagged.tex").write_text(flagged_table(sweep))
    (OUT / "journal_numbers.json").write_text(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
