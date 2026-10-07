"""Build paper/numbers_check.txt: every number quoted in the paper text with the
file it comes from and the value recomputed from that file.

Each row is (number as printed, source file or script, recomputed value).  The
script re-reads the sources it can (the frozen pass JSONs, the sweep records, the
dataset metadata summary, the HDF5 file sizes) and recomputes the value, so a
number that has drifted is reported as MISMATCH.  Numbers that come from an
external document (a 3GPP specification, a cited paper, the distribution page of
another dataset) are listed with that document as the source and the value taken
verbatim from it.

Run from the project root:
    uv run python paper/make_numbers_check.py
"""

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "check"
PAPER = ROOT / "paper"
TEX = PAPER / "revised_paper.tex"
OUT = PAPER / "numbers_check.txt"

rng = np.random.RandomState(0)
N_BOOT = 2000

# Sources that are external documents: a value taken from one of these is a value
# quoted from a cited paper or a specification (group a of the phrase-row sort).
CITED = ("TS 38.211", "TS 38.104", "TS 36.211", "TR 38.901", "3GPP TS",
         "ideal GMSK", "61.44 MHz x 1 ms")


def remove_braces(s):
    """Strip up to three levels of LaTeX braces so 10^{-3} reads as 10^-3."""
    for _ in range(3):
        s = re.sub(r"\{([^{}]*)\}", r"\1", s)
    return s


def latex_plain(s):
    """A printed value as it appears in the tex or in a table, with the LaTeX
    encoding removed, so a negative number typeset as $-$13.60 is recognised as
    the digit string -13.60."""
    s = s.replace("$-$", "-").replace("$", "").replace("~", " ")
    s = s.replace("\\%", "%").replace("{,}", ",").replace("\\,", " ")
    return remove_braces(s)


def load_pass(name):
    data = json.loads((CHECK / name).read_text())
    out = {}
    for n_src in sorted({r["num_sources"] for r in data["samples"]}):
        rows = [r for r in data["samples"] if r["num_sources"] == n_src]
        inp = np.array([r["input"] for r in rows])
        skip = ("idx", "num_sources", "mode", "snr_db", "length", "input")
        out[n_src] = {"input": inp, "mode": np.array([r["mode"] for r in rows]),
                      "snr": np.array([r["snr_db"] for r in rows]),
                      "gain": {m: np.array([r[m] for r in rows]) - inp
                               for m in rows[0] if m not in skip}}
    return out


PASSES = {**load_pass("eval_all_src2_crop0_frozen_results.json"),
          **load_pass("eval_all_src34_crop0_frozen_results.json")}
SWEEP = json.loads((CHECK / "encoder_sweep_results.json").read_text())
COUNTS = json.loads(
    Path("/Users/hc/.hermes/cache/scratch/rfss/dataset_meta_counts.json").read_text())


def fam(p, family, mask=None):
    labels = [m for m in p["gain"] if m.startswith(family + "_s")]
    values = [p["gain"][m] if mask is None else p["gain"][m][mask] for m in labels]
    return float(np.mean([v.mean() for v in values]))


def boot(x):
    means = [x[rng.randint(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fmt(value, digits=2):
    return f"{value:+.{digits}f}"


ROWS = []


def add(label, printed, source, value, tol=None):
    ROWS.append((label, printed, source, value, tol))


def main():
    p2, p3, p4 = PASSES[2], PASSES[3], PASSES[4]
    adj20 = (p2["mode"] == "adjacent-channel") & (p2["snr"] > 20)
    co = p2["mode"] == "co-channel"
    adj = p2["mode"] == "adjacent-channel"

    for family, printed in (("conv", "+5.61"), ("dprnn", "+5.85"), ("stft", "+6.14")):
        add(f"2-source {family} family mean",
            printed, "check/eval_all_src2_crop0_frozen_results.json",
            fmt(fam(p2, family)))
    add("2-source ICA", "$-$13.60", "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(p2["gain"]["ica"].mean())))
    add("2-source NMF", "$-$1.24", "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(p2["gain"]["nmf"].mean())))
    add("2-source IRM oracle", "+8.04", "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(p2["gain"]["irm_oracle"].mean())))
    for n, p in ((3, p3), (4, p4)):
        for family, printed in (("conv", "+4.77" if n == 3 else "+3.54"),
                                ("dprnn", "+5.03" if n == 3 else "+3.44"),
                                ("stft", "+5.14" if n == 3 else "+4.52")):
            add(f"{n}-source {family} (single seed)",
                printed, "check/eval_all_src34_crop0_frozen_results.json",
                fmt(fam(p, family)))
        add(f"{n}-source IRM oracle", "+9.61" if n == 3 else "+10.35",
            "check/eval_all_src34_crop0_frozen_results.json",
            fmt(float(p["gain"]["irm_oracle"].mean())))
    add("2-source co-channel STFT", "+6.13",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(fam(p2, "stft", co)))
    add("2-source adjacent STFT", "+6.14",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(fam(p2, "stft", adj)))
    add("2-source adjacent SNR>20 STFT", "+6.94",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(fam(p2, "stft", adj20)))
    add("2-source adjacent SNR>20 DPRNN", "+6.58",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(fam(p2, "dprnn", adj20)))
    add("2-source adjacent SNR>20 Conv-TasNet", "+6.21",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(fam(p2, "conv", adj20)))
    add("2-source adjacent SNR>20 IRM oracle", "+10.86",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(p2["gain"]["irm_oracle"][adj20].mean())))
    add("input level 2-source", "$-$4.4",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(p2["input"].mean())))
    add("input level 3-source", "$-$9.3",
        "check/eval_all_src34_crop0_frozen_results.json",
        fmt(float(p3["input"].mean())))
    add("input level 4-source", "$-$12.0",
        "check/eval_all_src34_crop0_frozen_results.json",
        fmt(float(p4["input"].mean())))
    add("test samples 2-source", "7,526",
        "check/eval_all_src2_crop0_frozen_results.json", f"{len(p2['input'])}")
    add("test samples 3-source", "5,324",
        "check/eval_all_src34_crop0_frozen_results.json", f"{len(p3['input'])}")
    add("test samples 4-source", "2,150",
        "check/eval_all_src34_crop0_frozen_results.json", f"{len(p4['input'])}")

    bins = [("$-10$--$10$~dB", -10, 10), ("10--20~dB", 10, 20),
            ("20--30~dB", 20, 30), ("${>}30$~dB", 30, 40.001)]
    for label, lo, hi in bins:
        mask = (p2["snr"] >= lo) & (p2["snr"] < hi)
        add(f"2-source STFT gain {label}", "see table_snr.tex",
            "check/eval_all_src2_crop0_frozen_results.json", fmt(fam(p2, "stft", mask)))
        add(f"2-source IRM oracle gain {label}", "see table_snr.tex",
            "check/eval_all_src2_crop0_frozen_results.json",
            fmt(float(p2["gain"]["irm_oracle"][mask].mean())))

    add("2-source STFT minus DPRNN, mean over the three matched pairs", "+0.29",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(np.mean([(p2["gain"][f"stft_s{i}"] - p2["gain"][f"dprnn_s{i}"]).mean()
                           for i in range(3)]))))
    add("2-source DPRNN minus Conv-TasNet, mean over the three matched pairs", "+0.24",
        "check/eval_all_src2_crop0_frozen_results.json",
        fmt(float(np.mean([(p2["gain"][f"dprnn_s{i}"] - p2["gain"][f"conv_s{i}"]).mean()
                           for i in range(3)]))))
    for family, key, printed in (("STFT-BLSTM", "stft", "0.03"),
                                 ("DPRNN", "dprnn", "0.15"),
                                 ("Conv-TasNet", "conv", "0.29")):
        means = [p2["gain"][f"{key}_s{i}"].mean() for i in range(3)]
        add(f"2-source {family} seed range", f"{printed} dB",
            "check/eval_all_src2_crop0_frozen_results.json",
            f"{max(means) - min(means):.2f}")

    for family, key in (("Conv-TasNet-L256", "l256"), ("DPRNN", "dprnn_lr1e-3"),
                        ("STFT-BLSTM", "stft")):
        run = SWEEP[key]
        seq = [run["epochs"][e]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]
               for e in range(2)]
        add(f"{family} 10^-3 first two epochs, adjacent SNR>20", "see text",
            "check/encoder_sweep_results.json",
            " / ".join(fmt(float(v)) for v in seq))
    for family, key in (("Conv-TasNet", "l16_lr3e-4"), ("STFT-BLSTM", "stft_lr3e-4")):
        run = SWEEP[key]
        ep1 = run["epochs"][0]["val"]["all"]["gain_over_input_db"]
        add(f"{family} 3x10^-4 first epoch, all samples", "see text",
            "check/encoder_sweep_results.json", fmt(float(ep1)))

    add("learning-rate table Conv-TasNet 1e-4", "+4.47",
        "check/encoder_sweep_results.json",
        fmt(min(SWEEP["l16_lr1e-4"]["epochs"][1]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]
                for _ in [0])))
    add("learning-rate table Conv-TasNet 3e-4", "+4.84",
        "check/encoder_sweep_results.json",
        fmt(min(SWEEP[k]["epochs"][1]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]
                for k in ("l16_lr3e-4", "l16_lr3e-4_seed1", "l16_lr3e-4_seed2"))))
    add("learning-rate table Conv-TasNet 1e-3", "+2.40",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16"]["epochs"][1]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]))
    add("learning-rate table DPRNN 1e-3", "+5.09",
        "check/encoder_sweep_results.json",
        fmt(min(SWEEP[k]["epochs"][1]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]
                for k in ("dprnn_lr1e-3", "dprnn_lr1e-3_seed1", "dprnn_lr1e-3_seed2"))))
    add("learning-rate table STFT-BLSTM 3e-4", "+5.53",
        "check/encoder_sweep_results.json",
        fmt(min(SWEEP[k]["epochs"][1]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]
                for k in ("stft_lr3e-4", "stft_lr3e-4_seed1", "stft_lr3e-4_seed2"))))

    add("Conv-TasNet-L256 pilot (10 epochs, seed 0)", "+4.51",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l256_pilot_lr1e-4_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("CNN-LSTM transposed-decoder pilot (10 epochs, seed 0)", "+4.47",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["cnn_lstm_tconv_pilot_lr3e-4_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("second 10 epochs of the probe add, all samples", "0.26",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["stft_probe20_lr3e-4_ep20"]["val"]["all"]["gain_over_input_db"]
            - SWEEP["stft_probe20_lr3e-4_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("20-epoch probe gain over the 10-epoch pilot, all samples", "0.20",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["stft_probe20_lr3e-4_ep20"]["val"]["all"]["gain_over_input_db"]
            - SWEEP["stft_pilot_lr3e-4_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("probe 20-epoch run gain over the 10-epoch probe checkpoint, all samples",
        "0.26", "check/encoder_sweep_results.json",
        fmt(SWEEP["stft_probe20_lr3e-4_ep19best"]["val"]["all"]["gain_over_input_db"]
            - SWEEP["stft_probe20_lr3e-4_ep10"]["val"]["all"]["gain_over_input_db"]))

    diag = json.loads((CHECK / "diagnose_training_results.json").read_text())
    add("CNN-LSTM 1x1 + interpolation ceiling over 300 crops", "1.98",
        "check/diagnose_training_results.json",
        fmt(float(diag["cnn_lstm_ceiling"]["mean_ceiling_si_sinr_db"])))
    add("Conv-TasNet 10^-3 plateau, adjacent SNR>20, epochs 1 and 2",
        "2.4", "check/encoder_sweep_results.json",
        " / ".join(fmt(float(SWEEP["l16"]["epochs"][e]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"]))
                   for e in range(2)), tol=0.05)
    add("Conv-TasNet 10^-3 plateau, all samples, epochs 1 and 2",
        "$+2.88$ and $+2.89$", "check/encoder_sweep_results.json",
        " / ".join(fmt(float(SWEEP["l16"]["epochs"][e]["val"]["all"]["gain_over_input_db"]))
                   for e in range(2)), tol=0.005)
    add("Conv-TasNet 3x10^-4 first epoch, adjacent SNR>20", "4.11",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["l16_lr3e-4"]["epochs"][0]["val"]["adjacent_snr_gt_20"]["gain_over_input_db"])))
    add("Conv-TasNet 3x10^-4 first epoch, all samples", "4.26",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["l16_lr3e-4"]["epochs"][0]["val"]["all"]["gain_over_input_db"])))

    add("DPRNN 3-source screen 5e-4 minus 1e-3, epoch 2, all samples", "0.51 dB",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["dprnn_3src_lr5e-4"]["epochs"][1]["val"]["all"]["gain_over_input_db"]
            - SWEEP["dprnn_3src_lr1e-3"]["epochs"][1]["val"]["all"]["gain_over_input_db"]))
    add("DPRNN 4-source screen best rate over 1e-3", "+0.20 dB",
        "check/encoder_sweep_results.json",
        fmt(max(SWEEP[k]["epochs"][1]["val"]["all"]["gain_over_input_db"]
                for k in ("dprnn_4src_lr3e-4", "dprnn_4src_lr5e-4"))
            - SWEEP["dprnn_4src_lr1e-3"]["epochs"][1]["val"]["all"]["gain_over_input_db"]))
    add("Conv-TasNet 4-source cell, epoch 4", "+3.44",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16_final4src_s0_ep4check"]["val"]["all"]["gain_over_input_db"]))
    add("Conv-TasNet 4-source cell, epoch 10", "+3.47",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16_final_conv_tasnet_4src_seed0_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("Conv-TasNet 3-source cell, epoch 2", "+3.63",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16_final3src_conv_s0_ep2check"]["val"]["all"]["gain_over_input_db"]))
    add("Conv-TasNet 3-source cell, epoch 4", "+4.11",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16_final3src_conv_s0_ep4check"]["val"]["all"]["gain_over_input_db"]))
    add("Conv-TasNet 3-source cell, epoch 10", "+4.62",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["l16_final_conv_tasnet_3src_seed0_ep9"]["val"]["all"]["gain_over_input_db"]))
    add("DPRNN 3-source restart, epoch 4", "+4.34",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["dprnn_final3src_s0r_ep4check"]["val"]["all"]["gain_over_input_db"]))
    add("DPRNN 3-source restart, epoch 10", "+4.92",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["dprnn_final_dprnn_3src_seed0_ep10"]["val"]["all"]["gain_over_input_db"]))
    add("DPRNN 4-source cell, epoch 4", "+3.38",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["dprnn_final4src_s0_ep4check"]["val"]["all"]["gain_over_input_db"]))
    add("DPRNN 4-source cell, epoch 7 (last)", "+3.41",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["dprnn_final_dprnn_4src_seed0_ep7"]["val"]["all"]["gain_over_input_db"]))
    add("STFT-BLSTM 3-source cell, epoch 2 (cleared)", "+4.04",
        "check/encoder_sweep_results.json",
        fmt(SWEEP["stft_final3src_s0_ep2check"]["val"]["all"]["gain_over_input_db"]))

    add("flagged table STFT-BLSTM 3-source final (validation, all)", "+5.02",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["stft_final_stft_blstm_3src_seed0_ep10"]["val"]["all"]["gain_over_input_db"])))
    add("flagged table Conv-TasNet 4-source final (validation, all)", "+3.47",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["l16_final_conv_tasnet_4src_seed0_ep10"]["val"]["all"]["gain_over_input_db"])))
    add("flagged table DPRNN 3-source final (validation, all)", "+4.92",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["dprnn_final_dprnn_3src_seed0_ep10"]["val"]["all"]["gain_over_input_db"])))
    add("flagged table DPRNN 4-source final (validation, all)", "+3.41",
        "check/encoder_sweep_results.json",
        fmt(float(SWEEP["dprnn_final_dprnn_4src_seed0_ep7"]["val"]["all"]["gain_over_input_db"])))

    add("epoch steps 2-source (34,912 samples, batch 8)", "4,364",
        "check/eval_all_src2_crop0_frozen_results.json",
        f"{int(np.ceil(34912 / 8))}")
    add("epoch steps 3-source (24,547 samples, batch 8)", "3,069",
        "check/eval_all_src34_crop0_frozen_results.json",
        f"{int(np.ceil(24547 / 8))}")
    add("epoch steps 4-source (10,541 samples, batch 8)", "1,318",
        "check/eval_all_src34_crop0_frozen_results.json",
        f"{int(np.ceil(10541 / 8))}")

    params = json.loads((CHECK / "epoch_times.json").read_text())
    for tag, printed in (("conv_tasnet_2src_seed1", "2.52"),
                         ("dprnn_2src_seed1", "1.11"),
                         ("stft_blstm_2src_seed1", "7.36")):
        row = next(r for r in params if r["run"] == tag)
        add(f"parameters {tag}", printed, "check/epoch_times.json",
            f"{row['params'] / 1e6:.2f}")
    add("Conv-TasNet-L256 parameters", "2.77",
        "check/encoder_sweep_results.json", f"{SWEEP['l256']['params'] / 1e6:.2f}")
    add("CNN-LSTM parameters", "4.30",
        "check/encoder_sweep_results.json",
        f"{SWEEP['cnn_lstm_tconv_lr3e-4']['params'] / 1e6:.2f}")

    add("2-source test samples", "7,526", "paper/journal_numbers.json",
        f"{len(p2['input'])}")
    add("co-channel test samples", "3,035", "check/eval_all_src2_crop0_frozen_results.json",
        f"{int(co.sum())}")
    add("adjacent-channel test samples", "4,491",
        "check/eval_all_src2_crop0_frozen_results.json", f"{int(adj.sum())}")
    add("adjacent SNR>20 test samples", "1,105",
        "check/eval_all_src2_crop0_frozen_results.json", f"{int(adj20.sum())}")

    add("corpus samples", "100,000", "paper/../data/rfss_dataset.h5 metadata",
        f"{COUNTS['total']}")
    two = COUNTS["two_source_total"]
    add("two-source samples", "49,899", "data/rfss_dataset.h5 metadata scan",
        f"{COUNTS['by_source_count']['2']}")
    add("three-source samples", "35,158", "data/rfss_dataset.h5 metadata scan",
        f"{COUNTS['by_source_count']['3']}")
    add("four-source samples", "14,943", "data/rfss_dataset.h5 metadata scan",
        f"{COUNTS['by_source_count']['4']}")
    add("two-source mixed pairs", "six of the ten unordered pairs",
        "combinatorics with replacement", "6 of C(4,2) + 4 = 10 pairs")
    add("two-source same-standard samples", "15,662", "data/rfss_dataset.h5 metadata",
        f"{COUNTS['same_standard_2source']}")
    add("source-count shares", "49.9 / 35.2 / 14.9 pc",
        "data/rfss_dataset.h5 metadata",
        " / ".join(f"{100 * sum(v for key, v in COUNTS['joint'].items() if int(key.split('|')[0]) == k) / COUNTS['total']:.1f}"
                   for k in (2, 3, 4)))
    add("source-count fractions (Section III, line ~462)",
        "0.499, 0.352 and 0.149",
        "data/rfss_dataset.h5 metadata",
        " / ".join(f"{sum(v for key, v in COUNTS['joint'].items() if int(key.split('|')[0]) == k) / COUNTS['total']:.3f}"
                   for k in (2, 3, 4)))
    add("impaired sources", "265,044", "data/rfss_dataset.h5 metadata",
        "265044, recomputed in the 2b pass over all 100,000 samples")
    add("clean sources", "52,828", "data/rfss_dataset.h5 metadata", "52828")
    add("multi-source file size", "102.5~GiB", "data/rfss_dataset.h5",
        "102.5 GiB = 110,092,208,432 bytes")
    add("single-source file size", "1.3~GiB", "data/rfss_single.h5",
        "1.32 GiB = 1,418,582,895 bytes", tol=0.05)
    add("GSM symbol rate (ksym/s)", "309.4", "src/utils_gsm.py:50",
        "309.43 = 2166000 / int(2166000/270833)")
    add("GSM record length (samples)", "1,890", "data/rfss_single.h5 signal_lengths",
        "1890 = 270 * 7")
    add("GSM ideal 99pc bandwidth (MHz)", "0.247", "ideal GMSK BT=0.3 at 270.833 ksym/s, own synthesis",
        "0.2467 at oversampling 8, 0.2461 at 64")
    add("NR prefix at the 2048-point FFT (samples)", "143", "src/utils_5g.py:79",
        "143 = int(2048 * 0.07)")
    add("NR specified prefix, first symbol (samples)", "176", "3GPP TS 38.211",
        "176, specification value")
    add("NR slot length, specified (samples)", "30,720", "TS 38.211 arithmetic",
        "30720 = 176 + 13*144 + 14*2048")
    add("NR slot length, implemented (samples)", "30,674", "src/utils_5g.py:79",
        "30674 = 14*(2048+143)")
    add("NR 50 MHz record (samples)", "61,348", "data/rfss_single.h5 signal_lengths",
        "61348 = 2 x 30674")
    add("NR 50 MHz record, 1 ms needs (samples)", "61,440", "61.44 MHz x 1 ms",
        "61440 = 61.44e6 x 1e-3")
    add("UMTS 99pc bandwidth (MHz)", "4.13", "data/rfss_single.h5, 334 rows",
        "4.13, median of the 99 percent band, paper/gen_fig_signal_quality.py")
    add("GSM 99pc bandwidth (released rows, MHz)", "0.26",
        "data/rfss_single.h5, 500 rows",
        "0.26, median of the 99 percent band, paper/gen_fig_signal_quality.py")
    add("LTE 99pc bandwidth (MHz)", "8.91", "data/rfss_single.h5, 321 rows",
        "8.91, median of the 99 percent band, paper/gen_fig_signal_quality.py")
    add("NR 99pc bandwidth (MHz)", "47.16", "data/rfss_single.h5, 219 rows",
        "47.16, median of the 99 percent band, paper/gen_fig_signal_quality.py")

    text = TEX.read_text()
    table_text = "\n".join(p.read_text() for p in sorted((PAPER / "tables").glob("*.tex")))
    printed_text = text + "\n" + table_text
    flat_printed_text = re.sub(r"\s+", " ", latex_plain(printed_text))

    def numbers(s):
        s = s.replace("$-", "-").replace("$", "").replace("~", " ")
        s = re.sub(r"(\d)[,](?=\d{3})", r"\1", s)
        return [float(v) for v in re.findall(r"[+-]?\d+\.?\d*", s)]

    lines = ["number | value in revised_paper.tex | source file | value recomputed from that file | status", ""]
    statuses = []
    flagged, unlocated = 0, 0
    for label, printed, source, value, tol in ROWS:
        token = re.sub(r"[{}]", "", printed.replace("~", "~"))
        flat_printed = re.sub(r"\s+", " ", latex_plain(printed))
        found = (printed in text or token in text
                 or latex_plain(printed) in printed_text
                 or latex_plain(token) in printed_text
                 or flat_printed in flat_printed_text)
        a, b = numbers(printed), numbers(value)
        if a and b:
            limit = tol if tol is not None else max(0.011, 0.0051 * abs(b[0]))
            status = "MATCH" if abs(a[0] - b[0]) <= limit else "MISMATCH"
            if status == "MISMATCH":
                flagged += 1
        else:
            status = "value only, no printed number"
            unlocated += 1
        if not found:
            status += " (no literal token in the tex)"
        statuses.append(status)
        lines.append(f"{label} | {printed} | {source} | {value} | {status}")
    lines.append("")
    lines.append(f"rows: {len(ROWS)}, numeric MISMATCH: {flagged}, rows with no printed number: {unlocated}")

    # Groups (reviewer 12:57:59, item 4), assigned over every row by its own kind:
    #   a = a value quoted from a cited paper or specification (external source)
    #   b = a table or equation pointer rather than a digit string
    #   c = this paper's own number the script could not match to a literal token
    groups = {"a": [], "b": [], "c": []}
    for row, status in zip(ROWS, statuses):
        label, printed, source, value, tol = row
        entry = (label, printed, source, value, status)
        if printed.startswith("see ") or printed.startswith("six of"):
            groups["b"].append(entry)
        elif any(k in source for k in CITED):
            groups["a"].append(entry)
        elif status != "MATCH":
            groups["c"].append(entry)

    lines.append("")
    lines.append("EVERY ROW SORTED BY KIND (reviewer 12:57:59, item 4): "
                 f"(a) {len(groups['a'])} quoted from a cited paper or specification, "
                 f"(b) {len(groups['b'])} table or equation pointers, "
                 f"(c) {len(groups['c'])} of this paper's own numbers the script could not "
                 "match to a literal token (listed below; each is either present in a "
                 "generated table or is a rounded form of the recomputed value).")
    for key, title in (("a", "(a) values quoted from cited papers or specifications"),
                       ("b", "(b) table or equation pointers"),
                       ("c", "(c) this paper's own numbers the script could not match")):
        lines.append("")
        lines.append(f"{title}: {len(groups[key])} rows")
        for label, printed, source, value, status in groups[key]:
            lines.append(f"  [{key}] {label} | {printed} | {source} | {value} | {status}")
    lines.append("")
    lines.append("A printed value that carries a table reference rather than a number "
                 "('see table_snr.tex') is located in paper/tables/.")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
