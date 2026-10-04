# RFSS: RF Signal Source Separation Dataset

DRAFT by Reviewer, 2026-10-03. Replace `README.md` after workstream A is settled. **[TBD]** marks open items.

RFSS is a dataset and generation framework for blind separation of simultaneous GSM, UMTS, LTE and 5G NR signals.
It combines 3GPP-based waveform generators, TDL fading channels (TR 38.901), hardware impairments, and
co-channel / adjacent-channel mixing, with per-source ground truth.

- Data: https://huggingface.co/datasets/Chrishao/rfss
- Paper: arXiv:2508.12106 **[TBD v2 link]**
- License: CC BY 4.0

## Repository layout

```
src/              generators (utils_gsm/umts/lte/5g), channel + impairments (utils_channel),
                  mixing (utils_mixing), dataset writer (utils_dataset), baselines, models, training
check/            unit tests, quality checks, baseline and breakdown evaluation scripts
paper/            LaTeX source of the paper and figure scripts
train_*.sh        training scripts used for the paper (Apple silicon, MPS)
```

## Install

```bash
git clone https://github.com/chenhao1umbc/dataset_RFSS.git
cd dataset_RFSS
uv sync            # or: pip install -e .
```

## Get the data

```bash
huggingface-cli download Chrishao/rfss --repo-type dataset --local-dir data
```

The full multi-source file is about 103 GiB. **[TBD-C2]** a small preview file is available for quick tests.

## Generate data yourself

**[TBD Builder]** exact command, seeds, and runtime for `src/generate_dataset.py`.

## Reproduce the benchmark

All methods are scored by `check/eval_all.py` on the same test samples (indices 85,000-99,999) and the same segment (the first 7,680 samples
of each signal, or the whole signal if shorter), with the exact per-source references from `src/utils_mixing.py:build_aligned_references`.

```bash
# 1. Train the nine models (3 architectures x 2/3/4 sources, 30 epochs each, about 60 h on an Apple M-series GPU)
nohup bash train_all.sh >> runs/train_all_v2.log 2>&1 &

# 2. Main table: input, noise-limited oracle, ICA, NMF and the trained models, with 95 percent bootstrap intervals
uv run python check/eval_all.py --dl conv_tasnet dprnn cnn_lstm            # writes check/eval_all_results.json

# 3. Robustness: random 7,680-sample windows instead of the first 7,680 samples
uv run python check/eval_all.py --dl conv_tasnet dprnn cnn_lstm --crop-seed 0   # also seeds 1 and 2

# Supplementary: ICA/NMF on the full-length signals
uv run python check/run_baselines.py                                         # writes check/baseline_results.json

# Data checks
uv run python check/verify_reference_alignment.py                            # reference rebuild vs stored mixtures
uv run python check/quality_check.py
```

Every number in the paper maps to one of these JSON files.

## Tests

```bash
uv run pytest
```

**[TBD Builder]** state the expected pass count. `check/unit_test_mixing.py::test_build_aligned_references_matches_mixer` needs no data file.

## Citation

**[TBD-F]** BibTeX for the arXiv paper (authors Hao Chen and Dayuan Tan).
