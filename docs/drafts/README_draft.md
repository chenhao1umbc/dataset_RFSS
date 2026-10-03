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

**[TBD Builder]** one command per table in the paper, each writing a JSON file under `results/`
(for example baselines: `check/run_baselines.py`; deep models: `train_all.sh` then the evaluation script).
Every number in the paper must map to one of these files.

## Tests

```bash
uv run pytest
```

**[TBD Builder]** state the expected pass count.

## Citation

**[TBD-F]** BibTeX for the arXiv paper (authors Hao Chen and Dayuan Tan).
