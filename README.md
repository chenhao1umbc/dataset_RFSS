# RFSS: RF Signal Source Separation Dataset

RFSS is a dataset and generation framework for blind separation of simultaneous GSM, UMTS, LTE and 5G NR signals.
It combines 3GPP-based waveform generators, TDL fading channels (TR 38.901), hardware impairments, and
co-channel / adjacent-channel mixing, with per-source ground truth.

- Data: https://huggingface.co/datasets/Chrishao/rfss
- Paper: arXiv:2508.12106 (a corrected version replaces the earlier ones and supersedes arXiv:2604.00398)
- License: data CC BY-NC 4.0 (`LICENSE-DATA`), code PolyForm Noncommercial 1.0.0 (`LICENSE`). Free for everyone, no commercial use.

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
hf download Chrishao/rfss --repo-type dataset --include "data/*" --local-dir .
```

The full multi-source file is about 103 GiB; `data/rfss_single.h5` (1.3 GiB) is a small file for quick tests.

## Generate data yourself

```bash
uv run python src/generate_dataset.py --output data/rfss_dataset.h5 --mode multi --num-samples 100000 --seed 42
uv run python src/generate_dataset.py --output data/rfss_single.h5 --mode single --num-samples-per-standard 1000 --seed 42
```

The released files were generated in February 2026 (`data/generation_status.json`); rerunning does not promise the same bytes. Generating 100,000 samples takes many hours.

## Reproduce the benchmark

All methods are scored on the same test samples (indices 85,000-99,999) with the exact per-source references from
`src/utils_mixing.py:build_aligned_references`. The headline pass scores one random 7,680-sample window per sample (crop seed 0);
the first-window pass and crop seeds 1 and 2 are robustness passes.

```bash
# Trained models used in the paper (15 checkpoints) from the Hugging Face repo
hf download Chrishao/rfss --repo-type dataset --include "checkpoints/*" --local-dir ckpt_tmp && mv ckpt_tmp/checkpoints final

# Frozen test passes: 2-source (9 checkpoints, ICA, NMF, oracles) and 3-/4-source (6 checkpoints)
CROP_SEED=0 bash check/run_test_passes.sh 2
CROP_SEED=0 bash check/run_test_passes.sh 34
uv run python check/test_summary.py 2 0          # per-bin gains with bootstrap intervals and paired differences
uv run python check/test_summary.py 34 0

# Train the models again (10 epochs; the STFT-BLSTM on the CPU lane, the others on the MPS lane)
bash train_all.sh all 10

# Paper tables and numbers from the committed result files
uv run python paper/make_journal_assets.py
uv run python paper/icc/make_icc_assets.py
```

Every number in the papers maps to a JSON file in `check/`.

## Tests

```bash
uv run pytest
```

56 tests pass. `check/unit_test_mixing.py::test_build_aligned_references_matches_mixer` needs no data file.

## Citation

```bibtex
@article{chen2026rfss,
  title   = {{RFSS}: A Multi-Standard {RF} Signal Source Separation Dataset with 3GPP-Standardized Channel and Hardware Impairments},
  author  = {Chen, Hao and Jin, Rui and Tan, Dayuan},
  journal = {arXiv preprint arXiv:2508.12106},
  year    = {2026}
}
```

## License

Data: CC BY-NC 4.0 (`LICENSE-DATA`). Code: PolyForm Noncommercial 1.0.0 (`LICENSE`).
