---
license: cc-by-nc-4.0
pretty_name: RFSS - RF Signal Source Separation Dataset
task_categories:
  - audio-to-audio
tags:
  - rf
  - wireless
  - source-separation
  - gsm
  - umts
  - lte
  - 5g-nr
  - hdf5
size_categories:
  - 100K<n<1M
---

# RFSS: Multi-Standard RF Signal Source Separation Dataset

RFSS contains complex baseband mixtures of 2 to 4 simultaneous cellular signals (GSM, UMTS, LTE, 5G NR)
together with per-source reference waveforms and full generation metadata. It is intended for
blind, single-channel RF source separation research.

Paper: arXiv:2508.12106. A corrected version replaces the earlier ones and supersedes arXiv:2604.00398, whose results and dataset description contained errors.
Code: https://github.com/chenhao1umbc/dataset_RFSS

## Files

| File | Samples | Size | Content |
|---|---|---|---|
| `data/rfss_dataset.h5` | 100,000 | about 103 GiB | 2-, 3- and 4-source mixtures |
| `data/rfss_single.h5` | 4,000 | about 1.3 GiB | single-source reference samples |
| `checkpoints/` | 15 files | about 0.8 GB | trained models of the paper (see Benchmark) |

## HDF5 layout (both files)

| Dataset | Shape (multi-source file) | Type | Meaning |
|---|---|---|---|
| `mixed_signals` | (100000, 122880) | complex64 | Received mixture, zero-padded after `signal_lengths[i]` |
| `source_signals` | (100000, 4, 122880) | complex64 | Per-source reference, unused slots are all zero |
| `signal_lengths` | (100000,) | int32 | Number of valid samples in the **mixture** of sample `i` |
| `metadata` | (100000,) | variable-length JSON string | Generation parameters (below) |

Root attributes: `actual_samples`, `max_samples`, `format` (`complex64`), `signal_duration_ms` (1.0),
`creation_time`, `version` (still `1.0`; the files of release v1.1 are the same bytes as v1.0). Arrays are gzip-compressed, one sample per chunk.

Read a sample:

```python
import h5py, json, numpy as np
with h5py.File("rfss_single.h5", "r") as f:
    i = 0
    L = int(f["signal_lengths"][i])
    mix = f["mixed_signals"][i, :L]
    srcs = f["source_signals"][i]          # (4, 122880), unused rows are zero
    meta = json.loads(f["metadata"][i])
```

Remote access without downloading everything: use HTTP range reads through `huggingface_hub`
or `fsspec`; each sample is one compressed chunk.

## Important properties of the data (read before using)

Updated 2026-10-03 after the Builder's forward-model check (`check/verify_reference_alignment.py`,
`check/reference_alignment_results.json`).

1. **Sample rates and lengths vary.** Each source is generated at its native 3GPP rate
   (GSM 2.166 MHz, UMTS 7.68 MHz, LTE 1.92-30.72 MHz, 5G NR 15.36-122.88 MHz) for about 1 ms.
   All sources of one sample are resampled to a common mixture rate equal to the highest source rate in that sample.
   `signal_lengths` therefore ranges from 1,890 to 122,880.
2. **Native length is not exactly 1 ms for every standard.** GSM sources hold 1,890 samples (nominal 2,166). 5G NR sources are slightly shorter than
   nominal (for example 122,696, 122,640, 61,348 and 30,660 samples; nominal 122,880, 61,440 and 30,720). LTE and UMTS are exact.
   Find a source's true length as the index of its last non-zero sample plus one. Do not assume `round(sample_rate * 0.001)`.
3. **What `source_signals` holds.** Each stored source is the waveform **after** its TDL channel and hardware impairments
   (CFO, SFO, I/Q imbalance, DC offset, phase noise, PA nonlinearity), at its native rate and zero-padded. It is **before** resampling to the mixture rate,
   before the adjacent-channel frequency shift, before power scaling, and before AWGN. It is not the clean transmitted waveform.
4. **Aligned reference (what the mixture actually contains).** The term each source contributes to `mixed_signals` is
   `normalize_power( freq_shift( pad( resample( source ) ) ), power_ratios_db[i] )`, where `freq_shift` is applied only in adjacent-channel mode using
   `mixing_params.frequency_offsets_hz`. Rebuilding the noiseless mixture this way with `SignalMixer` matched the stored mixture
   to within the expected AWGN level in 204 of 204 checked test samples. Use the aligned reference for scoring separation methods.
   Reference implementation: `build_aligned_references(source_block, meta, signal_len)` in `src/utils_mixing.py` of the GitHub repo. Independently checked by the Reviewer on 40 random test samples read from this repository (median residual gap 0.017 dB, maximum 0.167 dB versus the stored SNR).
5. **Do not score against the raw stored sources in adjacent-channel mixtures.** The unshifted reference scores a median of about -40 dB against the mixture,
   versus about -6 dB for the aligned reference (102 adjacent-channel test samples).
6. **Power scaling.** Sources are scaled by `mixing_params.power_ratios_db` (examples reach +/-25 dB), so the mixture power can be far above the stored reference power.
   SI-SINR is scale invariant.
7. **MIMO field is vestigial.** `metadata.mimo_config` reports `1x1`, `2x2` or `4x4`, but no MIMO processing is applied in the generator. All mixtures are single-stream (SISO).
   Ignore this field.

## Metadata schema

Per sample: `sample_id`, `seed`, `num_sources`, `snr_db`, `generation_time`, `mimo_config` (`num_tx`, `num_rx`, `spatial_correlation`),
`mixing_params` (`num_sources`, `mixing_mode` in {`co-channel`, `adjacent-channel`, `single`}, `power_ratios_db`, `frequency_offsets_hz`),
and `sources`: a list with, per source, `standard`, `signal_params` (bandwidth, modulation, sample_rate, numerology for NR),
`channel_params` (`tdl_model` in TDL-A..E, `doppler_hz`) and `impairment_params`
(`mode` in {clean, single, multiple}, `cfo_ppm`, `sfo_ppm`, `iq_amp_db`, `iq_phase_deg`, `dc_offset_dbc`, `phase_noise_dbc_hz`, `pa_backoff_db`).

## Generation summary (from the paper source and the 20,000-sample coverage check)

- Source counts: 2 / 3 / 4 sources with target weights 0.50 / 0.35 / 0.15 (realized about 0.50 / 0.35 / 0.15).
- Mixing mode: about 40% co-channel, 60% adjacent-channel. In adjacent-channel mode source k of N is shifted by (k - floor(N/2)) x 2 MHz for every standard and no adjacent-channel filtering is modelled, so most mixtures still contain overlapping bands (97% of the two-source adjacent-channel test samples). For the 1.8% of adjacent-channel samples whose mixture rate is below 4 MHz (GSM and narrow LTE sources only) the shift exceeds the Nyquist frequency and the shifted source wraps around in frequency; the aligned references apply the same wrap, so the labels stay exact.
- Channels: 3GPP TDL-A to TDL-E (TR 38.901), weighted selection, Jakes fading with Doppler up to 700 Hz.
- Noise: AWGN, SNR from -10 to 40 dB (observed mean about 12 dB).
- Hardware impairments per source: CFO, SFO, I/Q imbalance, DC offset, phase noise, PA nonlinearity (Rapp model);
  about 20% of sources clean, 30% one impairment, 50% several.
- Standards share in the scanned subset: 5G NR 0.38, LTE 0.37, GSM 0.12, UMTS 0.12.

## Splits

By sample index in `rfss_dataset.h5`: train 0-69,999; validation 70,000-84,999; test 85,000-99,999.
The source-count shares are the same in the three splits (2 / 3 / 4 sources: 50.2 / 35.5 / 14.3 % of the test split, 49.9 / 35.1 % for 2 / 3 sources in the training split).

## Benchmark

Phase-sensitive permutation-invariant SI-SINR (Le Roux et al., 2019) on the test split, one random 7,680-sample window per sample,
reported as the gain over the input mixture in dB. 2-source: mean of 3 training seeds, n = 7,526 test mixtures; 3- and 4-source: one seed,
n = 5,324 and 2,150. Model gains with 95 % bootstrap intervals, the seed spread and the paired differences are in the paper;
every number comes from the JSON files under `check/` of the GitHub repository (`eval_all_src*_crop0_frozen_results.json`).

| Method | 2 sources | 3 sources | 4 sources |
|---|---|---|---|
| STFT-BLSTM (7.4 M parameters) | +6.14 | +5.14 | +4.52 |
| DPRNN (1.1 M) | +5.85 | +5.03 | +3.44 |
| Conv-TasNet (2.5 M) | +5.61 | +4.77 | +3.54 |
| IRM oracle | +8.04 | +9.61 | +10.35 |
| Noise-limited oracle | +10.16 | +11.53 | +12.13 |
| NMF | -1.24 | -0.57 | -0.05 |
| FastICA | -13.60 | -11.09 | -9.49 |

The `checkpoints/` folder holds the 15 models scored in these tables, stored as `checkpoints/<run>/ckpt/<file>.pt`. Moving
`checkpoints` to `final` in a clone of the GitHub repository lets `check/run_test_passes.sh` score them.

## Known limitations

- Single-antenna (SISO) mixtures only; `mimo_config` in the metadata is not used.
- Downlink waveforms only; no uplink, NB-IoT, LTE-M or sidelink.
- Synthetic TDL channels, no measured channel data.
- Absolute separation scores of current methods are low; the dataset is hard for the baselines tested.
- Version history: see the next section.

## Version history

- `v1.0` (git tag, February 2026): the two HDF5 files, without a dataset card.
- `v1.1` (this card): the HDF5 files are byte-identical to v1.0 (SHA-256 of both files checked against the uploaded ones). What changed is the documentation and what is released with it:
  the reference definition above (what `source_signals` holds and how to rebuild the aligned reference), the adjacent-channel convention (a fixed 2 MHz spacing, no
  adjacent-channel filtering, so most adjacent-channel mixtures still overlap), the vestigial `mimo_config` field, the split indices, the evaluation protocol, the trained
  checkpoints and the licence. Earlier descriptions of the dataset (arXiv:2508.12106 v1 and arXiv:2604.00398) are wrong where they differ from this card.

## License and citation

Data: CC BY-NC 4.0, free for everyone to use and share with attribution, not for commercial use. Code (GitHub repository): PolyForm Noncommercial 1.0.0.

```bibtex
@article{chen2026rfss,
  title   = {{RFSS}: A Multi-Standard {RF} Signal Source Separation Dataset with 3GPP-Standardized Channel and Hardware Impairments},
  author  = {Chen, Hao and Jin, Rui and Tan, Dayuan},
  journal = {arXiv preprint arXiv:2508.12106},
  year    = {2026}
}
```
