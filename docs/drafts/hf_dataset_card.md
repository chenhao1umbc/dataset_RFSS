---
license: cc-by-4.0
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

DRAFT by Reviewer, 2026-10-03. Items marked **[TBD-A#]** wait on workstream A of `ACTION_PLAN.md`.
Do not publish until every TBD is resolved and the Builder has confirmed the numbers.

RFSS contains complex baseband mixtures of 2 to 4 simultaneous cellular signals (GSM, UMTS, LTE, 5G NR)
together with per-source reference waveforms and full generation metadata. It is intended for
blind, single-channel RF source separation research.

Paper: [arXiv:2508.12106] (replacement version, **[TBD-F]** link after v2 is live).
Code: https://github.com/chenhao1umbc/dataset_RFSS

## Files

| File | Samples | Size | Content |
|---|---|---|---|
| `data/rfss_dataset.h5` | 100,000 | about 103 GiB | 2-, 3- and 4-source mixtures |
| `data/rfss_single.h5` | 4,000 | about 1.3 GiB | single-source reference samples |
| **[TBD-C2]** `data/rfss_preview.h5` | about 1,000 | under 1 GB | quick-look subset |

## HDF5 layout (both files)

| Dataset | Shape (multi-source file) | Type | Meaning |
|---|---|---|---|
| `mixed_signals` | (100000, 122880) | complex64 | Received mixture, zero-padded after `signal_lengths[i]` |
| `source_signals` | (100000, 4, 122880) | complex64 | Per-source reference, unused slots are all zero |
| `signal_lengths` | (100000,) | int32 | Number of valid samples in the **mixture** of sample `i` |
| `metadata` | (100000,) | variable-length JSON string | Generation parameters (below) |

Root attributes: `actual_samples`, `max_samples`, `format` (`complex64`), `signal_duration_ms` (1.0),
`creation_time`, `version` (currently `1.0`). Arrays are gzip-compressed, one sample per chunk.

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
   **[TBD-A4]** Link the reference-builder function once it exists in the repo.
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
- Mixing mode: about 40% co-channel, 60% adjacent-channel.
- Channels: 3GPP TDL-A to TDL-E (TR 38.901), weighted selection, Jakes fading with Doppler up to 700 Hz.
- Noise: AWGN, SNR from -10 to 40 dB (observed mean about 12 dB).
- Hardware impairments per source: CFO, SFO, I/Q imbalance, DC offset, phase noise, PA nonlinearity (Rapp model);
  about 20% of sources clean, 30% one impairment, 50% several.
- Standards share in the scanned subset: 5G NR 0.38, LTE 0.37, GSM 0.12, UMTS 0.12.

## Splits

By sample index in `rfss_dataset.h5`: train 0-69,999; validation 70,000-84,999; test 85,000-99,999.
**[TBD-B]** Confirm that generation order is random with respect to scenario so contiguous index splits are balanced.

## Benchmark

**[TBD-B6]** Fill from committed result files only. Report PI-SI-SINR (Le Roux et al., 2019) for FastICA, NMF,
Conv-TasNet, DPRNN and CNN-LSTM, with confidence intervals and sample counts. Do not copy numbers from the April 2026 arXiv version.

## Known limitations

- Single-antenna (SISO) mixtures only; `mimo_config` in the metadata is not used.
- Downlink waveforms only; no uplink, NB-IoT, LTE-M or sidelink.
- Synthetic TDL channels, no measured channel data.
- Absolute separation scores of current methods are low; the dataset is hard for the baselines tested.
- Version history: `v1.0` (Feb 2026 files) and the corrected release **[TBD-A]**.

## License and citation

CC BY 4.0. **[TBD-F]** BibTeX for arXiv:2508.12106 (authors: Hao Chen, Dayuan Tan) once v2 is live.
