# RFSS Dataset Format Specification

## 1. Overview

The RFSS dataset consists of two HDF5 files:

| File | Samples | Size | Purpose |
|------|---------|------|---------|
| `data/rfss_dataset.h5` | 100,000 | ~103 GB | Multi-source mixtures (2–4 sources) |
| `data/rfss_single.h5` | 4,000 | ~1.3 GB | Single-standard samples (1,000/standard) |

Both files share the same internal layout and are produced by `src/generate_dataset.py`
using `src/utils_dataset.DatasetWriter`.

---

## 2. HDF5 Layout

```
rfss_dataset.h5 (or rfss_single.h5)
├── Attributes
│   ├── version              str    "1.0"
│   ├── creation_time        str    ISO-8601 UTC timestamp
│   ├── max_samples          int    pre-allocated capacity
│   ├── actual_samples       int    number of samples written
│   ├── signal_duration_ms   float  1.0
│   └── format               str    "complex64"
│
├── mixed_signals     (N, 122880)    complex64, gzip-6
├── source_signals    (N, 4, 122880) complex64, gzip-6
├── signal_lengths    (N,)           int32
└── metadata          (N,)           UTF-8 JSON strings
```

`N` = `actual_samples` at read time.

### 2.1 Chunking

| Dataset | Chunk shape | Rationale |
|---------|-------------|-----------|
| `mixed_signals` | `(1, 122880)` | One full sample per chunk; enables random access |
| `source_signals` | `(1, 1, 122880)` | One source per chunk |
| `signal_lengths` | contiguous | Small integer array |
| `metadata` | contiguous | Small string array |

---

## 3. Signal Layout

### 3.1 Mixed signal

`mixed_signals[i, :signal_lengths[i]]` — the received observation: a sum of all
source signals after channel + impairments, with AWGN added to achieve the target
SNR.  Samples beyond `signal_lengths[i]` are zero-padded.

### 3.2 Source signals

`source_signals[i, j, :signal_lengths[i]]` — ground-truth source `j` for sample `i`.

**Important:** sources are stored at the **mixed-signal length** (max rate,
122,880 samples at 122.88 MHz).  The sources were originally generated at their
native 3GPP sample rates (see Section 4) and resampled to the common rate by
`SignalMixer` before storage.  AWGN is not included in the stored sources —
they represent the clean contribution of each source to the mixture.

Slots `j = num_sources` … `3` are zero-filled for samples with fewer than 4
sources. Non-zero detection: `torch.any(source_signals[i, j, :] != 0)`.

For `rfss_single.h5`, only slot `j=0` is non-zero (`num_sources == 1`).

---

## 4. Native Sample Rates

Each standard operates at its canonical 3GPP sample rate:

| Standard | Native sample rate | Notes |
|----------|--------------------|-------|
| GSM | 2.166 MHz | 270.833 kbps × 8 samp/sym |
| UMTS | 7.68 MHz | 3.84 Mcps × 2 samp/chip |
| LTE | 1.92–30.72 MHz | 6 bandwidth options |
| 5G NR | 15.36–122.88 MHz | 4 numerology × bandwidth options |

The `SignalMixer` resamples all sources to the highest sample rate present in a
given mixture before summing.  The stored `mixed_signals` and `source_signals`
are both at this common (maximum) rate.

The common rate for a given sample is:
```
max(source['signal_params']['sample_rate'] for source in metadata['sources'])
```

Training code that computes a separation loss against stored sources does not
need to resample — both mixed and sources are at the same rate.

---

## 5. Metadata JSON Schema

Each `metadata[i]` is a UTF-8 JSON string with the following structure:

```json
{
  "sample_id":   int,          // global index (0-based)
  "seed":        int,          // RNG seed used (master_seed + sample_id)
  "num_sources": int,          // 1–4
  "sources": [
    {
      "standard": "GSM"|"UMTS"|"LTE"|"5G_NR",
      "signal_params": {
        "sample_rate":       float,   // native sample rate (Hz)
        "bandwidth_mhz":     float,   // signal bandwidth (MHz)
        "modulation":        str,     // modulation scheme
        ...                           // standard-specific fields
      },
      "channel_params": {
        "tdl_model":   "TDL-A"|"TDL-B"|"TDL-C"|"TDL-D"|"TDL-E",
        "doppler_hz":  float          // max Doppler frequency (Hz)
      },
      "impairment_params": {
        "cfo_ppm":           float,   // carrier frequency offset (ppm)
        "sfo_ppm":           float,   // sampling frequency offset (ppm)
        "iq_amp_db":         float,   // I/Q amplitude imbalance (dB)
        "iq_phase_deg":      float,   // I/Q phase imbalance (deg)
        "dc_offset_dbc":     float,   // DC offset relative to carrier (dBc)
        "phase_noise_dbc_hz":float,   // phase noise PSD floor (dBc/Hz)
        "pa_backoff_db":     float    // PA input backoff (dB)
      }
    }
  ],
  "mixing_params": {
    "num_sources":           int,
    "mixing_mode":           "co-channel"|"adjacent-channel",
    "power_ratios_db":       [float, ...],  // per-source power offset (dB)
    "frequency_offsets_hz":  [float, ...]   // per-source freq offset (Hz)
  },
  "snr_db":      float,        // target SNR for AWGN addition (dB)
  "mimo_config": {
    "num_tx":              int,    // 1, 2, or 4
    "num_rx":              int,    // 1, 2, or 4
    "spatial_correlation": float  // 0.0–0.9
  },
  "generation_time": str       // ISO-8601 UTC timestamp
}
```

For `rfss_single.h5`, `num_sources` is always 1 and `mixing_params` contains
`mixing_mode = "single"`.

---

## 6. Train / Val / Test Split

Splits are applied at **load time** by `RFSSDataset` using sequential index
boundaries (no shuffling during split assignment):

| Split | Fraction | Multi-source samples | Single-source samples |
|-------|----------|---------------------|----------------------|
| train | 70% | 70,000 | 2,800 |
| val | 15% | 15,000 | 600 |
| test | 15% | 15,000 | 600 |

The HDF5 file is not physically partitioned.  The split is determined by index
range: `[0, 70000)` → train, `[70000, 85000)` → val, `[85000, 100000)` → test.

---

## 7. PyTorch Interface

```python
from src.utils_dataset import RFSSDataset, create_dataloader

# Single item
ds = RFSSDataset('data/rfss_dataset.h5', split='train')
item = ds[0]
# item['mixed_signal']:   torch.complex64, shape (L,)
# item['source_signals']: list of torch.complex64, each shape (L,), len == num_sources
# item['metadata']:       dict (see Section 5)
# item['sample_id']:      int

# DataLoader with padding collation
loader = create_dataloader(
    'data/rfss_dataset.h5', split='train',
    batch_size=32, shuffle=True, num_workers=4
)
batch = next(iter(loader))
# batch['mixed_signals']:   torch.complex64, (B, max_L)
# batch['source_signals']:  torch.complex64, (B, max_sources, max_L)
# batch['signal_lengths']:  torch.int32,     (B,)
# batch['metadata']:        list of dict, length B
# batch['sample_ids']:      list of int, length B
```

---

## 8. Parameter Distributions (Multi-source Dataset)

Verified by coverage analysis over 20,000 samples (every 5th record):

| Parameter | Value | Intended | Actual |
|-----------|-------|----------|--------|
| num_sources | 2 | 50.0% | 50.1% |
| num_sources | 3 | 35.0% | 34.9% |
| num_sources | 4 | 15.0% | 14.9% |
| mixing_mode | co-channel | 40.0% | 39.7% |
| mixing_mode | adjacent-channel | 60.0% | 60.3% |
| mimo_config | 1×1 | 50.0% | 50.2% |
| mimo_config | 2×2 | 30.0% | 30.1% |
| mimo_config | 4×4 | 20.0% | 19.7% |

SNR range: –10.0 to 40.0 dB (mean 12.3 dB).
All deviations < 0.5%, well within the 5% design tolerance.

---

## 9. Reproducibility

All samples are deterministically reproducible from seeds:

| Dataset | Seed formula |
|---------|-------------|
| Multi-source | `master_seed (42) + sample_id` |
| Single-source | `master_seed (42) + 2,000,000 + global_idx` |

The 2,000,000 offset prevents any seed collision between datasets.

To regenerate:
```bash
uv run python -m src.generate_dataset \
    --output data/rfss_dataset.h5 --mode multi --num-samples 100000 --seed 42

uv run python -m src.generate_dataset \
    --output data/rfss_single.h5 --mode single --num-samples-per-standard 1000 --seed 42
```
