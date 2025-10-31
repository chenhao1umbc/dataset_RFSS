# Dataset Parameter Space Definition

## 1. Signal Generation Parameters

### 1.1 GSM (2G) Parameters

**Bandwidth:**
- Fixed: 200 kHz per 3GPP TS 45.005

**Modulation:**
- Fixed: GMSK with BT=0.3 per 3GPP TS 45.004
- No modulation variations (constant-envelope modulation)

**Sample Rate:**
- 2.166 MHz (270.833 kbps × 8 samples/symbol)

**Variations:** 1 configuration (fixed parameters)

### 1.2 UMTS (3G) Parameters

**Bandwidth:**
- Fixed: 5 MHz per 3GPP TS 25.104

**Modulation:**
- W-CDMA with QPSK data modulation
- Spreading factor variations could be explored but start with SF=16

**Sample Rate:**
- 7.68 MHz (3.84 Mcps × 2 samples/chip)

**Variations:** 1 configuration (fixed parameters for now)

### 1.3 LTE (4G) Parameters

**Bandwidth Options:** 6 options per 3GPP TS 36.104
- 1.4 MHz (6 RBs, 1.92 Msps)
- 3 MHz (15 RBs, 3.84 Msps)
- 5 MHz (25 RBs, 7.68 Msps)
- 10 MHz (50 RBs, 15.36 Msps)
- 15 MHz (75 RBs, 23.04 Msps)
- 20 MHz (100 RBs, 30.72 Msps)

**Modulation Schemes:** 4 options per 3GPP TS 36.211
- QPSK (4-QAM)
- 16-QAM
- 64-QAM
- 256-QAM

**Bandwidth Distribution:**
- 10 MHz: 30% (most common)
- 20 MHz: 25% (high capacity)
- 5 MHz: 20% (medium coverage)
- 15 MHz: 10%
- 3 MHz: 10%
- 1.4 MHz: 5% (legacy)

**Modulation Distribution:**
- 64-QAM: 35% (typical high SNR)
- 16-QAM: 30% (medium SNR)
- QPSK: 20% (low SNR/coverage)
- 256-QAM: 15% (very high SNR, LTE-Advanced)

**Variations:** 6 × 4 = 24 configurations

### 1.4 5G NR Parameters

**Numerologies:** 2 options (FR1 + FR2 coverage)
- μ=1: 30 kHz SCS (most common FR1, sub-6 GHz)
- μ=3: 120 kHz SCS (mmWave FR2)

**Bandwidth Options per Numerology:**
- μ=1: 10, 20, 50, 100 MHz (4 options)
- μ=3: 50, 100, 200, 400 MHz (4 options)

**Modulation Schemes:** 5 options per 3GPP TS 38.211
- QPSK (4-QAM)
- 16-QAM
- 64-QAM
- 256-QAM
- 1024-QAM (5G exclusive, high SNR)

**Numerology Distribution:**
- μ=1 (30 kHz): 75% (most common deployment)
- μ=3 (120 kHz): 25% (mmWave scenarios)

**Bandwidth Distribution (within numerology):**
- Highest available: 40%
- Medium: 35%
- Lower: 25%

**Modulation Distribution:**
- 64-QAM: 30%
- 256-QAM: 25%
- 16-QAM: 20%
- QPSK: 15%
- 1024-QAM: 10% (advanced)

**Variations:** 8 numerology-bandwidth pairs × 5 modulations = 40 configurations

### 1.5 Total Single-Standard Configurations

- GSM: 1
- UMTS: 1
- LTE: 24
- 5G NR: 40
- **Total: 66 single-standard configurations**

## 2. Channel Model Parameters

### 2.1 3GPP TDL Channel Models

**Models Available:** 5 options per 3GPP TR 38.901
- TDL-A: NLOS, low delay spread (23 taps)
- TDL-B: NLOS, medium delay spread (23 taps)
- TDL-C: NLOS, high delay spread (24 taps)
- TDL-D: LOS, low delay spread (13 taps, K=13.3 dB)
- TDL-E: LOS, high delay spread (14 taps, K=22 dB)

**Distribution:**
- TDL-A: 25% (typical urban NLOS)
- TDL-B: 20% (moderate delay spread)
- TDL-C: 15% (high delay spread, challenging)
- TDL-D: 20% (suburban LOS)
- TDL-E: 20% (rural LOS with delay)

### 2.2 Doppler Frequency (Mobility)

**Ranges:** Based on vehicle speeds and carrier frequencies

**Low Mobility (pedestrian, 3-5 km/h):**
- 2 GHz carrier: 5-10 Hz Doppler
- Range: 0-10 Hz
- Proportion: 30%

**Medium Mobility (vehicular urban, 30-60 km/h):**
- 2 GHz carrier: 50-120 Hz Doppler
- Range: 30-120 Hz
- Proportion: 40%

**High Mobility (highway, 100-120 km/h):**
- 2 GHz carrier: 200-250 Hz Doppler
- Range: 150-300 Hz
- Proportion: 20%

**Very High Mobility (high-speed rail, 250-350 km/h):**
- 2 GHz carrier: 500-700 Hz Doppler
- Range: 400-700 Hz
- Proportion: 10%

**Channel Variations:** 5 models × 4 mobility categories = 20 configurations

## 3. Hardware Impairment Parameters

### 3.1 Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO)

**Correlation:** CFO and SFO are correlated as both derive from the same clock oscillator error

**Oscillator Error Distribution:**
- Typical: ±0.05-0.1 ppm (40% of samples)
- Moderate: ±0.1-0.5 ppm (30% of samples)
- High: ±0.5-2.0 ppm (20% of samples)
- Stress: ±2.0-5.0 ppm (10% of samples)

**Sampling Strategy:**
1. Sample base oscillator error (ppm) from distribution above
2. Apply same error to both CFO and SFO
3. Random sign (positive or negative offset)

**Rationale per 3GPP TS 38.104, TS 38.101:**
- UE frequency error: ±0.05-0.1 ppm (typical)
- BS frequency error: ±0.05 ppm (tight tolerance)
- Combined worst case: ±0.15 ppm
- Clock drives both carrier and sampling frequencies

### 3.3 I/Q Imbalance

**Amplitude Imbalance per 3GPP TS 36.101:**
- Typical: 0.1-0.5 dB
- Moderate: 0.5-1.5 dB
- High: 1.5-3.0 dB
- Range: 0.1-3.0 dB

**Phase Imbalance:**
- Typical: 1-3 degrees
- Moderate: 3-6 degrees
- High: 6-10 degrees
- Range: 1-10 degrees

**Distribution:**
- Low (0.1-0.5 dB, 1-3 deg): 50%
- Medium (0.5-1.5 dB, 3-6 deg): 30%
- High (1.5-3.0 dB, 6-10 deg): 20%

### 3.4 DC Offset

**Ranges (LO leakage):**
- Typical: -40 to -35 dBc
- Moderate: -35 to -32 dBc
- High: -32 to -30 dBc

**Distribution:**
- -40 to -35 dBc: 50%
- -35 to -32 dBc: 30%
- -32 to -30 dBc: 20%

### 3.5 Phase Noise

**Ranges per 3GPP TS 25.102:**
- Good oscillator: -110 to -105 dBc/Hz @ 1 MHz offset
- Typical: -105 to -100 dBc/Hz
- Poor: -100 to -90 dBc/Hz

**Distribution:**
- -110 to -105 dBc/Hz: 30%
- -105 to -100 dBc/Hz: 50%
- -100 to -90 dBc/Hz: 20%

### 3.6 PA Nonlinearity (Rapp Model)

**Back-off Ranges:**
- High linearity (PA backed off): 7-9 dB
- Medium: 5-7 dB
- Low (PA near saturation): 3-5 dB

**Smoothness Parameter (p):**
- Typical: 2-4 (smooth transition)

**Distribution:**
- High linearity: 40%
- Medium: 35%
- Low linearity: 25%

### 3.7 Impairment Application Strategy

**Clean Signals:** 20% (no impairments, baseline)

**Single Impairment:** 30% (one impairment at a time)
- CFO only: 6%
- SFO only: 6%
- I/Q imbalance only: 6%
- DC offset only: 4%
- Phase noise only: 4%
- PA nonlinearity only: 4%

**Multiple Impairments:** 50% (realistic combination)
- CFO + I/Q + DC: 15%
- CFO + SFO + I/Q: 12%
- All impairments: 10%
- Random 3-4 impairments: 13%

## 4. SNR Ranges

**Training SNR Distribution:**
- Very Low SNR (-10 to 0 dB): 15% (challenging)
- Low SNR (0 to 10 dB): 25% (typical cell edge)
- Medium SNR (10 to 20 dB): 35% (typical operation)
- High SNR (20 to 30 dB): 20% (near base station)
- Very High SNR (30 to 40 dB): 5% (ideal conditions)

**Rationale:** Focus on 0-30 dB range (80% of samples) where deep learning can provide value

## 5. Mixed Signal Scenarios

### 5.1 Source Count Distribution

- Single-source (1): 30% (baseline, for comparison)
- 2-source: 35% (most common coexistence)
- 3-source: 25% (realistic multi-RAT)
- 4-source: 10% (full spectrum GSM+UMTS+LTE+5G)

### 5.2 Mixing Combinations

**2-Source Scenarios:** 10 combinations
- GSM + UMTS
- GSM + LTE
- GSM + 5G NR
- UMTS + LTE
- UMTS + 5G NR
- LTE + 5G NR (DSS scenarios)
- GSM + GSM (different bursts)
- LTE + LTE (different UEs)
- 5G + 5G (different UEs)
- UMTS + UMTS (different codes)

**3-Source Scenarios:** 4 combinations
- GSM + UMTS + LTE
- UMTS + LTE + 5G NR
- GSM + LTE + 5G NR
- GSM + UMTS + 5G NR

**4-Source Scenarios:** 1 combination
- GSM + UMTS + LTE + 5G NR (full multi-RAT)

**Total mixing scenarios:** 15 unique combinations

### 5.3 Mixing Mode Distribution

**Co-Channel Mixing:** 40%
- All sources at baseband (frequency offset = 0)
- Hardest case for separation (no frequency diversity)
- Requires blind source separation

**Adjacent-Channel Mixing:** 60%
- Sources at different frequencies
- Frequency offsets: realistic spectrum allocation
- Models ACIR ~32 dB per 3GPP coexistence studies

### 5.4 Power Ratio Ranges (SIR)

**Equal Power:** 20%
- All sources within ±2 dB

**Moderate Imbalance:** 40%
- SIR: -10 to +10 dB between sources

**Near-Far Effect:** 30%
- SIR: -20 to -10 dB or +10 to +20 dB

**Extreme Near-Far:** 10%
- SIR: beyond ±20 dB (very challenging)

**Rationale:** Based on 3GPP coexistence scenarios where near-far can reach ±20 dB

## 6. MIMO Configuration

### 6.1 Antenna Configurations

**SISO (1x1):** 50%
- Single antenna, baseline

**2x2 MIMO:** 30%
- Most common MIMO deployment

**4x4 MIMO:** 20%
- Advanced MIMO, LTE-Advanced/5G

**8x8 MIMO:** Not included initially
- Massive MIMO for future extension

### 6.2 Spatial Correlation

**Low Correlation:** 40%
- Antenna spacing > 10λ
- Independent fading

**Medium Correlation:** 40%
- Antenna spacing 4-10λ
- Moderate correlation

**High Correlation:** 20%
- Antenna spacing < 4λ
- Strong correlation (challenging)

## 7. Parameter Space Size Calculation

### 7.1 Single-Standard Samples

**Per Standard:**
- Signal config: varies per standard (1-70)
- Channel model: 5 options
- Doppler: 4 categories (sampled continuously)
- Impairments: 3 categories (clean, single, multiple)
- SNR: 5 ranges (sampled continuously)
- MIMO: 3 configs

**Approximate combinations per standard:**
- GSM: 1 × 5 × 4 × 3 × 5 × 3 = 900
- UMTS: 1 × 5 × 4 × 3 × 5 × 3 = 900
- LTE: 24 × 5 × 4 × 3 × 5 × 3 = 21,600
- 5G NR: 70 × 5 × 4 × 3 × 5 × 3 = 63,000

**Total single-standard space:** ~86,400 unique combinations

### 7.2 Mixed-Signal Samples

**Per mixing scenario:**
- Scenario type: 15 combinations
- Mixing mode: 2 (co-channel, adjacent-channel)
- Power ratio: 4 categories
- Per-source variations: channel × impairments × MIMO

**Conservative estimate:**
- 15 scenarios × 2 modes × 4 power ratios = 120 base configurations
- Each with per-source variations: 120 × (5 channels)^n_sources
- For 2-source: 120 × 25 = 3,000
- For 3-source: 120 × 125 = 15,000
- For 4-source: 120 × 625 = 75,000

**Total mixed-signal space:** ~93,000 unique combinations

### 7.3 Overall Parameter Space

**Total theoretical combinations:** ~180,000

**Practical dataset size:** Trade-off between coverage and training efficiency
- Demonstration: 500 samples (verify all corners)
- Small: 5,000-10,000 samples (quick experiments)
- Medium: 50,000 samples (good coverage)
- Large: 100,000-200,000 samples (comprehensive)

## 8. Sampling Strategy

### 8.1 Parameter Sampling

**Categorical Parameters:** Sample according to defined distributions
- Use weighted random sampling
- Ensure minimum representation of rare categories

**Continuous Parameters:** Sample uniformly or normally within ranges
- SNR: Uniform within each range category
- Doppler: Uniform within each mobility category
- Impairment levels: Uniform within severity ranges
- Power ratios: Uniform within SIR categories

### 8.2 Coverage Requirements

**Minimum samples per category:**
- Each signal standard: ≥100 samples
- Each TDL model: ≥100 samples
- Each mixing scenario: ≥50 samples
- Each MIMO config: ≥100 samples

**Rare scenario oversampling:**
- Ensure 4-source scenarios have adequate representation
- Ensure extreme SNR cases are included
- Ensure high impairment cases are tested

## 9. Reproducibility

### 9.1 Random Seed Management

**Master seed:** Fixed for entire dataset generation
- Enables reproducibility of entire dataset

**Per-sample seeds:** Derived from master seed + sample index
- Allows regeneration of individual samples
- Enables parallel generation

### 9.2 Version Tracking

**Code version:** Git commit hash
**Parameter version:** Hash of this parameter file
**Generation date:** ISO 8601 timestamp
**Generator environment:** Python version, PyTorch version, CUDA version

## 10. Metadata Schema

### 10.1 Required Metadata per Sample

**Sample Identification:**
- sample_id: Unique identifier
- split: 'train', 'val', or 'test'
- seed: Random seed used for generation

**Signal Parameters:**
- standard: 'GSM', 'UMTS', 'LTE', '5G_NR'
- For each source in mixed signals:
  - bandwidth_mhz
  - modulation_scheme
  - num_subframes/bursts
  - sample_rate

**Channel Parameters:**
- tdl_model: 'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'
- doppler_hz: Actual Doppler frequency
- delay_taps: Channel tap delays (ns)
- power_taps: Channel tap powers (dB)

**Impairment Parameters:**
- cfo_ppm: Carrier frequency offset
- sfo_ppm: Sampling frequency offset
- iq_amp_db: I/Q amplitude imbalance
- iq_phase_deg: I/Q phase imbalance
- dc_offset_dbc: DC offset level
- phase_noise_dbc_hz: Phase noise level
- pa_backoff_db: PA back-off

**Mixing Parameters (if applicable):**
- num_sources: 1, 2, 3, or 4
- mixing_mode: 'co-channel' or 'adjacent-channel'
- power_ratios_db: Power of each source
- frequency_offsets_hz: Frequency offset of each source
- timing_offsets_samples: Timing offset of each source

**MIMO Parameters:**
- num_tx: Number of transmit antennas
- num_rx: Number of receive antennas
- spatial_correlation: Correlation coefficient

**Quality Metrics:**
- snr_db: Signal-to-noise ratio
- papr_db: Peak-to-average power ratio
- bandwidth_hz: Measured occupied bandwidth

**Ground Truth:**
- source_signals: Clean separated source signals (before mixing)
- channel_responses: Channel impulse responses applied
- mixed_signal: Final mixed received signal
- noise: Added noise signal

## 11. Implementation Notes

### 11.1 Generation Pipeline

1. Sample parameters according to distributions
2. Generate clean source signals
3. Apply independent channel per source
4. Apply independent impairments per source
5. Mix signals (co-channel or adjacent-channel)
6. Add AWGN to target SNR
7. Store signals + metadata
8. Validate quality metrics

### 11.2 Storage Considerations

**Signal precision:** float32 complex
- 8 bytes per complex sample

**Estimated size per sample (1 ms duration):**
- Single source (1 ms @ 30.72 Msps): 30.7k samples × 8 bytes = 0.25 MB
- 4-source mixed with ground truth: ~1.0 MB uncompressed
- With compression (HDF5 gzip): ~0.3-0.5 MB

**Dataset size estimates:**
- 10k samples: 3-5 GB
- 50k samples: 15-25 GB
- 100k samples: 30-50 GB

**Final decision: 100k samples @ 1ms = ~50 GB (well under 700 GB limit)**

### 11.3 Validation Checks

**Per sample validation:**
- Power within expected range
- Bandwidth matches specification
- Ground truth signals preserved
- Metadata completeness

**Dataset-wide validation:**
- Parameter distribution matches target
- No missing categories
- SNR distribution correct
- Mixing ratio distribution correct

## 12. Final Design Decisions

1. **Dataset size:** 100,000 samples (provides ~56% parameter space coverage)
2. **Signal duration:** 1 ms (one LTE subframe, captures complete structure, 50 GB total)
3. **Test set strategy:** Random split (70/15/15: train/val/test with 100k ensures statistical coverage)
4. **Impairment correlation:** Correlated CFO and SFO (both derived from same clock oscillator error, more realistic)
5. **5G numerology focus:** μ=1 (30 kHz FR1) and μ=3 (120 kHz mmWave) only (~30 configurations)
6. **mmWave scenarios:** Included via μ=3 (120 kHz)
7. **Storage format:** HDF5 with gzip compression (version-agnostic, 60-70% compression, industry standard)

## 13. Summary

**Final Dataset Specifications:**
- **Size:** 100,000 samples
- **Signal duration:** 1 ms (one LTE subframe)
- **Storage:** ~50 GB (HDF5 with gzip compression)
- **Split:** 70,000 train / 15,000 val / 15,000 test (random)
- **Standards:** GSM, UMTS, LTE, 5G NR (μ=1, μ=3)
- **Configurations:** 66 single-standard + 15 mixing scenarios
- **Parameter space coverage:** ~56% of theoretical combinations

**Key design principles:**
- Realistic distributions based on 3GPP deployment scenarios
- Correlated impairments (CFO/SFO from same oscillator)
- Adequate coverage of rare but important cases
- Full reproducibility through seed management
- Comprehensive metadata for analysis and debugging
- Balanced representation across all dimensions
- Version-agnostic HDF5 storage format
