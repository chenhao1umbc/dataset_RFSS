# Signal Mixing Scenarios for RFSS Dataset

## 1. Introduction

This document defines realistic multi-standard RF signal mixing scenarios based on 3GPP coexistence studies, spectrum sharing research, and practical deployment scenarios. The goal is to create a dataset that accurately represents real-world wireless coexistence challenges for source separation research.

## 2. Background from Literature

### 2.1 Dynamic Spectrum Sharing (DSS)

**3GPP Standards:**
- Release 15/16: LTE-NR coexistence in shared spectrum
- Frequency offset: 7.5 kHz between LTE and NR subcarrier grids
- Initial deployments target sub-1 GHz bands for coverage

**Key Findings:**
- Mixed numerology (15 kHz vs 30 kHz) causes inter-numerology interference
- DSS deployment shows ~25% throughput loss vs exclusive spectrum
- Near-far problem is significant in real deployments

### 2.2 Adjacent Channel Interference

**3GPP Technical Specifications:**
- ACLR (Adjacent Channel Leakage Ratio): 45 dB (BS), 30 dB (UE)
- ACIR (Adjacent Channel Interference Ratio): ~32 dB for 5 MHz BW
- Interference-to-Noise ratio: -6 dB standard coexistence criteria

**Sources:**
- Co-channel interference (CCI): Same frequency, highest separation difficulty
- Adjacent channel interference (ACI): Nearby frequencies with ACIR constraints

### 2.3 Multi-Standard Coexistence

**Historical deployments:**
- GSM (900/1800 MHz) + UMTS (2100 MHz) coexistence
- UMTS + LTE coexistence in same bands
- LTE + 5G NR dynamic spectrum sharing

**Interference characteristics:**
- UMTS-GSM: Blocking and receiver sensitivity issues
- LTE-NR: 7.5 kHz frequency offset, mixed numerology
- All combinations: Near-far problem with varying SIR

## 3. Realistic Mixing Scenarios

### 3.1 Co-Channel Mixing (Hardest Case)

All source signals occupy the same frequency band (baseband mixing).

**Characteristics:**
- Maximum spectral overlap
- Separation relies purely on time-domain/waveform differences
- Most challenging for blind source separation
- Represents worst-case interference scenario

**Signal-to-Interference Ratios (SIR):**
- Equal power: SIR = 0 dB for all sources
- Near-far scenarios: SIR range -20 to +20 dB
- Realistic: Follow log-normal distribution with std 6-8 dB

### 3.2 Adjacent-Channel Mixing

Source signals at different frequency offsets with realistic ACIR.

**Frequency Offsets:**
- Based on actual channel bandwidths:
  - GSM: 200 kHz channel spacing
  - UMTS: 5 MHz channel bandwidth
  - LTE: 5/10/20 MHz channel bandwidth
  - 5G NR: Flexible bandwidth (5/10/20/40/100 MHz)

**ACIR Modeling:**
- Adjacent channel: 32-45 dB isolation
- Separation depends on filter quality and frequency offset
- LTE-NR specific: 7.5 kHz subcarrier offset

### 3.3 MIMO Spatial Mixing

Multi-antenna receivers with spatially correlated channels.

**Spatial Configurations:**
- 2x2 MIMO: Minimum spatial diversity
- 4x4 MIMO: Typical LTE/5G deployment
- 8x8 MIMO: Advanced 5G massive MIMO

**Spatial Correlation:**
- Weichselberger model for realistic correlation
- Correlation depends on: antenna spacing, angular spread, wavelength
- Low correlation (< 0.3): Well-separated antennas
- High correlation (> 0.7): Compact antenna arrays

## 4. Standard Combinations

### 4.1 Two-Source Scenarios

**All pairwise combinations:**
1. GSM + UMTS (2G + 3G legacy coexistence)
2. GSM + LTE (2G + 4G coexistence)
3. GSM + 5G NR (2G + 5G coexistence)
4. UMTS + LTE (3G + 4G common deployment)
5. UMTS + 5G NR (3G + 5G coexistence)
6. LTE + 5G NR (4G + 5G DSS, most relevant modern scenario)

**Parameters per combination:**
- Co-channel or adjacent-channel
- Power ratios: Equal (0 dB), Near (±3-10 dB), Far (±10-20 dB)
- MIMO: SISO, 2x2, 4x4

### 4.2 Three-Source Scenarios

**Relevant combinations:**
1. GSM + UMTS + LTE (2G/3G/4G legacy deployment)
2. UMTS + LTE + 5G NR (3G/4G/5G modern deployment)
3. GSM + LTE + 5G NR (2G/4G/5G)
4. GSM + UMTS + 5G NR (2G/3G/5G)

**Mixing characteristics:**
- Mixed co-channel and adjacent-channel
- Power ratios: One dominant + two weak, or balanced
- MIMO: Increased spatial separation capability

### 4.3 Four-Source Scenario

**Full spectrum:**
- GSM + UMTS + LTE + 5G NR (all standards coexisting)

**Characteristics:**
- Highest complexity
- Realistic for wideband cognitive radio scenarios
- Tests maximum separation capability
- Power distribution: Log-normal with realistic spread

## 5. Channel Effects Per Source

Each source signal undergoes independent channel effects before mixing:

### 5.1 Per-Source Channel Application

**Channel models (from Phase 1.2):**
- TDL-A/B/C (NLOS scenarios)
- TDL-D/E (LOS scenarios)
- Different models for different sources

**Hardware impairments:**
- CFO: Independent carrier frequency offset per source (±0.05-5 ppm)
- SFO: Independent sampling frequency offset
- I/Q imbalance: Different per transmitter (0.1-3 dB amplitude, 1-10 deg phase)
- DC offset: -40 to -30 dBc per source
- Phase noise: -90 to -110 dBc/Hz per source
- PA nonlinearity: Different back-off per source (3-9 dB)

**Rationale:**
- Realistic: Each transmitter has different impairments
- Challenging: Separators must handle heterogeneous distortions
- Complete: Covers full transmitter-channel-receiver chain

### 5.2 Timing Offsets

**Asynchronous arrival:**
- Source signals have random time delays
- Uniform distribution: 0 to signal_duration × 0.5
- Reflects real-world asynchronous transmission

**Edge handling:**
- Zero-padding for non-overlapping portions
- Ground truth tracks actual overlap regions
- Metadata preserves timing information

## 6. Ground Truth Management

### 6.1 Preservation Requirements

For each mixed signal, preserve:

**Source signals:**
- Original clean baseband signals (before channel)
- After individual channel application (before mixing)
- Time-aligned versions for evaluation

**Channel parameters per source:**
- TDL model name and parameters
- CFO, SFO, I/Q imbalance, DC offset, phase noise, PA values
- MIMO channel matrices (if applicable)

**Mixing parameters:**
- Power ratios (SIR values)
- Frequency offsets (adjacent-channel)
- Timing offsets
- MIMO spatial correlation parameters

### 6.2 Output Format

```python
mixing_output = {
    # Mixed signals
    'mixed_signal': torch.Tensor,  # Shape: (num_samples,) for SISO
    'mixed_signals_mimo': torch.Tensor,  # Shape: (num_rx, num_samples) for MIMO

    # Ground truth sources
    'source_signals_clean': List[torch.Tensor],  # Before channel
    'source_signals_channelized': List[torch.Tensor],  # After channel, before mixing
    'source_signals_aligned': List[torch.Tensor],  # Time-aligned for evaluation

    # Channel parameters per source
    'channel_params': List[dict],  # TDL, CFO, SFO, etc. for each source

    # Mixing parameters
    'mixing_params': {
        'num_sources': int,
        'mode': str,  # 'co-channel' or 'adjacent-channel'
        'power_ratios_db': List[float],  # SIR for each source
        'frequency_offsets_hz': List[float],  # 0 for co-channel
        'timing_offsets_samples': List[int],
        'sample_rate': float,
        'mimo_config': dict,  # If MIMO
    },

    # Metadata for reproducibility
    'metadata': {
        'scenario_name': str,  # e.g., 'LTE+5G_DSS_near-far'
        'standards': List[str],  # ['LTE', '5G NR']
        'generation_timestamp': str,
        'generator_version': str,
    }
}
```

## 7. Realistic Parameter Ranges

### 7.1 Power Ratios (SIR)

**Equal power scenarios:**
- All sources at 0 dB relative power
- SIR = 0 dB for all pairs

**Near-far scenarios:**
- Dominant source: 0 dB
- Weak sources: -20 to -3 dB
- Realistic distribution: Log-normal, std 6-8 dB

**Extreme near-far:**
- Strong: +10 to +20 dB
- Weak: -20 to -10 dB
- Tests robustness to power imbalance

### 7.2 Frequency Offsets (Adjacent-Channel)

**Based on channel bandwidths:**
- GSM: ±200 kHz, ±400 kHz (1-2 channels)
- UMTS: ±5 MHz, ±10 MHz (1-2 channels)
- LTE: ±5/10/20 MHz (depending on BW)
- 5G NR: ±10/20/40 MHz (depending on BW)

**LTE-NR specific:**
- 7.5 kHz subcarrier offset for DSS scenarios
- Mixed numerology: 15 kHz (LTE) vs 30 kHz (NR)

### 7.3 MIMO Correlation

**Low correlation (< 0.3):**
- Well-separated antennas (> λ/2 spacing)
- Wide angular spread
- Best case for spatial separation

**Medium correlation (0.3-0.7):**
- Typical urban deployment
- Moderate angular spread
- Realistic case

**High correlation (> 0.7):**
- Compact arrays
- Narrow angular spread
- Challenging case

## 8. Validation Criteria

### 8.1 Power Accuracy

After mixing, verify:
- Actual SIR matches target SIR (tolerance: ±0.5 dB)
- Total mixed power is correct
- Individual source powers preserved in ground truth

### 8.2 Frequency Accuracy

For adjacent-channel:
- Spectrum peaks at correct frequency offsets (tolerance: ±1% of offset)
- ACIR approximately matches theoretical values (±5 dB)
- No spectral leakage artifacts

### 8.3 Timing Accuracy

- Timing offsets applied correctly (±1 sample tolerance)
- Overlap regions identified correctly
- Edge handling produces valid signals

### 8.4 MIMO Properties

- Spatial correlation matches target (±0.1 tolerance)
- Per-antenna mixtures are independent
- Channel matrices have correct rank

## 9. Implementation References

### 9.1 3GPP Specifications

- TR 38.901: Channel models (TDL/CDL)
- TS 38.104: Base station RF requirements (CFO, ACLR)
- TS 38.101: UE RF requirements (CFO, ACIR)
- TS 36.211: LTE physical layer
- TS 38.211: NR physical layer
- TS 25.213: UMTS spreading and modulation

### 9.2 Research References

- Dynamic Spectrum Sharing: 3GPP Release 15/16
- LTE-NR Coexistence: MediaTek DSS White Paper
- MIMO Correlation: Weichselberger model (IEEE TWC 2006)
- Blind Source Separation: MIT RF Challenge dataset
- Adjacent Channel Interference: CEPT Report 40

## 10. Summary

This document defines comprehensive, realistic signal mixing scenarios covering:
- **2-source:** 6 combinations (all pairwise standards)
- **3-source:** 4 combinations (relevant triplets)
- **4-source:** 1 combination (all standards)
- **Mixing modes:** Co-channel, adjacent-channel, MIMO
- **Power scenarios:** Equal, near, far, extreme (SIR: -20 to +20 dB)
- **Channel effects:** Independent per source with full 3GPP compliance
- **Ground truth:** Complete preservation for evaluation

Total scenario space: ~100+ unique configurations for comprehensive dataset generation.
