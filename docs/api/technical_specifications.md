# RF Signal Source Separation Dataset - Technical Specifications

## Overview

This document provides detailed technical specifications for the RF Signal Source Separation (RFSS) dataset generation system, designed for research in wireless communication signal processing and source separation algorithms.

## Signal Generation Standards

### 2G GSM Signals
- **Modulation**: Gaussian Minimum Shift Keying (GMSK)
- **Bandwidth**: 200 kHz (channel spacing)
- **Symbol Rate**: 270.833 ksps
- **BT Product**: 0.3 (Gaussian filter)
- **Peak-to-Average Power Ratio (PAPR)**: ≤ 2 dB
- **Standards Compliance**: 3GPP TS 45.004

### 3G UMTS Signals  
- **Multiple Access**: Code Division Multiple Access (CDMA)
- **Bandwidth**: 5 MHz (W-CDMA)
- **Chip Rate**: 3.84 Mcps
- **Spreading Factors**: 4-512 (configurable)
- **Modulation**: QPSK, 16-QAM
- **PAPR Range**: 3-8 dB
- **Standards Compliance**: 3GPP TS 25.211, TS 25.212

### 4G LTE Signals
- **Multiple Access**: Orthogonal Frequency Division Multiple Access (OFDMA)
- **Bandwidth Options**: 1.4, 3, 5, 10, 15, 20 MHz
- **Subcarrier Spacing**: 15 kHz
- **Resource Blocks**: 6-100 (bandwidth dependent)
- **Modulation**: QPSK, 16-QAM, 64-QAM, 256-QAM
- **PAPR Range**: 6-15 dB
- **Standards Compliance**: 3GPP TS 36.211, TS 36.212

### 5G NR Signals
- **Multiple Access**: OFDMA (DL), SC-FDMA (UL)
- **Bandwidth Options**: 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100 MHz
- **Numerology**: μ = 0, 1, 2, 3 (subcarrier spacing: 15, 30, 60, 120 kHz)
- **Modulation**: QPSK, 16-QAM, 64-QAM, 256-QAM, 1024-QAM
- **PAPR Range**: 8-18 dB
- **Standards Compliance**: 3GPP TS 38.211, TS 38.212

## Channel Models

### Basic Channel Effects
1. **Additive White Gaussian Noise (AWGN)**
   - Configurable SNR: -20 to +40 dB
   - Complex Gaussian distribution

2. **Multipath Fading**
   - Exponential delay profile
   - Configurable RMS delay spread: 0.1-10 μs
   - Number of paths: 1-20

3. **Rayleigh Fading**
   - Jake's Doppler spectrum
   - Configurable Doppler frequency: 1-500 Hz
   - Non-line-of-sight (NLOS) conditions

4. **Rician Fading**
   - Configurable K-factor: 0-20 dB
   - Line-of-sight (LOS) component
   - Doppler spread: 1-200 Hz

### Propagation Environments
- **Urban Mobile**: High Doppler (200 Hz), multipath, Rayleigh fading
- **Rural LOS**: Low Doppler (50 Hz), Rician fading (K=10 dB)
- **Indoor NLOS**: Low Doppler (10 Hz), dense multipath, Rayleigh fading

## MIMO System Specifications

### Supported Configurations
- **Antenna Arrays**: 2×2, 4×4, 8×8, 16×16
- **Channel Models**: Independent Rayleigh, Correlated channels
- **Spatial Correlation**: Low (0.1), Medium (0.5), High (0.8)

### Processing Techniques
1. **Zero Forcing (ZF)**
   - Linear receiver
   - Noise enhancement possible

2. **Minimum Mean Square Error (MMSE)**
   - Optimal linear receiver
   - Noise-interference tradeoff

3. **Maximum Likelihood (ML)**
   - Optimal receiver (computationally intensive)
   - Exhaustive search over constellation

### Performance Metrics
- **Signal-to-Noise Ratio (SNR)**: Measured post-processing
- **Bit Error Rate (BER)**: Symbol-level error statistics
- **Channel Capacity**: Shannon limit calculations
- **Condition Number**: Channel matrix conditioning

## Signal Mixing and Interference

### Interference Types
1. **Adjacent Channel Interference (ACI)**
   - Frequency offset: ±10-50% of signal bandwidth
   - Power offset: -20 to +10 dB relative to desired signal

2. **Co-channel Interference (CCI)**
   - Same frequency band
   - Different spreading codes (CDMA)
   - Power offset: -15 to +5 dB

3. **Wideband Interference**
   - Gaussian noise with shaped spectrum
   - Bandwidth: 1-10× signal bandwidth
   - Spectral masks: rectangular, raised cosine

4. **Narrowband Interference**
   - Continuous wave (CW) tones
   - Multiple sinusoids
   - Frequency-hopping patterns

### Multi-Standard Coexistence
- **Frequency Allocation**: ITU-R frequency bands
- **Power Control**: Dynamic range -30 to +20 dB
- **Timing Synchronization**: Frame-level alignment
- **Cross-Standard Interference**: Realistic interference scenarios

## Validation Framework

### Signal Quality Metrics
1. **Power Metrics**
   - Average power (dBm)
   - Peak power (dBm)
   - Peak-to-Average Power Ratio (PAPR)
   - Root Mean Square (RMS) power

2. **Spectral Metrics**
   - Bandwidth (3 dB, 6 dB, 99% power)
   - Center frequency
   - Power spectral density
   - Spectral efficiency

3. **Error Metrics**
   - Error Vector Magnitude (EVM)
   - Symbol Error Rate (SER)
   - Constellation analysis
   - Phase noise

### Standards Compliance Validation
- **Bandwidth Verification**: ±20% tolerance from specification
- **PAPR Validation**: Within expected ranges per standard
- **Spectral Masks**: ITU-R emission standards
- **Adjacent Channel Power Ratio (ACPR)**

## Data Format Specifications

### File Formats
- **Signal Data**: NumPy binary format (.npy)
- **Metadata**: JSON or pickled Python dictionaries
- **Documentation**: Markdown (.md) format

### Signal Representation
- **Data Type**: Complex float64 (numpy.complex128)
- **Sample Organization**: Time-domain, baseband equivalent
- **Normalization**: Unit average power (0 dBFS reference)

### Metadata Structure
```python
{
    'signal_type': str,          # 'GSM', 'UMTS', 'LTE', 'NR'
    'sample_rate': float,        # Hz
    'duration': float,           # seconds
    'bandwidth': float,          # Hz
    'center_frequency': float,   # Hz (if upconverted)
    'modulation': str,           # Modulation scheme
    'snr_db': float,            # Signal-to-noise ratio
    'channel_model': dict,       # Channel parameters
    'generation_params': dict,   # Generator-specific parameters
    'validation_results': dict,  # Quality metrics
    'timestamp': str,           # ISO 8601 format
    'version': str              # Dataset version
}
```

## Performance Benchmarks

### Computational Requirements
- **Memory Usage**: ~100 MB per 10 ms signal at 30.72 MHz sampling
- **Generation Speed**: ~1000× real-time on modern CPU
- **GPU Acceleration**: CUDA support for large datasets

### Signal Quality Targets
- **EVM**: < 5% for QPSK, < 3% for higher-order QAM
- **PAPR Accuracy**: ±0.5 dB from theoretical values
- **Bandwidth Accuracy**: ±10% from specification
- **SNR Range**: -20 to +40 dB with 0.1 dB precision

## Reproducibility

### Random Seed Management
- **Global Seeds**: NumPy, Python random
- **Generator-Specific Seeds**: Independent per signal type
- **Configuration Hashing**: Deterministic parameter sets

### Version Control
- **Semantic Versioning**: Major.Minor.Patch format
- **Backward Compatibility**: Maintained for minor versions
- **Migration Tools**: For major version updates

## Research Applications

### Primary Use Cases
1. **Blind Source Separation**: Multi-standard signal separation
2. **Interference Mitigation**: Advanced cancellation algorithms
3. **MIMO Processing**: Spatial multiplexing research
4. **Machine Learning**: Training data for AI/ML models
5. **Standards Development**: Algorithm validation and testing

### Dataset Variants
- **Training Sets**: Large-scale, diverse parameter sweeps
- **Validation Sets**: Known-good reference signals
- **Test Sets**: Challenging, realistic scenarios
- **Benchmark Sets**: Standardized performance comparisons