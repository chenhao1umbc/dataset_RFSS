"""
GSM signal generation utilities.

Implements GMSK modulation per 3GPP TS 45.004:
- Gaussian filter with BT=0.3
- 270.833 kbps data rate
- 200 kHz channel bandwidth
"""

import torch
import math
from typing import Dict, Any
from utils_modulation import GmskModulator


# GSM specifications per 3GPP TS 45.004
GSM_SPECS = {
    'bit_rate': 270833,  # 270.833 kbps
    'symbol_rate': 270833,  # Same as bit rate for GMSK
    'channel_bandwidth': 200000,  # 200 kHz
    'bt_product': 0.3,  # Bandwidth-time product for Gaussian filter
    'samples_per_symbol': 8,  # Oversampling factor
    'burst_length_bits': 148,  # Normal burst length
    'filter_span_symbols': 4,  # Gaussian filter span
}


def generate_gsm_baseband(
    num_bits: int,
    sample_rate: float,
    device: str = 'cpu',
    **kwargs
) -> torch.Tensor:
    """
    Generate GSM GMSK baseband signal.

    Args:
        num_bits: Number of data bits to generate
        sample_rate: Sampling rate in Hz
        device: PyTorch device
        **kwargs: Additional parameters (for future extension)

    Returns:
        Complex baseband signal
    """
    # Generate random data bits
    bits = torch.randint(0, 2, (num_bits,), dtype=torch.int64, device=device)

    # Calculate samples per symbol based on sample rate and symbol rate
    samples_per_symbol = int(sample_rate / GSM_SPECS['symbol_rate'])

    # Create GMSK modulator
    modulator = GmskModulator(
        BT=GSM_SPECS['bt_product'],
        samples_per_symbol=samples_per_symbol,
        filter_span=GSM_SPECS['filter_span_symbols'],
        device=device
    )

    # Modulate
    signal = modulator.modulate(bits)

    return signal


def generate_gsm_burst(device: str = 'cpu') -> torch.Tensor:
    """
    Generate GSM normal burst structure.

    Normal burst structure (148 bits):
    - Tail bits: 3 bits (0)
    - Encrypted data: 58 bits
    - Training sequence: 26 bits
    - Encrypted data: 58 bits
    - Tail bits: 3 bits (0)

    Args:
        device: PyTorch device

    Returns:
        Burst bit sequence (148 bits)
    """
    # Training sequence (26 bits) - using TS0 as reference
    # In real GSM, there are 8 training sequences defined
    # TS0 is: 0 0 1 0 0 1 0 1 1 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1 1 1
    training_seq = torch.tensor([
        0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1
    ], dtype=torch.int64, device=device)

    # Generate random data bits
    data1 = torch.randint(0, 2, (58,), dtype=torch.int64, device=device)
    data2 = torch.randint(0, 2, (58,), dtype=torch.int64, device=device)

    # Tail bits (all zeros)
    tail = torch.zeros(3, dtype=torch.int64, device=device)

    # Assemble burst: tail + data + training + data + tail
    burst = torch.cat([tail, data1, training_seq, data2, tail])

    return burst


def validate_gsm_signal(signal: torch.Tensor, sample_rate: float) -> Dict[str, Any]:
    """
    Validate GSM signal properties.

    Args:
        signal: Complex baseband signal
        sample_rate: Sampling rate in Hz

    Returns:
        Dictionary with validation metrics
    """
    # Calculate power
    power_avg = torch.mean(torch.abs(signal) ** 2).item()
    power_peak = torch.max(torch.abs(signal) ** 2).item()

    # Calculate PAPR
    if power_avg > 0:
        papr_db = 10 * math.log10(power_peak / power_avg)
    else:
        papr_db = float('inf')

    # Calculate bandwidth (estimate from spectrum)
    spectrum = torch.fft.fft(signal)
    power_spectrum = torch.abs(spectrum) ** 2

    # Find -3dB bandwidth
    max_power = torch.max(power_spectrum)
    threshold = max_power / 2  # -3dB point

    # Frequency bins
    freqs = torch.fft.fftfreq(len(signal), d=1 / sample_rate)

    # Find bandwidth
    above_threshold = power_spectrum > threshold
    if torch.any(above_threshold):
        freq_indices = torch.where(above_threshold)[0]
        bandwidth = (freqs[freq_indices.max()] - freqs[freq_indices.min()]).item()
        bandwidth = abs(bandwidth)
    else:
        bandwidth = 0.0

    return {
        'power_avg_dbm': 10 * math.log10(power_avg + 1e-12),
        'power_peak_dbm': 10 * math.log10(power_peak + 1e-12),
        'papr_db': papr_db,
        'bandwidth_hz': bandwidth,
        'duration_ms': len(signal) / sample_rate * 1000,
        'num_samples': len(signal)
    }
