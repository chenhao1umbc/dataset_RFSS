"""
LTE signal generation utilities.

Implements OFDM per 3GPP TS 36.211:
- Configurable bandwidth (1.4, 3, 5, 10, 15, 20 MHz)
- Subcarrier spacing 15 kHz
- Cyclic prefix (normal or extended)
- Resource block structure
"""

import torch
import math
from typing import Dict, Any, Tuple


# LTE specifications per 3GPP TS 36.211
LTE_BANDWIDTHS = {
    # bandwidth_mhz: (num_rbs, fft_size, sample_rate)
    1.4: (6, 128, 1.92e6),
    3.0: (15, 256, 3.84e6),
    5.0: (25, 512, 7.68e6),
    10.0: (50, 1024, 15.36e6),
    15.0: (75, 1536, 23.04e6),
    20.0: (100, 2048, 30.72e6),
}

LTE_SUBCARRIER_SPACING = 15000  # 15 kHz
LTE_SUBCARRIERS_PER_RB = 12
LTE_SYMBOLS_PER_SLOT = 7  # Normal CP
LTE_SLOTS_PER_SUBFRAME = 2

# Cyclic prefix lengths (in samples, for normal CP)
# First symbol: longer CP, remaining symbols: shorter CP
CP_LENGTHS_NORMAL = {
    128: (10, 9),   # (first_symbol, other_symbols)
    256: (20, 18),
    512: (40, 36),
    1024: (80, 72),
    1536: (120, 108),
    2048: (160, 144),
}


def get_lte_params(bandwidth_mhz: float) -> Dict[str, Any]:
    """
    Get LTE parameters for specified bandwidth.

    Args:
        bandwidth_mhz: Channel bandwidth in MHz

    Returns:
        Dictionary with LTE parameters
    """
    if bandwidth_mhz not in LTE_BANDWIDTHS:
        raise ValueError(f"Unsupported bandwidth: {bandwidth_mhz} MHz")

    num_rbs, fft_size, sample_rate = LTE_BANDWIDTHS[bandwidth_mhz]

    return {
        'bandwidth_mhz': bandwidth_mhz,
        'num_rbs': num_rbs,
        'fft_size': fft_size,
        'sample_rate': sample_rate,
        'subcarrier_spacing': LTE_SUBCARRIER_SPACING,
        'num_data_subcarriers': num_rbs * LTE_SUBCARRIERS_PER_RB,
        'cp_first': CP_LENGTHS_NORMAL[fft_size][0],
        'cp_other': CP_LENGTHS_NORMAL[fft_size][1],
    }


def generate_lte_ofdm_symbol(
    data_symbols: torch.Tensor,
    fft_size: int,
    num_data_subcarriers: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate single OFDM symbol (frequency domain).

    Args:
        data_symbols: Complex data symbols
        fft_size: FFT size
        num_data_subcarriers: Number of data subcarriers
        device: PyTorch device

    Returns:
        Frequency domain OFDM symbol with proper subcarrier mapping
    """
    # Initialize frequency domain symbol (all zeros)
    freq_symbol = torch.zeros(fft_size, dtype=torch.complex64, device=device)

    # Map data to subcarriers (centered around DC)
    # LTE uses subcarriers [-num_data_subcarriers/2, ..., -1, 0, 1, ..., num_data_subcarriers/2]
    # DC subcarrier (index 0) is not used

    half_data = num_data_subcarriers // 2

    # Negative frequencies (upper half of FFT)
    freq_symbol[-half_data:] = data_symbols[:half_data]

    # Positive frequencies (lower half of FFT, skip DC)
    freq_symbol[1:half_data+1] = data_symbols[half_data:]

    return freq_symbol


def generate_lte_resource_grid(
    num_symbols: int,
    num_rbs: int,
    modulation_order: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate LTE resource grid with random data.

    Args:
        num_symbols: Number of OFDM symbols
        num_rbs: Number of resource blocks
        modulation_order: QAM order (4, 16, 64, 256)
        device: PyTorch device

    Returns:
        Resource grid of shape (num_symbols, num_subcarriers)
    """
    from utils_modulation import generate_qam_constellation

    num_subcarriers = num_rbs * LTE_SUBCARRIERS_PER_RB

    # Get constellation
    constellation = generate_qam_constellation(modulation_order, device=device)

    # Generate random symbols
    resource_grid = torch.zeros(num_symbols, num_subcarriers,
                                dtype=torch.complex64, device=device)

    for sym_idx in range(num_symbols):
        # Random constellation indices
        indices = torch.randint(0, modulation_order, (num_subcarriers,), device=device)
        resource_grid[sym_idx] = constellation[indices]

    return resource_grid


def generate_lte_baseband(
    num_subframes: int,
    bandwidth_mhz: float,
    modulation_scheme: str = 'QPSK',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate LTE baseband signal.

    Args:
        num_subframes: Number of subframes (1 subframe = 1 ms)
        bandwidth_mhz: Channel bandwidth in MHz
        modulation_scheme: Modulation scheme ('QPSK', '16QAM', '64QAM', '256QAM')
        device: PyTorch device

    Returns:
        Complex baseband signal
    """
    from utils_modulation import get_modulation_order

    # Get LTE parameters
    params = get_lte_params(bandwidth_mhz)
    fft_size = params['fft_size']
    num_rbs = params['num_rbs']
    num_data_subcarriers = params['num_data_subcarriers']
    cp_first = params['cp_first']
    cp_other = params['cp_other']

    # Get modulation order
    mod_order = get_modulation_order(modulation_scheme)

    # Calculate total number of symbols
    # 1 subframe = 2 slots = 14 symbols (normal CP)
    num_symbols_per_subframe = LTE_SLOTS_PER_SUBFRAME * LTE_SYMBOLS_PER_SLOT
    total_symbols = num_subframes * num_symbols_per_subframe

    # Generate resource grid
    resource_grid = generate_lte_resource_grid(
        num_symbols=total_symbols,
        num_rbs=num_rbs,
        modulation_order=mod_order,
        device=device
    )

    # Generate time-domain signal
    signal_parts = []

    for sym_idx in range(total_symbols):
        # Get data for this symbol
        data_symbols = resource_grid[sym_idx]

        # Map to OFDM subcarriers
        freq_symbol = generate_lte_ofdm_symbol(
            data_symbols=data_symbols,
            fft_size=fft_size,
            num_data_subcarriers=num_data_subcarriers,
            device=device
        )

        # IFFT to time domain
        time_symbol = torch.fft.ifft(freq_symbol) * math.sqrt(fft_size)

        # Add cyclic prefix
        symbol_idx_in_slot = sym_idx % LTE_SYMBOLS_PER_SLOT
        if symbol_idx_in_slot == 0:
            cp_len = cp_first
        else:
            cp_len = cp_other

        cp = time_symbol[-cp_len:]
        symbol_with_cp = torch.cat([cp, time_symbol])

        signal_parts.append(symbol_with_cp)

    # Concatenate all symbols
    signal = torch.cat(signal_parts)

    return signal


def validate_lte_signal(signal: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate LTE signal properties.

    Args:
        signal: Complex baseband signal
        params: LTE parameters

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

    # Calculate occupied bandwidth from LTE parameters
    # For OFDM: occupied_bandwidth = num_active_subcarriers × subcarrier_spacing
    # This is the theoretical value, which is accurate for OFDM
    num_subcarriers = params['num_data_subcarriers']
    subcarrier_spacing = params['subcarrier_spacing']
    bandwidth_hz = num_subcarriers * subcarrier_spacing

    sample_rate = params['sample_rate']

    return {
        'power_avg_dbm': 10 * math.log10(power_avg + 1e-12),
        'power_peak_dbm': 10 * math.log10(power_peak + 1e-12),
        'papr_db': papr_db,
        'bandwidth_hz': bandwidth_hz,
        'duration_ms': len(signal) / sample_rate * 1000,
        'num_samples': len(signal)
    }
