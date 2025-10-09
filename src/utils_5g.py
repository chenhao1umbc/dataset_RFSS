"""
5G NR signal generation utilities.

Implements OFDM with flexible numerology per 3GPP TS 38.211:
- Multiple numerologies (μ = 0, 1, 2, 3, 4)
- Subcarrier spacing: 15 * 2^μ kHz
- Configurable bandwidth
- DMRS reference signals
"""

import torch
import math
from typing import Dict, Any, Optional


# 5G NR numerology specifications per 3GPP TS 38.211
NR_NUMEROLOGIES = {
    # μ: (subcarrier_spacing_khz, cyclic_prefix_type)
    0: (15, 'normal'),    # Same as LTE
    1: (30, 'normal'),    # Most common for FR1
    2: (60, 'normal'),    # Extended CP available
    3: (120, 'normal'),   # FR2 (mmWave)
    4: (240, 'normal'),   # FR2 (mmWave)
}

# FFT sizes and resource blocks for different bandwidths and numerologies
# Format: (numerology, bandwidth_mhz): (num_rbs, fft_size)
NR_CONFIGURATIONS = {
    # Numerology 0 (15 kHz SCS)
    (0, 10): (52, 1024),
    (0, 20): (106, 2048),
    (0, 50): (270, 4096),
    # Numerology 1 (30 kHz SCS)
    (1, 10): (24, 512),
    (1, 20): (51, 1024),
    (1, 50): (133, 2048),
    (1, 100): (273, 4096),
    # Numerology 2 (60 kHz SCS)
    (2, 50): (66, 1024),
    (2, 100): (132, 2048),
    (2, 200): (264, 4096),
    # Numerology 3 (120 kHz SCS) - mmWave
    (3, 50): (32, 512),
    (3, 100): (66, 1024),
    (3, 200): (132, 2048),
    (3, 400): (264, 4096),
}

NR_SUBCARRIERS_PER_RB = 12
NR_SYMBOLS_PER_SLOT = 14  # Normal CP


def get_nr_params(numerology: int, bandwidth_mhz: float) -> Dict[str, Any]:
    """
    Get 5G NR parameters for specified numerology and bandwidth.

    Args:
        numerology: Numerology μ (0, 1, 2, 3, 4)
        bandwidth_mhz: Channel bandwidth in MHz

    Returns:
        Dictionary with 5G NR parameters
    """
    if numerology not in NR_NUMEROLOGIES:
        raise ValueError(f"Unsupported numerology: {numerology}")

    if (numerology, bandwidth_mhz) not in NR_CONFIGURATIONS:
        raise ValueError(f"Unsupported configuration: μ={numerology}, BW={bandwidth_mhz} MHz")

    subcarrier_spacing_khz, cp_type = NR_NUMEROLOGIES[numerology]
    num_rbs, fft_size = NR_CONFIGURATIONS[(numerology, bandwidth_mhz)]

    # Calculate sample rate
    subcarrier_spacing = subcarrier_spacing_khz * 1000  # Convert to Hz
    sample_rate = fft_size * subcarrier_spacing

    # Calculate CP length (simplified - depends on μ)
    # For normal CP: ~7% of symbol duration
    cp_ratio = 0.07
    cp_samples = int(fft_size * cp_ratio)

    return {
        'numerology': numerology,
        'bandwidth_mhz': bandwidth_mhz,
        'num_rbs': num_rbs,
        'fft_size': fft_size,
        'sample_rate': sample_rate,
        'subcarrier_spacing': subcarrier_spacing,
        'num_data_subcarriers': num_rbs * NR_SUBCARRIERS_PER_RB,
        'cp_samples': cp_samples,
        'cp_type': cp_type,
    }


def generate_dmrs_sequence(length: int, slot_idx: int = 0, device: str = 'cpu') -> torch.Tensor:
    """
    Generate DMRS (Demodulation Reference Signal) sequence.

    Implements pseudo-random sequence per 3GPP TS 38.211 Section 5.2.2.
    Uses Gold sequence (simplified implementation).

    Args:
        length: Sequence length
        slot_idx: Slot index for sequence initialization
        device: PyTorch device

    Returns:
        Complex DMRS sequence
    """
    # Initialize Gold sequence generator with c_init based on slot index
    # Simplified: use slot_idx as seed for reproducibility
    c_init = slot_idx + 1000  # Offset to avoid seed=0

    # Generate pseudo-random sequence using torch.Generator for determinism
    generator = torch.Generator(device=device).manual_seed(c_init)

    # Generate random phase values
    phases = torch.rand(length, generator=generator, device=device) * 2 * math.pi

    # Create QPSK-like DMRS (real and imaginary are ±1/sqrt(2))
    dmrs_real = torch.cos(phases) / math.sqrt(2)
    dmrs_imag = torch.sin(phases) / math.sqrt(2)

    # Round to nearest QPSK constellation point
    dmrs_real = torch.sign(dmrs_real) / math.sqrt(2)
    dmrs_imag = torch.sign(dmrs_imag) / math.sqrt(2)

    dmrs = torch.complex(dmrs_real, dmrs_imag)

    return dmrs


def generate_nr_ofdm_symbol(
    data_symbols: torch.Tensor,
    fft_size: int,
    num_data_subcarriers: int,
    dmrs_symbols: Optional[torch.Tensor] = None,
    dmrs_positions: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate single 5G NR OFDM symbol (frequency domain).

    Args:
        data_symbols: Complex data symbols
        fft_size: FFT size
        num_data_subcarriers: Number of data subcarriers
        dmrs_symbols: DMRS symbols (optional)
        dmrs_positions: DMRS subcarrier positions (optional)
        device: PyTorch device

    Returns:
        Frequency domain OFDM symbol
    """
    # Initialize frequency domain symbol
    freq_symbol = torch.zeros(fft_size, dtype=torch.complex64, device=device)

    # Map data to subcarriers (centered around DC)
    half_data = num_data_subcarriers // 2

    # Create data symbol index
    data_idx = 0

    # Negative frequencies
    for i in range(fft_size - half_data, fft_size):
        if dmrs_positions is not None and i in dmrs_positions:
            # Insert DMRS
            dmrs_idx = (dmrs_positions == i).nonzero(as_tuple=True)[0][0]
            freq_symbol[i] = dmrs_symbols[dmrs_idx]
        else:
            # Insert data
            if data_idx < len(data_symbols):
                freq_symbol[i] = data_symbols[data_idx]
                data_idx += 1

    # Positive frequencies (skip DC)
    for i in range(1, half_data + 1):
        if dmrs_positions is not None and i in dmrs_positions:
            # Insert DMRS
            dmrs_idx = (dmrs_positions == i).nonzero(as_tuple=True)[0][0]
            freq_symbol[i] = dmrs_symbols[dmrs_idx]
        else:
            # Insert data
            if data_idx < len(data_symbols):
                freq_symbol[i] = data_symbols[data_idx]
                data_idx += 1

    return freq_symbol


def generate_nr_resource_grid(
    num_symbols: int,
    num_rbs: int,
    modulation_order: int,
    add_dmrs: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate 5G NR resource grid with random data and DMRS.

    Args:
        num_symbols: Number of OFDM symbols
        num_rbs: Number of resource blocks
        modulation_order: QAM order (4, 16, 64, 256)
        add_dmrs: Whether to add DMRS symbols
        device: PyTorch device

    Returns:
        Resource grid of shape (num_symbols, num_subcarriers)
    """
    from utils_modulation import generate_qam_constellation

    num_subcarriers = num_rbs * NR_SUBCARRIERS_PER_RB

    # Get constellation
    constellation = generate_qam_constellation(modulation_order, device=device)

    # Generate random symbols
    resource_grid = torch.zeros(num_symbols, num_subcarriers,
                                dtype=torch.complex64, device=device)

    for sym_idx in range(num_symbols):
        # Add DMRS on certain symbols (e.g., symbol 2 and 11 in each slot)
        is_dmrs_symbol = add_dmrs and (sym_idx % NR_SYMBOLS_PER_SLOT in [2, 11])

        if is_dmrs_symbol:
            # Every 2nd subcarrier is DMRS, rest is data
            dmrs_seq = generate_dmrs_sequence(num_subcarriers // 2, slot_idx=sym_idx // NR_SYMBOLS_PER_SLOT, device=device)

            for sc_idx in range(num_subcarriers):
                if sc_idx % 2 == 0:
                    # DMRS
                    resource_grid[sym_idx, sc_idx] = dmrs_seq[sc_idx // 2]
                else:
                    # Data
                    data_idx = torch.randint(0, modulation_order, (1,), device=device)[0]
                    resource_grid[sym_idx, sc_idx] = constellation[data_idx]
        else:
            # All data
            indices = torch.randint(0, modulation_order, (num_subcarriers,), device=device)
            resource_grid[sym_idx] = constellation[indices]

    return resource_grid


def generate_nr_baseband(
    num_slots: int,
    numerology: int,
    bandwidth_mhz: float,
    modulation_scheme: str = 'QPSK',
    add_dmrs: bool = True,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate 5G NR baseband signal.

    Args:
        num_slots: Number of slots
        numerology: Numerology μ (0, 1, 2, 3, 4)
        bandwidth_mhz: Channel bandwidth in MHz
        modulation_scheme: Modulation scheme ('QPSK', '16QAM', '64QAM', '256QAM', '1024QAM')
        add_dmrs: Whether to add DMRS
        device: PyTorch device

    Returns:
        Complex baseband signal
    """
    from utils_modulation import get_modulation_order

    # Get NR parameters
    params = get_nr_params(numerology, bandwidth_mhz)
    fft_size = params['fft_size']
    num_rbs = params['num_rbs']
    num_data_subcarriers = params['num_data_subcarriers']
    cp_samples = params['cp_samples']

    # Get modulation order
    mod_order = get_modulation_order(modulation_scheme)

    # Calculate total number of symbols
    total_symbols = num_slots * NR_SYMBOLS_PER_SLOT

    # Generate resource grid
    resource_grid = generate_nr_resource_grid(
        num_symbols=total_symbols,
        num_rbs=num_rbs,
        modulation_order=mod_order,
        add_dmrs=add_dmrs,
        device=device
    )

    # Generate time-domain signal
    signal_parts = []

    for sym_idx in range(total_symbols):
        # Get data for this symbol
        data_symbols = resource_grid[sym_idx]

        # Map to OFDM subcarriers
        freq_symbol = generate_nr_ofdm_symbol(
            data_symbols=data_symbols,
            fft_size=fft_size,
            num_data_subcarriers=num_data_subcarriers,
            device=device
        )

        # IFFT to time domain
        time_symbol = torch.fft.ifft(freq_symbol) * math.sqrt(fft_size)

        # Add cyclic prefix
        cp = time_symbol[-cp_samples:]
        symbol_with_cp = torch.cat([cp, time_symbol])

        signal_parts.append(symbol_with_cp)

    # Concatenate all symbols
    signal = torch.cat(signal_parts)

    return signal


def validate_nr_signal(signal: torch.Tensor, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate 5G NR signal properties.

    Args:
        signal: Complex baseband signal
        params: 5G NR parameters

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

    # Calculate spectrum
    spectrum = torch.fft.fft(signal)
    power_spectrum = torch.abs(spectrum) ** 2

    # Occupied bandwidth (99% power)
    total_power = torch.sum(power_spectrum)
    cumsum = torch.cumsum(power_spectrum, dim=0)
    bandwidth_99 = torch.sum(cumsum < 0.99 * total_power).item()

    sample_rate = params['sample_rate']
    bandwidth_hz = bandwidth_99 * sample_rate / len(signal)

    return {
        'power_avg_dbm': 10 * math.log10(power_avg + 1e-12),
        'power_peak_dbm': 10 * math.log10(power_peak + 1e-12),
        'papr_db': papr_db,
        'bandwidth_hz': bandwidth_hz,
        'duration_ms': len(signal) / sample_rate * 1000,
        'num_samples': len(signal)
    }
