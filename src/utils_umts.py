"""
UMTS signal generation utilities.

Implements W-CDMA per 3GPP TS 25.213 and TS 25.214:
- OVSF spreading codes
- Gold scrambling codes (proper implementation)
- Root-raised cosine pulse shaping
- QPSK modulation with spreading factor 4-256
"""

import torch
import math
from typing import Dict, Any, Tuple


# UMTS specifications per 3GPP TS 25.213
UMTS_SPECS = {
    'chip_rate': 3.84e6,  # 3.84 Mcps
    'channel_bandwidth': 5e6,  # 5 MHz
    'spreading_factors': [4, 8, 16, 32, 64, 128, 256],
    'rrc_rolloff': 0.22,  # Root-raised cosine roll-off factor
    'samples_per_chip': 4,  # Oversampling factor
}


def generate_ovsf_codes(spreading_factor: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate OVSF (Orthogonal Variable Spreading Factor) codes.

    Uses tree structure per 3GPP TS 25.213:
    C_1 = [1]
    C_2n = [C_n, C_n]
    C_2n+1 = [C_n, -C_n]

    Args:
        spreading_factor: Spreading factor (must be power of 2)
        device: PyTorch device

    Returns:
        OVSF code matrix of shape (spreading_factor, spreading_factor)
    """
    if spreading_factor & (spreading_factor - 1) != 0:
        raise ValueError(f"Spreading factor must be power of 2, got {spreading_factor}")

    # Start with Hadamard matrix of size 1
    H = torch.tensor([[1]], dtype=torch.float32, device=device)

    # Build recursively
    current_sf = 1
    while current_sf < spreading_factor:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        current_sf *= 2

    return H


def generate_gold_code_sequence(length: int, code_number: int = 0, device: str = 'cpu') -> torch.Tensor:
    """
    Generate Gold code sequence per 3GPP TS 25.213.

    Gold codes are generated from two m-sequences (x and y):
    - x: polynomial 1 + X^7 + X^18
    - y: polynomial 1 + X^5 + X^7 + X^10 + X^18

    Args:
        length: Sequence length (typically 38400 chips for 10ms frame)
        code_number: Gold code number (0-16777215)
        device: PyTorch device

    Returns:
        Gold code sequence (+1/-1)
    """
    # Register length for UMTS Gold codes
    n = 18

    # Initialize shift registers with code_number
    # x register initialized with code_number
    # y register initialized to all 1s
    x_reg = torch.zeros(n, dtype=torch.int64, device=device)
    y_reg = torch.ones(n, dtype=torch.int64, device=device)

    # Initialize x register based on code_number
    for i in range(n):
        x_reg[i] = (code_number >> i) & 1

    # If x_reg is all zeros, set to all ones (avoid all-zero state)
    if torch.all(x_reg == 0):
        x_reg[:] = 1

    # Generate sequence
    sequence = torch.zeros(length, dtype=torch.int64, device=device)

    for i in range(length):
        # Output is XOR of last bits of both registers
        sequence[i] = x_reg[-1] ^ y_reg[-1]

        # Feedback for x: taps at positions 7 and 18 (0-indexed: 6 and 17)
        x_feedback = x_reg[17] ^ x_reg[6]

        # Feedback for y: taps at positions 5, 7, 10, 18 (0-indexed: 4, 6, 9, 17)
        y_feedback = y_reg[17] ^ y_reg[9] ^ y_reg[6] ^ y_reg[4]

        # Shift registers
        x_reg = torch.roll(x_reg, 1)
        x_reg[0] = x_feedback

        y_reg = torch.roll(y_reg, 1)
        y_reg[0] = y_feedback

    # Convert to bipolar (+1/-1)
    bipolar_sequence = 1.0 - 2.0 * sequence.float()

    return bipolar_sequence


def generate_rrc_filter(
    rolloff: float,
    span_symbols: int,
    samples_per_symbol: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate root-raised cosine (RRC) pulse shaping filter.

    Per 3GPP TS 25.104 for UMTS transmit filtering.

    Args:
        rolloff: Roll-off factor (α = 0.22 for UMTS)
        span_symbols: Filter span in symbols
        samples_per_symbol: Samples per symbol (oversampling)
        device: PyTorch device

    Returns:
        RRC filter coefficients
    """
    # Time vector
    filter_len = span_symbols * samples_per_symbol + 1
    t = torch.arange(-(filter_len // 2), filter_len // 2 + 1,
                     dtype=torch.float32, device=device)
    t = t / samples_per_symbol

    # RRC filter formula
    h = torch.zeros_like(t)

    # Handle special cases
    eps = 1e-10

    for i, t_val in enumerate(t):
        if abs(t_val) < eps:
            # t = 0
            h[i] = (1 + rolloff * (4 / math.pi - 1))
        elif abs(abs(t_val) - 1 / (4 * rolloff)) < eps:
            # t = ±1/(4α)
            h[i] = (rolloff / math.sqrt(2)) * (
                (1 + 2 / math.pi) * math.sin(math.pi / (4 * rolloff)) +
                (1 - 2 / math.pi) * math.cos(math.pi / (4 * rolloff))
            )
        else:
            # General case
            numerator = math.sin(math.pi * t_val * (1 - rolloff)) + \
                       4 * rolloff * t_val * math.cos(math.pi * t_val * (1 + rolloff))
            denominator = math.pi * t_val * (1 - (4 * rolloff * t_val) ** 2)
            h[i] = numerator / denominator

    # Normalize energy
    h = h / torch.sqrt(torch.sum(h ** 2))

    return h


def generate_umts_baseband(
    num_symbols: int,
    spreading_factor: int = 16,
    sample_rate: float = 15.36e6,  # 4x chip rate
    scrambling_code: int = 0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate UMTS W-CDMA baseband signal.

    Args:
        num_symbols: Number of data symbols
        spreading_factor: Spreading factor (4-256)
        sample_rate: Sampling rate in Hz
        scrambling_code: Gold code number for scrambling
        device: PyTorch device

    Returns:
        Complex baseband signal
    """
    from utils_modulation import generate_qam_constellation

    # Generate random QPSK symbols
    qpsk_constellation = generate_qam_constellation(4, device=device)
    symbol_indices = torch.randint(0, 4, (num_symbols,), device=device)
    data_symbols = qpsk_constellation[symbol_indices]

    # Generate OVSF spreading code (use code 0)
    ovsf_codes = generate_ovsf_codes(spreading_factor, device=device)
    spreading_code = ovsf_codes[0]  # Use first code

    # Spread symbols
    spread_chips = []
    for symbol in data_symbols:
        # Multiply symbol by spreading code
        chips = symbol * spreading_code
        spread_chips.append(chips)

    spread_signal = torch.cat(spread_chips)

    # Generate scrambling code
    num_chips = len(spread_signal)
    scrambling_sequence = generate_gold_code_sequence(num_chips, scrambling_code, device=device)

    # Apply scrambling
    scrambled_signal = spread_signal * torch.complex(scrambling_sequence, torch.zeros_like(scrambling_sequence))

    # Upsample for pulse shaping
    samples_per_chip = int(sample_rate / UMTS_SPECS['chip_rate'])
    upsampled_len = len(scrambled_signal) * samples_per_chip
    upsampled = torch.zeros(upsampled_len, dtype=torch.complex64, device=device)
    upsampled[::samples_per_chip] = scrambled_signal

    # Generate RRC filter
    rrc_filter = generate_rrc_filter(
        rolloff=UMTS_SPECS['rrc_rolloff'],
        span_symbols=8,  # 8 chip spans
        samples_per_symbol=samples_per_chip,
        device=device
    )

    # Apply pulse shaping (convolve with RRC)
    # Separate real and imaginary parts
    real_filtered = torch.nn.functional.conv1d(
        upsampled.real.unsqueeze(0).unsqueeze(0),
        rrc_filter.unsqueeze(0).unsqueeze(0),
        padding=len(rrc_filter) // 2
    ).squeeze()

    imag_filtered = torch.nn.functional.conv1d(
        upsampled.imag.unsqueeze(0).unsqueeze(0),
        rrc_filter.unsqueeze(0).unsqueeze(0),
        padding=len(rrc_filter) // 2
    ).squeeze()

    # Truncate to original length
    real_filtered = real_filtered[:upsampled_len]
    imag_filtered = imag_filtered[:upsampled_len]

    # Combine
    shaped_signal = torch.complex(real_filtered, imag_filtered)

    return shaped_signal


def validate_umts_signal(signal: torch.Tensor, sample_rate: float) -> Dict[str, Any]:
    """
    Validate UMTS signal properties.

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

    # Calculate spectrum
    spectrum = torch.fft.fft(signal)
    power_spectrum = torch.abs(spectrum) ** 2

    # Find -3dB bandwidth
    max_power = torch.max(power_spectrum)
    threshold = max_power / 2

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
