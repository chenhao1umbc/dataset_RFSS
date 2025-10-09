"""
Digital modulation schemes for wireless standards.

Implements QAM constellations with Gray coding per 3GPP specifications:
- BPSK, QPSK
- 16-QAM, 64-QAM, 256-QAM, 1024-QAM
- Power normalization for unit average energy
"""

import torch
import math


def gray_code(n: int) -> int:
    """Convert binary to Gray code."""
    return n ^ (n >> 1)


def inverse_gray_code(g: int) -> int:
    """Convert Gray code to binary."""
    n = g
    while g > 1:
        g >>= 1
        n ^= g
    return n


def generate_qam_constellation(M: int, device: str = 'cpu') -> torch.Tensor:
    """
    Generate square QAM constellation with Gray coding.

    Follows 3GPP specifications for constellation mapping:
    - TS 36.211 Section 7.1 (LTE)
    - TS 38.211 Section 5.1 (5G NR)

    Args:
        M: Modulation order (4, 16, 64, 256, 1024)
        device: PyTorch device

    Returns:
        Complex constellation tensor of shape (M,)
    """
    if M not in [2, 4, 16, 64, 256, 1024]:
        raise ValueError(f"Unsupported modulation order: {M}")

    # For square QAM, sqrt(M) must be integer
    bits_per_symbol = int(math.log2(M))
    k = bits_per_symbol // 2  # bits per dimension

    if M == 2:
        # BPSK: +1, -1 (on real axis)
        constellation = torch.tensor([1.0, -1.0], dtype=torch.complex64, device=device)
    elif M == 4:
        # QPSK: (1+1j), (1-1j), (-1+1j), (-1-1j) / sqrt(2)
        constellation = torch.tensor([
            1.0 + 1.0j,  # 00
            1.0 - 1.0j,  # 01
            -1.0 + 1.0j, # 10
            -1.0 - 1.0j  # 11
        ], dtype=torch.complex64, device=device) / math.sqrt(2)
    else:
        # Higher-order square QAM
        sqrt_M = int(math.sqrt(M))

        # Generate PAM constellation for each dimension
        pam_levels = torch.arange(sqrt_M, dtype=torch.float32, device=device)
        pam_levels = 2 * pam_levels - (sqrt_M - 1)  # Center around 0

        # Create I and Q components
        constellation = torch.zeros(M, dtype=torch.complex64, device=device)

        for idx in range(M):
            # Apply Gray coding
            gray_idx = gray_code(idx)

            # Extract I and Q indices from Gray-coded index
            i_idx = gray_idx >> k
            q_idx = gray_idx & ((1 << k) - 1)

            # Map to constellation point
            i_val = pam_levels[i_idx]
            q_val = pam_levels[q_idx]

            constellation[idx] = complex(i_val, q_val)

    # Normalize to unit average power
    avg_power = torch.mean(torch.abs(constellation) ** 2)
    constellation = constellation / torch.sqrt(avg_power)

    return constellation


def modulate_symbols(bits: torch.Tensor, M: int, device: str = 'cpu') -> torch.Tensor:
    """
    Modulate bit sequence to QAM symbols.

    Args:
        bits: Binary tensor of shape (N,) with values 0 or 1
        M: Modulation order (2, 4, 16, 64, 256, 1024)
        device: PyTorch device

    Returns:
        Complex symbol tensor of shape (N // log2(M),)
    """
    bits_per_symbol = int(math.log2(M))

    # Ensure bits length is multiple of bits_per_symbol
    num_symbols = len(bits) // bits_per_symbol
    bits = bits[:num_symbols * bits_per_symbol]

    # Get constellation
    constellation = generate_qam_constellation(M, device=device)

    # Reshape bits into symbols
    bits_reshaped = bits.reshape(num_symbols, bits_per_symbol)

    # Convert bit groups to indices
    powers = 2 ** torch.arange(bits_per_symbol - 1, -1, -1, dtype=torch.int64, device=device)
    indices = torch.sum(bits_reshaped * powers, dim=1)

    # Map to constellation
    symbols = constellation[indices]

    return symbols


def demodulate_symbols(symbols: torch.Tensor, M: int) -> torch.Tensor:
    """
    Demodulate QAM symbols to bits (hard decision).

    Args:
        symbols: Complex symbol tensor
        M: Modulation order

    Returns:
        Binary tensor
    """
    device = symbols.device
    constellation = generate_qam_constellation(M, device=device)

    # Find nearest constellation point for each symbol
    distances = torch.abs(symbols.unsqueeze(1) - constellation.unsqueeze(0))
    indices = torch.argmin(distances, dim=1)

    # Convert indices to bits
    bits_per_symbol = int(math.log2(M))
    bits = torch.zeros(len(symbols) * bits_per_symbol, dtype=torch.int64, device=device)

    for i, idx in enumerate(indices):
        # Convert index to binary
        idx_val = idx.item()
        for b in range(bits_per_symbol):
            bit_pos = i * bits_per_symbol + b
            bits[bit_pos] = (idx_val >> (bits_per_symbol - 1 - b)) & 1

    return bits


class GmskModulator:
    """
    Gaussian Minimum Shift Keying (GMSK) modulator for GSM.

    Implements GMSK per 3GPP TS 45.004 with BT=0.3.
    """

    def __init__(self, BT: float = 0.3, samples_per_symbol: int = 4, filter_span: int = 4,
                 device: str = 'cpu'):
        """
        Initialize GMSK modulator.

        Args:
            BT: Bandwidth-time product (0.3 for GSM)
            samples_per_symbol: Oversampling factor
            filter_span: Filter length in symbols
            device: PyTorch device
        """
        self.BT = BT
        self.samples_per_symbol = samples_per_symbol
        self.filter_span = filter_span
        self.device = device

        # Pre-compute Gaussian pulse shaping filter
        self.gaussian_filter = self._generate_gaussian_filter()

    def _generate_gaussian_filter(self) -> torch.Tensor:
        """
        Generate Gaussian pulse shaping filter.

        h(t) = (sqrt(2*pi) / (T * sqrt(ln(2)))) * exp(-2*pi^2*B^2*t^2 / ln(2))

        where B = BT / T
        """
        # Time vector
        filter_len = self.filter_span * self.samples_per_symbol
        t = torch.arange(-filter_len // 2, filter_len // 2, dtype=torch.float32, device=self.device)
        t = t / self.samples_per_symbol  # Normalize by symbol period

        # Gaussian filter
        alpha = math.sqrt(2 * math.log(2)) / (2 * math.pi * self.BT)
        h = torch.exp(-t ** 2 / (2 * alpha ** 2))

        # Normalize
        h = h / torch.sum(h)

        return h

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Modulate binary data using GMSK.

        Args:
            bits: Binary tensor (0 or 1)

        Returns:
            Complex baseband GMSK signal
        """
        # Convert bits to NRZ: 0 -> -1, 1 -> +1
        nrz = 2.0 * bits.float() - 1.0

        # Upsample
        upsampled = torch.zeros(len(nrz) * self.samples_per_symbol,
                                dtype=torch.float32, device=self.device)
        upsampled[::self.samples_per_symbol] = nrz

        # Apply Gaussian filter
        filtered = torch.nn.functional.conv1d(
            upsampled.unsqueeze(0).unsqueeze(0),
            self.gaussian_filter.unsqueeze(0).unsqueeze(0),
            padding=len(self.gaussian_filter) // 2
        ).squeeze()

        # Truncate to original length
        filtered = filtered[:len(upsampled)]

        # Integrate filtered signal for frequency modulation
        # Phase(t) = pi * h * integral(filtered(tau), tau=0..t)
        # For MSK, h = 0.5
        h = 0.5
        phase = math.pi * h * torch.cumsum(filtered, dim=0)

        # Generate complex signal
        signal = torch.exp(1j * phase)

        return signal


def get_modulation_order(modulation_scheme: str) -> int:
    """
    Get modulation order M from scheme name.

    Args:
        modulation_scheme: Name like 'QPSK', '16QAM', '64QAM', etc.

    Returns:
        Modulation order M
    """
    scheme_upper = modulation_scheme.upper()

    if scheme_upper in ['BPSK']:
        return 2
    elif scheme_upper in ['QPSK', '4QAM']:
        return 4
    elif scheme_upper in ['16QAM']:
        return 16
    elif scheme_upper in ['64QAM']:
        return 64
    elif scheme_upper in ['256QAM']:
        return 256
    elif scheme_upper in ['1024QAM']:
        return 1024
    else:
        raise ValueError(f"Unknown modulation scheme: {modulation_scheme}")
