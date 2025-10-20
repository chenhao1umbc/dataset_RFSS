"""
UMTS signal generator.

Generates 3G UMTS signals using W-CDMA per 3GPP TS 25.213.
"""

import torch
import argparse
import math
from src.utils_umts import generate_umts_baseband, validate_umts_signal, UMTS_SPECS
from src.utils_shared import normalize_power, add_awgn_noise, add_carrier_frequency


def generate_umts_signal(
    duration_ms: float = 10.0,
    spreading_factor: int = 16,
    sample_rate: float = 15.36e6,  # 4x chip rate
    carrier_freq: float = 2.1e9,  # 2.1 GHz (typical UMTS band)
    power_dbm: float = 0.0,
    snr_db: float = None,
    scrambling_code: int = 0,
    device: str = 'cpu'
) -> dict:
    """
    Generate UMTS signal with specified parameters.

    Args:
        duration_ms: Signal duration in milliseconds
        spreading_factor: Spreading factor (4, 8, 16, 32, 64, 128, 256)
        sample_rate: Sampling rate in Hz
        carrier_freq: Carrier frequency in Hz (0 for baseband only)
        power_dbm: Target signal power in dBm
        snr_db: SNR in dB (if None, no noise added)
        scrambling_code: Gold code number for scrambling
        device: PyTorch device ('cpu', 'cuda', 'mps')

    Returns:
        Dictionary containing:
            - signal: Complex signal tensor
            - metadata: Signal parameters and statistics
    """
    if spreading_factor not in UMTS_SPECS['spreading_factors']:
        raise ValueError(f"Spreading factor must be one of {UMTS_SPECS['spreading_factors']}")

    # Calculate number of symbols needed
    chip_rate = UMTS_SPECS['chip_rate']
    duration_sec = duration_ms / 1000.0
    num_chips = int(chip_rate * duration_sec)
    num_symbols = num_chips // spreading_factor

    # Generate baseband signal
    baseband_signal = generate_umts_baseband(
        num_symbols=num_symbols,
        spreading_factor=spreading_factor,
        sample_rate=sample_rate,
        scrambling_code=scrambling_code,
        device=device
    )

    # Normalize power
    signal = normalize_power(baseband_signal, target_power_db=power_dbm)

    # Add carrier frequency if specified
    if carrier_freq > 0:
        signal = add_carrier_frequency(signal, carrier_freq, sample_rate)

    # Add noise if specified
    if snr_db is not None:
        signal = add_awgn_noise(signal, snr_db)

    # Validate signal
    validation = validate_umts_signal(signal, sample_rate)

    # Calculate effective bit rate
    symbol_rate = chip_rate / spreading_factor
    bit_rate = symbol_rate * 2  # QPSK = 2 bits per symbol

    # Prepare metadata
    metadata = {
        'standard': 'UMTS',
        'modulation': 'QPSK with W-CDMA',
        'chip_rate': chip_rate,
        'spreading_factor': spreading_factor,
        'symbol_rate': symbol_rate,
        'bit_rate': bit_rate,
        'sample_rate': sample_rate,
        'carrier_freq': carrier_freq,
        'duration_ms': duration_ms,
        'num_symbols': num_symbols,
        'scrambling_code': scrambling_code,
        'power_dbm': power_dbm,
        'snr_db': snr_db,
        'validation': validation
    }

    return {
        'signal': signal,
        'metadata': metadata
    }


def main():
    """Command-line interface for UMTS signal generation."""
    parser = argparse.ArgumentParser(description='Generate UMTS signals')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Signal duration in ms (default: 10.0)')
    parser.add_argument('--spreading-factor', type=int, default=16,
                        choices=[4, 8, 16, 32, 64, 128, 256],
                        help='Spreading factor (default: 16)')
    parser.add_argument('--sample-rate', type=float, default=15.36e6,
                        help='Sample rate in Hz (default: 15.36e6)')
    parser.add_argument('--carrier-freq', type=float, default=0.0,
                        help='Carrier frequency in Hz (default: 0 for baseband)')
    parser.add_argument('--power', type=float, default=0.0,
                        help='Signal power in dBm (default: 0)')
    parser.add_argument('--snr', type=float, default=None,
                        help='SNR in dB (default: None, no noise)')
    parser.add_argument('--scrambling-code', type=int, default=0,
                        help='Gold scrambling code number (default: 0)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='PyTorch device (default: cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: None, print only)')

    args = parser.parse_args()

    # Generate signal
    result = generate_umts_signal(
        duration_ms=args.duration,
        spreading_factor=args.spreading_factor,
        sample_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        power_dbm=args.power,
        snr_db=args.snr,
        scrambling_code=args.scrambling_code,
        device=args.device
    )

    # Print metadata
    print("UMTS Signal Generated")
    print("-" * 50)
    for key, value in result['metadata'].items():
        if key == 'validation':
            print("Validation:")
            for vkey, vvalue in value.items():
                print(f"  {vkey}: {vvalue}")
        else:
            print(f"{key}: {value}")

    # Save if requested
    if args.output:
        torch.save(result, args.output)
        print(f"\nSignal saved to: {args.output}")


if __name__ == '__main__':
    main()
