"""
GSM signal generator.

Generates 2G GSM signals using GMSK modulation per 3GPP TS 45.004.
"""

import torch
import argparse
from src.utils_gsm import generate_gsm_baseband, generate_gsm_burst, validate_gsm_signal, GSM_SPECS
from src.utils_shared import normalize_power, add_awgn_noise, add_carrier_frequency
from src.utils_modulation import GmskModulator


def generate_gsm_signal(
    duration_ms: float = 10.0,
    sample_rate: float = 2.166e6,  # ~8x symbol rate
    carrier_freq: float = 900e6,  # 900 MHz (typical GSM band)
    power_dbm: float = 0.0,
    snr_db: float = None,
    device: str = 'cpu'
) -> dict:
    """
    Generate GSM signal with specified parameters.

    Args:
        duration_ms: Signal duration in milliseconds
        sample_rate: Sampling rate in Hz
        carrier_freq: Carrier frequency in Hz (0 for baseband only)
        power_dbm: Target signal power in dBm
        snr_db: SNR in dB (if None, no noise added)
        device: PyTorch device ('cpu', 'cuda', 'mps')

    Returns:
        Dictionary containing:
            - signal: Complex signal tensor
            - metadata: Signal parameters and statistics
    """
    # Calculate number of bits needed
    bit_rate = GSM_SPECS['bit_rate']
    duration_sec = duration_ms / 1000.0
    num_bits = int(bit_rate * duration_sec)

    # Generate baseband signal
    baseband_signal = generate_gsm_baseband(
        num_bits=num_bits,
        sample_rate=sample_rate,
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
    validation = validate_gsm_signal(signal, sample_rate)

    # Prepare metadata
    metadata = {
        'standard': 'GSM',
        'modulation': 'GMSK',
        'bit_rate': bit_rate,
        'sample_rate': sample_rate,
        'carrier_freq': carrier_freq,
        'duration_ms': duration_ms,
        'num_bits': num_bits,
        'power_dbm': power_dbm,
        'snr_db': snr_db,
        'validation': validation
    }

    return {
        'signal': signal,
        'metadata': metadata
    }


def main():
    """Command-line interface for GSM signal generation."""
    parser = argparse.ArgumentParser(description='Generate GSM signals')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Signal duration in ms (default: 10.0)')
    parser.add_argument('--sample-rate', type=float, default=2.166e6,
                        help='Sample rate in Hz (default: 2.166e6)')
    parser.add_argument('--carrier-freq', type=float, default=0.0,
                        help='Carrier frequency in Hz (default: 0 for baseband)')
    parser.add_argument('--power', type=float, default=0.0,
                        help='Signal power in dBm (default: 0)')
    parser.add_argument('--snr', type=float, default=None,
                        help='SNR in dB (default: None, no noise)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='PyTorch device (default: cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: None, print only)')

    args = parser.parse_args()

    # Generate signal
    result = generate_gsm_signal(
        duration_ms=args.duration,
        sample_rate=args.sample_rate,
        carrier_freq=args.carrier_freq,
        power_dbm=args.power,
        snr_db=args.snr,
        device=args.device
    )

    # Print metadata
    print("GSM Signal Generated")
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
