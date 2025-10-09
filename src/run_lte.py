"""
LTE signal generator.

Generates 4G LTE signals using OFDM per 3GPP TS 36.211.
"""

import torch
import argparse
import math
from utils_lte import generate_lte_baseband, get_lte_params, validate_lte_signal
from utils_shared import normalize_power, add_awgn_noise, add_carrier_frequency


def generate_lte_signal(
    duration_ms: float = 10.0,
    bandwidth_mhz: float = 10.0,
    modulation_scheme: str = 'QPSK',
    carrier_freq: float = 2.0e9,  # 2 GHz (typical LTE band)
    power_dbm: float = 0.0,
    snr_db: float = None,
    device: str = 'cpu'
) -> dict:
    """
    Generate LTE signal with specified parameters.

    Args:
        duration_ms: Signal duration in milliseconds
        bandwidth_mhz: Channel bandwidth (1.4, 3, 5, 10, 15, 20 MHz)
        modulation_scheme: Modulation ('QPSK', '16QAM', '64QAM', '256QAM')
        carrier_freq: Carrier frequency in Hz (0 for baseband only)
        power_dbm: Target signal power in dBm
        snr_db: SNR in dB (if None, no noise added)
        device: PyTorch device ('cpu', 'cuda', 'mps')

    Returns:
        Dictionary containing:
            - signal: Complex signal tensor
            - metadata: Signal parameters and statistics
    """
    # Get LTE parameters
    params = get_lte_params(bandwidth_mhz)

    # Calculate number of subframes (1 subframe = 1 ms)
    num_subframes = int(math.ceil(duration_ms))

    # Generate baseband signal
    baseband_signal = generate_lte_baseband(
        num_subframes=num_subframes,
        bandwidth_mhz=bandwidth_mhz,
        modulation_scheme=modulation_scheme,
        device=device
    )

    # Normalize power
    signal = normalize_power(baseband_signal, target_power_db=power_dbm)

    # Add carrier frequency if specified
    if carrier_freq > 0:
        sample_rate = params['sample_rate']
        signal = add_carrier_frequency(signal, carrier_freq, sample_rate)

    # Add noise if specified
    if snr_db is not None:
        signal = add_awgn_noise(signal, snr_db)

    # Validate signal
    validation = validate_lte_signal(signal, params)

    # Prepare metadata
    metadata = {
        'standard': 'LTE',
        'modulation': modulation_scheme,
        'bandwidth_mhz': bandwidth_mhz,
        'num_rbs': params['num_rbs'],
        'fft_size': params['fft_size'],
        'sample_rate': params['sample_rate'],
        'carrier_freq': carrier_freq,
        'duration_ms': duration_ms,
        'num_subframes': num_subframes,
        'power_dbm': power_dbm,
        'snr_db': snr_db,
        'validation': validation
    }

    return {
        'signal': signal,
        'metadata': metadata
    }


def main():
    """Command-line interface for LTE signal generation."""
    parser = argparse.ArgumentParser(description='Generate LTE signals')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Signal duration in ms (default: 10.0)')
    parser.add_argument('--bandwidth', type=float, default=10.0,
                        choices=[1.4, 3.0, 5.0, 10.0, 15.0, 20.0],
                        help='Bandwidth in MHz (default: 10.0)')
    parser.add_argument('--modulation', type=str, default='QPSK',
                        choices=['QPSK', '16QAM', '64QAM', '256QAM'],
                        help='Modulation scheme (default: QPSK)')
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
    result = generate_lte_signal(
        duration_ms=args.duration,
        bandwidth_mhz=args.bandwidth,
        modulation_scheme=args.modulation,
        carrier_freq=args.carrier_freq,
        power_dbm=args.power,
        snr_db=args.snr,
        device=args.device
    )

    # Print metadata
    print("LTE Signal Generated")
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
