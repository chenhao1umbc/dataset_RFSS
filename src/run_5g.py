"""
5G NR signal generator.

Generates 5G NR signals using OFDM with flexible numerology per 3GPP TS 38.211.
"""

import torch
import argparse
import math
from utils_5g import generate_nr_baseband, get_nr_params, validate_nr_signal
from utils_shared import normalize_power, add_awgn_noise, add_carrier_frequency


def generate_5g_signal(
    duration_ms: float = 10.0,
    numerology: int = 1,
    bandwidth_mhz: float = 100.0,
    modulation_scheme: str = 'QPSK',
    carrier_freq: float = 3.5e9,  # 3.5 GHz (typical 5G band)
    power_dbm: float = 0.0,
    snr_db: float = None,
    add_dmrs: bool = True,
    device: str = 'cpu'
) -> dict:
    """
    Generate 5G NR signal with specified parameters.

    Args:
        duration_ms: Signal duration in milliseconds
        numerology: Numerology μ (0, 1, 2, 3, 4)
        bandwidth_mhz: Channel bandwidth in MHz
        modulation_scheme: Modulation ('QPSK', '16QAM', '64QAM', '256QAM', '1024QAM')
        carrier_freq: Carrier frequency in Hz (0 for baseband only)
        power_dbm: Target signal power in dBm
        snr_db: SNR in dB (if None, no noise added)
        add_dmrs: Add DMRS reference signals
        device: PyTorch device ('cpu', 'cuda', 'mps')

    Returns:
        Dictionary containing:
            - signal: Complex signal tensor
            - metadata: Signal parameters and statistics
    """
    # Get NR parameters
    params = get_nr_params(numerology, bandwidth_mhz)

    # Calculate number of slots
    # Slot duration = 1 / (2^μ) ms for numerology μ
    slot_duration_ms = 1.0 / (2 ** numerology)
    num_slots = int(math.ceil(duration_ms / slot_duration_ms))

    # Generate baseband signal
    baseband_signal = generate_nr_baseband(
        num_slots=num_slots,
        numerology=numerology,
        bandwidth_mhz=bandwidth_mhz,
        modulation_scheme=modulation_scheme,
        add_dmrs=add_dmrs,
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
    validation = validate_nr_signal(signal, params)

    # Prepare metadata
    metadata = {
        'standard': '5G NR',
        'modulation': modulation_scheme,
        'numerology': numerology,
        'subcarrier_spacing_khz': params['subcarrier_spacing'] / 1000,
        'bandwidth_mhz': bandwidth_mhz,
        'num_rbs': params['num_rbs'],
        'fft_size': params['fft_size'],
        'sample_rate': params['sample_rate'],
        'carrier_freq': carrier_freq,
        'duration_ms': duration_ms,
        'num_slots': num_slots,
        'slot_duration_ms': slot_duration_ms,
        'add_dmrs': add_dmrs,
        'power_dbm': power_dbm,
        'snr_db': snr_db,
        'validation': validation
    }

    return {
        'signal': signal,
        'metadata': metadata
    }


def main():
    """Command-line interface for 5G NR signal generation."""
    parser = argparse.ArgumentParser(description='Generate 5G NR signals')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Signal duration in ms (default: 10.0)')
    parser.add_argument('--numerology', type=int, default=1,
                        choices=[0, 1, 2, 3, 4],
                        help='Numerology mu (default: 1)')
    parser.add_argument('--bandwidth', type=float, default=100.0,
                        help='Bandwidth in MHz (default: 100.0)')
    parser.add_argument('--modulation', type=str, default='QPSK',
                        choices=['QPSK', '16QAM', '64QAM', '256QAM', '1024QAM'],
                        help='Modulation scheme (default: QPSK)')
    parser.add_argument('--carrier-freq', type=float, default=0.0,
                        help='Carrier frequency in Hz (default: 0 for baseband)')
    parser.add_argument('--power', type=float, default=0.0,
                        help='Signal power in dBm (default: 0)')
    parser.add_argument('--snr', type=float, default=None,
                        help='SNR in dB (default: None, no noise)')
    parser.add_argument('--no-dmrs', action='store_true',
                        help='Disable DMRS reference signals')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='PyTorch device (default: cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: None, print only)')

    args = parser.parse_args()

    # Generate signal
    result = generate_5g_signal(
        duration_ms=args.duration,
        numerology=args.numerology,
        bandwidth_mhz=args.bandwidth,
        modulation_scheme=args.modulation,
        carrier_freq=args.carrier_freq,
        power_dbm=args.power,
        snr_db=args.snr,
        add_dmrs=not args.no_dmrs,
        device=args.device
    )

    # Print metadata
    print("5G NR Signal Generated")
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
