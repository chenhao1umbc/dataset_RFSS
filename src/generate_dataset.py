"""
Dataset generation orchestration script.

Integrates ParameterSampler with signal generators, channel models, and mixer
to produce complete dataset samples per paper/dataset_parameters.md.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from tqdm import tqdm

from src.utils_dataset import ParameterSampler, DatasetWriter
from src.run_gsm import generate_gsm_signal
from src.run_umts import generate_umts_signal
from src.run_lte import generate_lte_signal
from src.run_5g import generate_5g_signal
from src.utils_channel import (
    apply_tdl_channel, apply_cfo, apply_sfo, apply_iq_imbalance,
    apply_dc_offset, apply_phase_noise, apply_pa_nonlinearity
)
from src.utils_mixing import SignalMixer
from src.utils_shared import add_awgn_noise


def generate_single_source(
    config: Dict[str, Any],
    duration_ms: float = 1.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate single source signal from configuration.

    Args:
        config: Source configuration from ParameterSampler
        duration_ms: Signal duration in milliseconds
        device: PyTorch device

    Returns:
        Tuple of (signal tensor, metadata dict)
    """
    standard = config['standard']
    signal_params = config['signal_params']
    channel_params = config['channel_params']
    impairment_params = config['impairment_params']

    # Generate clean signal based on standard
    if standard == 'GSM':
        result = generate_gsm_signal(
            duration_ms=duration_ms,
            sample_rate=signal_params['sample_rate'],
            carrier_freq=0.0,  # Baseband
            power_dbm=0.0,
            snr_db=None,  # No noise yet
            device=device
        )
    elif standard == 'UMTS':
        result = generate_umts_signal(
            duration_ms=duration_ms,
            sample_rate=signal_params['sample_rate'],
            carrier_freq=0.0,
            power_dbm=0.0,
            snr_db=None,
            device=device
        )
    elif standard == 'LTE':
        result = generate_lte_signal(
            duration_ms=duration_ms,
            bandwidth_mhz=signal_params['bandwidth_mhz'],
            modulation_scheme=signal_params['modulation'],
            carrier_freq=0.0,
            power_dbm=0.0,
            snr_db=None,
            device=device
        )
    elif standard == '5G_NR':
        result = generate_5g_signal(
            duration_ms=duration_ms,
            numerology=signal_params['numerology'],
            bandwidth_mhz=signal_params['bandwidth_mhz'],
            modulation_scheme=signal_params['modulation'],
            carrier_freq=0.0,
            power_dbm=0.0,
            snr_db=None,
            device=device
        )
    else:
        raise ValueError(f"Unknown standard: {standard}")

    signal = result['signal']
    metadata = result['metadata']

    # Apply channel model
    signal_with_channel = apply_tdl_channel(
        signal,
        tdl_model=channel_params['tdl_model'],
        doppler_hz=channel_params['doppler_hz'],
        sample_rate=signal_params['sample_rate'],
        device=device
    )

    # Apply hardware impairments
    signal_impaired = signal_with_channel

    # CFO (convert ppm to Hz)
    if abs(impairment_params['cfo_ppm']) > 1e-6:
        carrier_freq = 2e9  # Nominal carrier frequency
        cfo_hz = impairment_params['cfo_ppm'] * carrier_freq / 1e6
        signal_impaired = apply_cfo(
            signal_impaired,
            cfo_hz=cfo_hz,
            sample_rate=signal_params['sample_rate'],
            device=device
        )

    # SFO
    if abs(impairment_params['sfo_ppm']) > 1e-6:
        signal_impaired = apply_sfo(
            signal_impaired,
            sfo_ppm=impairment_params['sfo_ppm'],
            device=device
        )

    # I/Q imbalance
    if abs(impairment_params['iq_amp_db']) > 1e-6 or abs(impairment_params['iq_phase_deg']) > 1e-6:
        signal_impaired = apply_iq_imbalance(
            signal_impaired,
            amplitude_imb_db=impairment_params['iq_amp_db'],
            phase_imb_deg=impairment_params['iq_phase_deg'],
            device=device
        )

    # DC offset
    if impairment_params['dc_offset_dbc'] > -90.0:
        signal_impaired = apply_dc_offset(
            signal_impaired,
            dc_level_dbc=impairment_params['dc_offset_dbc'],
            device=device
        )

    # Phase noise
    if impairment_params['phase_noise_dbc_hz'] > -115.0:
        signal_impaired = apply_phase_noise(
            signal_impaired,
            phase_noise_dbc_hz=impairment_params['phase_noise_dbc_hz'],
            sample_rate=signal_params['sample_rate'],
            device=device
        )

    # PA nonlinearity
    if impairment_params['pa_backoff_db'] < 9.5:
        signal_impaired = apply_pa_nonlinearity(
            signal_impaired,
            input_backoff_db=impairment_params['pa_backoff_db'],
            smoothness=3.0,
            device=device
        )

    # Update metadata
    metadata['channel'] = channel_params
    metadata['impairments'] = impairment_params

    return signal_impaired, metadata


def generate_sample(
    config: Dict[str, Any],
    duration_ms: float = 1.0,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Generate complete dataset sample from configuration.

    Args:
        config: Complete sample configuration from ParameterSampler
        duration_ms: Signal duration in milliseconds
        device: PyTorch device

    Returns:
        Dictionary containing mixed signal, source signals, and metadata
    """
    num_sources = config['num_sources']
    mixing_params = config['mixing_params']
    snr_db = config['snr_db']
    mimo_config = config['mimo_config']

    # Generate all source signals
    source_signals = []
    source_metadata = []

    for i, source_config in enumerate(config['sources']):
        signal, metadata = generate_single_source(
            source_config,
            duration_ms=duration_ms,
            device=device
        )
        source_signals.append(signal)
        source_metadata.append(metadata)

    # Mix signals if multiple sources
    if num_sources == 1:
        mixed_signal = source_signals[0]
    else:
        # Determine common sample rate (use maximum)
        max_sample_rate = max(metadata['sample_rate'] for metadata in source_metadata)

        # Create mixer
        mixer = SignalMixer(sample_rate=max_sample_rate, device=device)

        # Add sources with appropriate parameters
        for i, (signal, metadata) in enumerate(zip(source_signals, source_metadata)):
            source_sample_rate = metadata['sample_rate']
            power_ratio_db = mixing_params['power_ratios_db'][i]
            freq_offset_hz = mixing_params['frequency_offsets_hz'][i]
            standard = metadata['standard']

            mixer.add_source(
                signal,
                label=standard,
                power_db=power_ratio_db,
                freq_offset_hz=freq_offset_hz,
                timing_offset_samples=0,
                source_sample_rate=source_sample_rate if source_sample_rate != max_sample_rate else None
            )

        # Mix
        mix_result = mixer.mix(mode=mixing_params['mixing_mode'])
        mixed_signal = mix_result['mixed_signal']

    # Add AWGN noise to achieve target SNR
    if snr_db is not None:
        mixed_signal = add_awgn_noise(mixed_signal, snr_db)

    # Prepare output
    sample = {
        'mixed_signal': mixed_signal,
        'source_signals': source_signals,
        'num_sources': num_sources,
        'snr_db': snr_db,
        'mixing_params': mixing_params,
        'mimo_config': mimo_config,
        'source_metadata': source_metadata,
        'config': config
    }

    return sample


def generate_dataset(
    output_path: Path,
    num_samples: int = 500,
    duration_ms: float = 1.0,
    master_seed: int = 42,
    device: str = 'cpu',
    show_progress: bool = True
) -> None:
    """
    Generate complete dataset.

    Args:
        output_path: Path to output HDF5 file
        num_samples: Number of samples to generate
        duration_ms: Signal duration in milliseconds
        master_seed: Master random seed
        device: PyTorch device
        show_progress: Show progress bar
    """
    # Initialize sampler
    sampler = ParameterSampler(seed=master_seed)

    # Initialize writer
    with DatasetWriter(output_path, max_samples=num_samples) as writer:
        # Generate samples
        iterator = range(num_samples)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating dataset")

        for sample_id in iterator:
            # Generate configuration
            config = sampler.generate_sample_config(sample_id)

            # Generate sample
            sample = generate_sample(config, duration_ms=duration_ms, device=device)

            # Write to dataset
            writer.write_sample(
                mixed_signal=sample['mixed_signal'],
                sources=sample['source_signals'],
                config=config
            )

    print(f"\nDataset generation complete: {num_samples} samples")
    print(f"Output: {output_path}")


def main():
    """Command-line interface for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate RFSS dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of samples (default: 500)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Signal duration in ms (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='PyTorch device (default: cpu)')

    args = parser.parse_args()

    output_path = Path(args.output)

    generate_dataset(
        output_path=output_path,
        num_samples=args.num_samples,
        duration_ms=args.duration,
        master_seed=args.seed,
        device=args.device,
        show_progress=True
    )


if __name__ == '__main__':
    main()
