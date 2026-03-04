"""
Dataset generation orchestration script.

Integrates ParameterSampler with signal generators, channel models, and mixer
to produce complete dataset samples per paper/dataset_parameters.md.
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, Tuple
from tqdm import tqdm

from src.utils_dataset import ParameterSampler, DatasetWriter, STANDARDS
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

    # Use actual sample rate from generator (not sampler, which may differ for 5G)
    actual_sample_rate = metadata['sample_rate']

    # Apply channel model
    signal_with_channel = apply_tdl_channel(
        signal,
        tdl_model=channel_params['tdl_model'],
        doppler_hz=channel_params['doppler_hz'],
        sample_rate=actual_sample_rate,
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
            sample_rate=actual_sample_rate,
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
            sample_rate=actual_sample_rate,
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

    # Determine common sample rate (use maximum across all sources)
    max_sample_rate = max(meta['sample_rate'] for meta in source_metadata)

    # Create mixer and add all sources
    mixer = SignalMixer(sample_rate=max_sample_rate, device=device)
    for i, (signal, meta) in enumerate(zip(source_signals, source_metadata)):
        source_sample_rate = meta['sample_rate']
        mixer.add_source(
            signal,
            label=meta['standard'],
            power_db=mixing_params['power_ratios_db'][i],
            freq_offset_hz=mixing_params['frequency_offsets_hz'][i],
            timing_offset_samples=0,
            source_sample_rate=source_sample_rate if source_sample_rate != max_sample_rate else None
        )

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


def generate_single_source_sample(
    config: Dict[str, Any],
    duration_ms: float = 1.0,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Generate a single-standard sample (no mixing).

    The source signal (channel + impairments, before AWGN) is stored as ground
    truth. The mixed signal is the same signal with AWGN added.

    Args:
        config: Single-source configuration from ParameterSampler.generate_single_source_config()
        duration_ms: Signal duration in milliseconds
        device: PyTorch device

    Returns:
        Dictionary containing mixed signal, source signal, and metadata
    """
    signal, metadata = generate_single_source(config['sources'][0], duration_ms, device)
    source_signal = signal.clone()
    snr_db = config['snr_db']
    if snr_db is not None:
        mixed_signal = add_awgn_noise(signal, snr_db)
    else:
        mixed_signal = signal
    return {
        'mixed_signal': mixed_signal,
        'source_signals': [source_signal],
        'num_sources': 1,
        'snr_db': snr_db,
        'config': config,
    }


def generate_single_source_dataset(
    output_path: Path,
    num_samples_per_standard: int = 1000,
    duration_ms: float = 1.0,
    master_seed: int = 42,
    device: str = 'cpu',
    show_progress: bool = True,
    checkpoint_interval: int = 500
) -> None:
    """
    Generate single-source dataset: 1000 samples per standard (GSM/UMTS/LTE/5G NR).

    Samples are ordered by standard: 0–999 GSM, 1000–1999 UMTS,
    2000–2999 LTE, 3000–3999 5G NR.

    Args:
        output_path: Path to output HDF5 file (e.g. data/rfss_single.h5)
        num_samples_per_standard: Samples per standard (default 1000)
        duration_ms: Signal duration in milliseconds
        master_seed: Master random seed
        device: PyTorch device
        show_progress: Show tqdm progress bar
        checkpoint_interval: Flush + save checkpoint every N samples
    """
    output_path = Path(output_path)
    checkpoint_path = output_path.with_suffix('.ckpt.json')
    total_samples = num_samples_per_standard * len(STANDARDS)

    start_idx = 0
    resume = False
    if checkpoint_path.exists() and output_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt['next_idx']
        resume = True
        print(f"Resuming single-source generation from {start_idx}/{total_samples}")

    sampler = ParameterSampler(seed=master_seed)

    with DatasetWriter(output_path, max_samples=total_samples, resume=resume) as writer:
        iterator = range(start_idx, total_samples)
        if show_progress:
            iterator = tqdm(iterator, total=total_samples - start_idx,
                            initial=start_idx, desc="Generating single-source dataset")

        for global_idx in iterator:
            standard = STANDARDS[global_idx // num_samples_per_standard]
            config = sampler.generate_single_source_config(standard, global_idx)
            sample = generate_single_source_sample(config, duration_ms=duration_ms, device=device)
            writer.write_sample(
                mixed_signal=sample['mixed_signal'],
                sources=sample['source_signals'],
                config=config,
            )

            if (global_idx + 1) % checkpoint_interval == 0:
                writer.flush()
                with open(checkpoint_path, 'w') as f:
                    json.dump({'next_idx': global_idx + 1, 'total': total_samples}, f)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nSingle-source dataset complete: {total_samples} samples")
    print(f"Output: {output_path}")


def generate_dataset(
    output_path: Path,
    num_samples: int = 100000,
    duration_ms: float = 1.0,
    master_seed: int = 42,
    device: str = 'cpu',
    show_progress: bool = True,
    checkpoint_interval: int = 1000
) -> None:
    """
    Generate complete dataset with checkpointing for safe resumption.

    Args:
        output_path: Path to output HDF5 file
        num_samples: Number of samples to generate
        duration_ms: Signal duration in milliseconds
        master_seed: Master random seed
        device: PyTorch device
        show_progress: Show progress bar
        checkpoint_interval: Save checkpoint every N samples
    """
    output_path = Path(output_path)
    checkpoint_path = output_path.with_suffix('.ckpt.json')

    # Check for existing checkpoint to resume
    start_idx = 0
    resume = False
    if checkpoint_path.exists() and output_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_idx = ckpt['next_idx']
        resume = True
        print(f"Resuming from checkpoint: {start_idx}/{num_samples} samples done")

    sampler = ParameterSampler(seed=master_seed)

    with DatasetWriter(output_path, max_samples=num_samples, resume=resume) as writer:
        iterator = range(start_idx, num_samples)
        if show_progress:
            iterator = tqdm(iterator, total=num_samples - start_idx,
                            initial=start_idx, desc="Generating dataset")

        for sample_id in iterator:
            config = sampler.generate_sample_config(sample_id)
            sample = generate_sample(config, duration_ms=duration_ms, device=device)
            writer.write_sample(
                mixed_signal=sample['mixed_signal'],
                sources=sample['source_signals'],
                config=config
            )

            # Periodic checkpoint — flush HDF5 first to prevent corruption on kill
            if (sample_id + 1) % checkpoint_interval == 0:
                writer.flush()
                with open(checkpoint_path, 'w') as f:
                    json.dump({'next_idx': sample_id + 1, 'total': num_samples}, f)

    # Remove checkpoint on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"\nDataset generation complete: {num_samples} samples")
    print(f"Output: {output_path}")


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(description='Generate RFSS dataset')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--mode', type=str, default='multi',
                        choices=['multi', 'single'],
                        help='Generation mode: multi-source (default) or single-source')
    parser.add_argument('--num-samples', type=int, default=100000,
                        help='Number of samples for multi mode (default: 100000)')
    parser.add_argument('--num-samples-per-standard', type=int, default=1000,
                        help='Samples per standard for single mode (default: 1000)')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Signal duration in ms (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Master random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='PyTorch device (default: cpu)')

    args = parser.parse_args()
    output_path = Path(args.output)

    if args.mode == 'single':
        generate_single_source_dataset(
            output_path=output_path,
            num_samples_per_standard=args.num_samples_per_standard,
            duration_ms=args.duration,
            master_seed=args.seed,
            device=args.device,
            show_progress=True,
        )
    else:
        generate_dataset(
            output_path=output_path,
            num_samples=args.num_samples,
            duration_ms=args.duration,
            master_seed=args.seed,
            device=args.device,
            show_progress=True,
        )


if __name__ == '__main__':
    main()
