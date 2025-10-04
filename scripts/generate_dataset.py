#!/usr/bin/env python3
"""
RF Signal Dataset Generator - PyTorch Format

Generates multi-standard RF signal dataset and saves as PyTorch tensors (.pt files).

Each sample contains:
- mixed_signal: The mixed RF signal
- source_signals: Dictionary of individual source signals
- labels: Signal types present
- metadata: Generation parameters

Directory structure:
    data/
        train/
            sample_000000.pt
            sample_000001.pt
            ...
        val/
            sample_000000.pt
            ...
        test/
            sample_000000.pt
            ...
        dataset_info.json
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mixing.signal_mixer import SignalMixer, InterferenceGenerator
from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator
from channel_models.basic_channels import ChannelSimulator


class RFSignalDatasetGenerator:
    """Generate RF signal dataset in PyTorch format"""

    def __init__(self, sample_rate=30.72e6, signal_duration=0.01):
        """
        Args:
            sample_rate: Sampling rate in Hz (default: 30.72 MHz)
            signal_duration: Duration of each signal in seconds (default: 10ms)
        """
        self.sample_rate = sample_rate
        self.signal_duration = signal_duration
        self.num_samples = int(sample_rate * signal_duration)

    def generate_single_standard_sample(self, standard=None, seed=None):
        """
        Generate a single-standard sample

        Returns:
            dict with 'mixed_signal', 'source_signals', 'labels', 'metadata'
        """
        if seed is not None:
            np.random.seed(seed)

        if standard is None:
            standard = np.random.choice(['GSM', 'UMTS', 'LTE', '5G_NR'])

        # Generate signal
        if standard == 'GSM':
            gen = GSMGenerator(sample_rate=self.sample_rate,
                             duration=self.signal_duration)
            signal = gen.generate_baseband()
            params = {'standard': 'GSM', 'band': gen.band}

        elif standard == 'UMTS':
            spreading_factor = np.random.choice([64, 128, 256])
            gen = UMTSGenerator(sample_rate=self.sample_rate,
                              duration=self.signal_duration,
                              spreading_factor=spreading_factor)
            signal = gen.generate_baseband()
            params = {'standard': 'UMTS', 'spreading_factor': spreading_factor}

        elif standard == 'LTE':
            bandwidth = np.random.choice([10, 15, 20])
            modulation = np.random.choice(['16QAM', '64QAM'])
            gen = LTEGenerator(sample_rate=self.sample_rate,
                             duration=self.signal_duration,
                             bandwidth=bandwidth,
                             modulation=modulation)
            signal = gen.generate_baseband()
            params = {'standard': 'LTE', 'bandwidth': bandwidth,
                     'modulation': modulation}

        else:  # 5G_NR
            bandwidth = np.random.choice([20, 50, 100])
            modulation = np.random.choice(['64QAM', '256QAM'])
            numerology = np.random.choice([0, 1])
            gen = NRGenerator(sample_rate=self.sample_rate,
                            duration=self.signal_duration,
                            bandwidth=bandwidth,
                            modulation=modulation,
                            numerology=numerology)
            signal = gen.generate_baseband()
            params = {'standard': '5G_NR', 'bandwidth': bandwidth,
                     'modulation': modulation, 'numerology': numerology}

        # Apply channel effects (30% probability)
        if np.random.random() < 0.3:
            channel = ChannelSimulator(self.sample_rate)
            if np.random.random() < 0.5:
                signal = channel.add_rayleigh_fading(signal, max_doppler=50)
                params['channel'] = 'rayleigh_fading'
            else:
                signal = channel.add_multipath(signal, num_paths=3)
                params['channel'] = 'multipath'

        return {
            'mixed_signal': torch.from_numpy(signal),
            'source_signals': {standard: torch.from_numpy(signal)},
            'labels': [standard],
            'metadata': params
        }

    def generate_two_standard_coexistence(self, seed=None):
        """Generate a sample with two coexisting standards"""
        if seed is not None:
            np.random.seed(seed)

        # Select two different standards
        standards = np.random.choice(['GSM', 'UMTS', 'LTE', '5G_NR'],
                                   size=2, replace=False)

        mixer = SignalMixer(self.sample_rate)
        source_signals = {}
        params = {'scenario': 'two_standard_coexistence', 'standards': []}

        for standard in standards:
            # Generate signal
            if standard == 'GSM':
                gen = GSMGenerator(sample_rate=self.sample_rate,
                                 duration=self.signal_duration)
                sig = gen.generate_baseband()
                carrier_freq = 900e6

            elif standard == 'UMTS':
                gen = UMTSGenerator(sample_rate=self.sample_rate,
                                  duration=self.signal_duration)
                sig = gen.generate_baseband()
                carrier_freq = 2.1e9

            elif standard == 'LTE':
                gen = LTEGenerator(sample_rate=self.sample_rate,
                                 duration=self.signal_duration,
                                 bandwidth=20)
                sig = gen.generate_baseband()
                carrier_freq = 1.8e9

            else:  # 5G_NR
                gen = NRGenerator(sample_rate=self.sample_rate,
                                duration=self.signal_duration,
                                bandwidth=50)
                sig = gen.generate_baseband()
                carrier_freq = 3.5e9

            power_db = np.random.uniform(-3, 3)
            mixer.add_signal(sig, carrier_freq=carrier_freq, power_db=power_db,
                           label=standard)

            source_signals[standard] = torch.from_numpy(sig)
            params['standards'].append({
                'type': standard,
                'carrier_freq': carrier_freq,
                'power_db': power_db
            })

        mixed_signal, _ = mixer.mix_signals(duration=self.signal_duration)

        return {
            'mixed_signal': torch.from_numpy(mixed_signal),
            'source_signals': source_signals,
            'labels': list(standards),
            'metadata': params
        }

    def generate_multi_standard_interference(self, seed=None):
        """Generate a sample with 3-4 standards plus interference"""
        if seed is not None:
            np.random.seed(seed)

        num_standards = np.random.choice([3, 4])
        standards = np.random.choice(['GSM', 'UMTS', 'LTE', '5G_NR'],
                                   size=num_standards, replace=False)

        mixer = SignalMixer(self.sample_rate)
        source_signals = {}
        params = {'scenario': 'multi_standard_interference', 'standards': []}

        for standard in standards:
            # Simplified signal generation
            if standard == 'GSM':
                gen = GSMGenerator(sample_rate=self.sample_rate,
                                 duration=self.signal_duration)
                sig = gen.generate_baseband()
                carrier_freq = 900e6
            elif standard == 'UMTS':
                gen = UMTSGenerator(sample_rate=self.sample_rate,
                                  duration=self.signal_duration)
                sig = gen.generate_baseband()
                carrier_freq = 2.1e9
            elif standard == 'LTE':
                gen = LTEGenerator(sample_rate=self.sample_rate,
                                 duration=self.signal_duration,
                                 bandwidth=15)
                sig = gen.generate_baseband()
                carrier_freq = 1.8e9
            else:  # 5G_NR
                gen = NRGenerator(sample_rate=self.sample_rate,
                                duration=self.signal_duration,
                                bandwidth=20)
                sig = gen.generate_baseband()
                carrier_freq = 3.5e9

            power_db = np.random.uniform(-5, 2)
            mixer.add_signal(sig, carrier_freq=carrier_freq, power_db=power_db,
                           label=standard)
            source_signals[standard] = torch.from_numpy(sig)
            params['standards'].append({
                'type': standard,
                'carrier_freq': carrier_freq,
                'power_db': power_db
            })

        # Add interference (50% probability)
        if np.random.random() < 0.5:
            interf_freq = np.random.uniform(0.5e9, 4e9)
            interf = InterferenceGenerator.generate_cw_tone(
                self.sample_rate, self.signal_duration,
                freq=interf_freq, power_db=-10
            )
            mixer.add_signal(interf, carrier_freq=interf_freq,
                           power_db=-10, label='interference')
            params['interference'] = {'type': 'cw_tone', 'freq': interf_freq}

        mixed_signal, _ = mixer.mix_signals(duration=self.signal_duration)

        return {
            'mixed_signal': torch.from_numpy(mixed_signal),
            'source_signals': source_signals,
            'labels': list(standards),
            'metadata': params
        }

    def generate_sample(self, sample_id, scenario=None):
        """Generate a single sample based on scenario"""
        if scenario is None:
            scenario = np.random.choice([
                'single_standard',
                'two_standard_coexistence',
                'multi_standard_interference'
            ], p=[0.4, 0.4, 0.2])

        if scenario == 'single_standard':
            return self.generate_single_standard_sample(seed=sample_id)
        elif scenario == 'two_standard_coexistence':
            return self.generate_two_standard_coexistence(seed=sample_id)
        else:
            return self.generate_multi_standard_interference(seed=sample_id)


def generate_dataset(output_dir, num_train=1000, num_val=100, num_test=50):
    """
    Generate complete dataset with train/val/test splits

    Args:
        output_dir: Output directory
        num_train: Number of training samples
        num_val: Number of validation samples
        num_test: Number of test samples
    """
    output_dir = Path(output_dir)

    # Create directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'

    for dir in [train_dir, val_dir, test_dir]:
        dir.mkdir(parents=True, exist_ok=True)

    generator = RFSignalDatasetGenerator()

    dataset_info = {
        'created': datetime.now().isoformat(),
        'sample_rate': generator.sample_rate,
        'signal_duration': generator.signal_duration,
        'num_samples_per_signal': generator.num_samples,
        'num_train': num_train,
        'num_val': num_val,
        'num_test': num_test,
        'total_samples': num_train + num_val + num_test
    }

    # Generate training set
    print(f"Generating {num_train} training samples...")
    for i in tqdm(range(num_train)):
        sample = generator.generate_sample(i)
        torch.save(sample, train_dir / f'sample_{i:06d}.pt')

    # Generate validation set
    print(f"Generating {num_val} validation samples...")
    for i in tqdm(range(num_val)):
        sample = generator.generate_sample(i + num_train)
        torch.save(sample, val_dir / f'sample_{i:06d}.pt')

    # Generate test set
    print(f"Generating {num_test} test samples...")
    for i in tqdm(range(num_test)):
        sample = generator.generate_sample(i + num_train + num_val)
        torch.save(sample, test_dir / f'sample_{i:06d}.pt')

    # Save dataset info
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\nDataset generated successfully!")
    print(f"Location: {output_dir}")
    print(f"Train: {num_train} samples")
    print(f"Val: {num_val} samples")
    print(f"Test: {num_test} samples")
    print(f"Total: {num_train + num_val + num_test} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RF signal dataset')
    parser.add_argument('--output', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--train', type=int, default=1000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--val', type=int, default=100,
                       help='Number of validation samples (default: 100)')
    parser.add_argument('--test', type=int, default=50,
                       help='Number of test samples (default: 50)')

    args = parser.parse_args()

    generate_dataset(args.output, args.train, args.val, args.test)
