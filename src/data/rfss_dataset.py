"""
PyTorch Dataset for RF Signal Source Separation

Provides easy loading and batching of RF signal data.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import numpy as np


class RFSSDataset(Dataset):
    """
    PyTorch Dataset for RF Signal Source Separation

    Each sample contains:
        - mixed_signal: Complex tensor of mixed RF signal
        - source_signals: Dict of individual source signals
        - labels: List of signal types present
        - metadata: Generation parameters

    Example:
        >>> dataset = RFSSDataset('data/train')
        >>> sample = dataset[0]
        >>> mixed_signal = sample['mixed_signal']  # Complex tensor
        >>> labels = sample['labels']  # ['LTE', 'GSM']
    """

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Directory containing .pt files
            transform: Optional transform to apply to signals
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Find all .pt files
        self.samples = sorted(list(self.data_dir.glob('sample_*.pt')))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir}")

        # Load dataset info if available
        info_file = self.data_dir.parent / 'dataset_info.json'
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.info = json.load(f)
        else:
            self.info = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load a single sample"""
        sample = torch.load(self.samples[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_statistics(self):
        """Compute dataset statistics"""
        print(f"Computing statistics for {len(self)} samples...")

        label_counts = {}
        scenario_counts = {}
        signal_powers = []

        for i in range(len(self)):
            sample = self[i]

            # Count labels
            for label in sample['labels']:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Count scenarios
            scenario = sample['metadata'].get('scenario', 'single_standard')
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

            # Measure power
            mixed = sample['mixed_signal'].numpy()
            power = float(np.mean(np.abs(mixed)**2))
            signal_powers.append(power)

        stats = {
            'num_samples': len(self),
            'label_distribution': label_counts,
            'scenario_distribution': scenario_counts,
            'mean_power': float(np.mean(signal_powers)),
            'std_power': float(np.std(signal_powers)),
            'sample_rate': self.info['sample_rate'] if self.info else None,
            'signal_duration': self.info['signal_duration'] if self.info else None
        }

        return stats


class ComplexToMagnitudePhase:
    """Transform: Convert complex signal to magnitude and phase channels"""

    def __call__(self, sample):
        mixed = sample['mixed_signal']

        # Convert to magnitude and phase
        mag = torch.abs(mixed)
        phase = torch.angle(mixed)

        # Stack as 2-channel tensor
        sample['mixed_signal'] = torch.stack([mag, phase], dim=0)

        # Also convert source signals if requested
        if 'source_signals' in sample:
            for key in sample['source_signals']:
                src = sample['source_signals'][key]
                mag = torch.abs(src)
                phase = torch.angle(src)
                sample['source_signals'][key] = torch.stack([mag, phase], dim=0)

        return sample


class ComplexToRealImag:
    """Transform: Convert complex signal to real and imaginary channels"""

    def __call__(self, sample):
        mixed = sample['mixed_signal']

        # Convert to real and imaginary
        real = torch.real(mixed)
        imag = torch.imag(mixed)

        # Stack as 2-channel tensor
        sample['mixed_signal'] = torch.stack([real, imag], dim=0)

        # Also convert source signals
        if 'source_signals' in sample:
            for key in sample['source_signals']:
                src = sample['source_signals'][key]
                real = torch.real(src)
                imag = torch.imag(src)
                sample['source_signals'][key] = torch.stack([real, imag], dim=0)

        return sample


class NormalizePower:
    """Transform: Normalize signal power to target value"""

    def __init__(self, target_power=1.0):
        self.target_power = target_power

    def __call__(self, sample):
        mixed = sample['mixed_signal']

        # Compute current power
        current_power = torch.mean(torch.abs(mixed)**2)

        # Scale to target power
        scale = torch.sqrt(self.target_power / current_power)
        sample['mixed_signal'] = mixed * scale

        # Also normalize source signals
        if 'source_signals' in sample:
            for key in sample['source_signals']:
                src = sample['source_signals'][key]
                src_power = torch.mean(torch.abs(src)**2)
                src_scale = torch.sqrt(self.target_power / src_power)
                sample['source_signals'][key] = src * src_scale

        return sample


def create_dataloaders(data_dir, batch_size=16, num_workers=4,
                       transform=None):
    """
    Create train/val/test dataloaders

    Args:
        data_dir: Root directory containing train/val/test folders
        batch_size: Batch size
        num_workers: Number of workers for data loading
        transform: Optional transform to apply

    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    data_dir = Path(data_dir)

    dataloaders = {}

    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split

        if split_dir.exists():
            dataset = RFSSDataset(split_dir, transform=transform)

            # Use shuffle for training, not for val/test
            shuffle = (split == 'train')

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )

            dataloaders[split] = loader

    return dataloaders


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Test RFSS Dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')

    args = parser.parse_args()

    # Test loading
    print("Testing RFSS Dataset...")

    train_dir = Path(args.data_dir) / 'train'
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        print("Run generate_dataset.py first to create the dataset.")
        exit(1)

    # Load dataset
    dataset = RFSSDataset(train_dir)

    print(f"\nDataset loaded: {len(dataset)} samples")

    # Load one sample
    sample = dataset[0]

    print("\nSample structure:")
    print(f"  mixed_signal: {sample['mixed_signal'].shape}, dtype: {sample['mixed_signal'].dtype}")
    print(f"  labels: {sample['labels']}")
    print(f"  metadata: {sample['metadata']}")

    # Compute statistics
    stats = dataset.get_statistics()

    print("\nDataset statistics:")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Label distribution: {stats['label_distribution']}")
    print(f"  Scenario distribution: {stats['scenario_distribution']}")
    print(f"  Mean power: {stats['mean_power']:.6f}")
    print(f"  Std power: {stats['std_power']:.6f}")

    # Test dataloader
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    print(f"  Batch mixed_signal shape: {batch['mixed_signal'].shape}")
    print(f"  Batch labels: {batch['labels']}")

    print("\nDataset test completed successfully!")
