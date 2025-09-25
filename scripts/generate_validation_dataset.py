#!/usr/bin/env python3
"""
Generate validation dataset matching paper descriptions
Based on paper claims:
- Single Standard Signals (20,000 samples): 5,000 each for GSM/UMTS/LTE/5G
- Multi-Standard Coexistence (25,000 samples): GSM+LTE, UMTS+LTE, LTE+5G combinations  
- Complex Interference Scenarios (7,847 samples): 3-4 simultaneous standards

For validation, we'll generate a smaller representative dataset: 1000 samples total
"""
import sys
import os
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator
from mixing.signal_mixer import SignalMixer, InterferenceGenerator
from channel_models.basic_channels import ChannelSimulator


def generate_single_standard_samples(num_per_standard=50):
    """Generate single standard signal samples"""
    print("=== Generating Single Standard Signals ===")
    
    sample_rate = 30.72e6  # 30.72 MHz
    duration = 0.01  # 10 ms
    samples = []
    labels = []
    
    # GSM samples
    print(f"Generating {num_per_standard} GSM samples...")
    for i in tqdm(range(num_per_standard)):
        gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        signal = gsm_gen.generate_baseband()
        
        samples.append(signal)
        labels.append({
            'type': 'single_standard',
            'standard': 'GSM',
            'sample_rate': sample_rate,
            'duration': duration,
            'power_db': 0
        })
    
    # UMTS samples  
    print(f"Generating {num_per_standard} UMTS samples...")
    for i in tqdm(range(num_per_standard)):
        umts_gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
        signal = umts_gen.generate_baseband()
        
        samples.append(signal)
        labels.append({
            'type': 'single_standard',
            'standard': 'UMTS',
            'sample_rate': sample_rate,
            'duration': duration,
            'power_db': 0
        })
    
    # LTE samples
    print(f"Generating {num_per_standard} LTE samples...")
    for i in tqdm(range(num_per_standard)):
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, 
                              bandwidth=20, modulation='64QAM')
        signal = lte_gen.generate_baseband()
        
        samples.append(signal)
        labels.append({
            'type': 'single_standard',
            'standard': 'LTE',
            'sample_rate': sample_rate,
            'duration': duration,
            'bandwidth': 20,
            'modulation': '64QAM',
            'power_db': 0
        })
    
    # 5G NR samples
    print(f"Generating {num_per_standard} 5G NR samples...")
    for i in tqdm(range(num_per_standard)):
        nr_gen = NRGenerator(sample_rate=sample_rate, duration=duration,
                            bandwidth=50, modulation='256QAM', numerology=1)
        signal = nr_gen.generate_baseband()
        
        samples.append(signal)
        labels.append({
            'type': 'single_standard', 
            'standard': '5G_NR',
            'sample_rate': sample_rate,
            'duration': duration,
            'bandwidth': 50,
            'modulation': '256QAM',
            'numerology': 1,
            'power_db': 0
        })
    
    return samples, labels


def generate_coexistence_samples(num_samples=150):
    """Generate multi-standard coexistence scenarios"""
    print("=== Generating Multi-Standard Coexistence ===")
    
    sample_rate = 30.72e6
    duration = 0.01
    samples = []
    labels = []
    
    # GSM + LTE coexistence (50 samples)
    print(f"Generating GSM+LTE coexistence samples...")
    for i in tqdm(range(50)):
        mixer = SignalMixer(sample_rate)
        
        # Generate individual signals
        gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        gsm_signal = gsm_gen.generate_baseband()
        
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=20)
        lte_signal = lte_gen.generate_baseband()
        
        # Mix signals with different frequencies and powers
        mixer.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
        mixer.add_signal(lte_signal, carrier_freq=1.8e9, power_db=-3, label='LTE')
        
        mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
        
        samples.append(mixed_signal)
        labels.append({
            'type': 'coexistence',
            'scenario': 'GSM_LTE',
            'standards': ['GSM', 'LTE'],
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        })
    
    # UMTS + LTE coexistence (50 samples)
    print(f"Generating UMTS+LTE coexistence samples...")
    for i in tqdm(range(50)):
        mixer = SignalMixer(sample_rate)
        
        umts_gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
        umts_signal = umts_gen.generate_baseband()
        
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=10)
        lte_signal = lte_gen.generate_baseband()
        
        mixer.add_signal(umts_signal, carrier_freq=2.1e9, power_db=0, label='UMTS')
        mixer.add_signal(lte_signal, carrier_freq=2.6e9, power_db=-2, label='LTE')
        
        mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
        
        samples.append(mixed_signal)
        labels.append({
            'type': 'coexistence',
            'scenario': 'UMTS_LTE', 
            'standards': ['UMTS', 'LTE'],
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        })
    
    # LTE + 5G NR coexistence (50 samples)
    print(f"Generating LTE+5G coexistence samples...")
    for i in tqdm(range(50)):
        mixer = SignalMixer(sample_rate)
        
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=15)
        lte_signal = lte_gen.generate_baseband()
        
        nr_gen = NRGenerator(sample_rate=sample_rate, duration=duration, bandwidth=30)
        nr_signal = nr_gen.generate_baseband()
        
        mixer.add_signal(lte_signal, carrier_freq=2.1e9, power_db=0, label='LTE')
        mixer.add_signal(nr_signal, carrier_freq=3.5e9, power_db=-1, label='5G_NR')
        
        mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
        
        samples.append(mixed_signal)
        labels.append({
            'type': 'coexistence',
            'scenario': 'LTE_5G',
            'standards': ['LTE', '5G_NR'],
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        })
    
    return samples, labels


def generate_complex_interference_samples(num_samples=50):
    """Generate complex interference scenarios with 3-4 standards"""
    print("=== Generating Complex Interference Scenarios ===")
    
    sample_rate = 30.72e6
    duration = 0.01
    samples = []
    labels = []
    
    for i in tqdm(range(num_samples)):
        mixer = SignalMixer(sample_rate)
        
        # Generate 3-4 simultaneous standards
        num_standards = np.random.choice([3, 4])
        
        # Always include LTE as primary signal
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=20)
        lte_signal = lte_gen.generate_baseband()
        mixer.add_signal(lte_signal, carrier_freq=2.1e9, power_db=0, label='LTE_Primary')
        
        standards_used = ['LTE']
        
        # Add additional standards randomly
        available_standards = ['GSM', 'UMTS', '5G_NR']
        np.random.shuffle(available_standards)
        
        for j, std in enumerate(available_standards[:num_standards-1]):
            if std == 'GSM':
                gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
                signal = gen.generate_baseband()
                mixer.add_signal(signal, carrier_freq=900e6, power_db=-10, label='GSM')
            elif std == 'UMTS':
                gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
                signal = gen.generate_baseband()
                mixer.add_signal(signal, carrier_freq=1.9e9, power_db=-8, label='UMTS')
            elif std == '5G_NR':
                gen = NRGenerator(sample_rate=sample_rate, duration=duration, bandwidth=40)
                signal = gen.generate_baseband()
                mixer.add_signal(signal, carrier_freq=3.5e9, power_db=-5, label='5G_NR')
            
            standards_used.append(std)
        
        # Add narrowband interference
        interference = InterferenceGenerator.generate_narrowband_noise(
            sample_rate, duration, center_freq=0, bandwidth=1e6, power_db=-20
        )
        mixer.add_signal(interference, carrier_freq=2.05e9, power_db=-20, label='Narrowband_Interference')
        
        mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
        
        samples.append(mixed_signal)
        labels.append({
            'type': 'complex_interference',
            'num_standards': num_standards,
            'standards': standards_used,
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        })
    
    return samples, labels


def apply_channel_effects(samples, labels, effect_probability=0.3):
    """Apply realistic channel effects to some samples"""
    print("=== Applying Channel Effects ===")
    
    processed_samples = []
    processed_labels = []
    sample_rate = 30.72e6
    
    for i, (sample, label) in enumerate(tqdm(zip(samples, labels))):
        if np.random.random() < effect_probability:
            # Apply channel effects
            channel = ChannelSimulator(sample_rate)
            
            # Randomly choose effects
            effects = []
            if np.random.random() < 0.5:
                channel.add_awgn(snr_db=np.random.uniform(10, 25))
                effects.append('AWGN')
            
            if np.random.random() < 0.3:
                channel.add_rayleigh_fading(doppler_freq=np.random.uniform(10, 100))
                effects.append('Rayleigh')
            
            if np.random.random() < 0.2:
                channel.add_multipath()
                effects.append('Multipath')
            
            processed_sample = channel.apply(sample)
            
            # Update label
            new_label = label.copy()
            new_label['channel_effects'] = effects
            
            processed_samples.append(processed_sample)
            processed_labels.append(new_label)
        else:
            processed_samples.append(sample)
            processed_labels.append(label)
    
    return processed_samples, processed_labels


def save_dataset(samples, labels, filename='validation_dataset.h5'):
    """Save dataset in HDF5 format"""
    print(f"=== Saving Dataset: {filename} ===")
    
    output_dir = Path('data/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    with h5py.File(filepath, 'w') as f:
        # Save signals
        signals_group = f.create_group('signals')
        for i, sample in enumerate(samples):
            signals_group.create_dataset(f'signal_{i:06d}', data=sample)
        
        # Save metadata
        metadata_group = f.create_group('metadata')
        for i, label in enumerate(labels):
            sample_group = metadata_group.create_group(f'sample_{i:06d}')
            
            # Save basic metadata
            for key, value in label.items():
                if key == 'mixing_info' and value:
                    # Handle mixing info separately
                    mixing_group = sample_group.create_group('mixing_info')
                    for mix_key, mix_value in value.items():
                        if isinstance(mix_value, (list, np.ndarray)):
                            mixing_group.create_dataset(mix_key, data=str(mix_value))
                        else:
                            mixing_group.attrs[mix_key] = str(mix_value)
                elif isinstance(value, (list, np.ndarray)):
                    sample_group.create_dataset(key, data=[str(v) for v in value])
                else:
                    sample_group.attrs[key] = str(value)
        
        # Dataset statistics
        f.attrs['total_samples'] = len(samples)
        f.attrs['sample_rate'] = 30.72e6
        f.attrs['duration'] = 0.01
        f.attrs['generation_time'] = str(np.datetime64('now'))
    
    print(f"Dataset saved: {filepath}")
    print(f"Total samples: {len(samples)}")
    print(f"File size: {filepath.stat().st_size / (1024*1024):.1f} MB")


def main():
    """Main dataset generation function"""
    print("RFSS Validation Dataset Generation")
    print("=" * 50)
    print("Generating small-scale validation dataset...")
    
    all_samples = []
    all_labels = []
    
    # Generate single standard samples (200 total: 50 each)
    samples, labels = generate_single_standard_samples(num_per_standard=50)
    all_samples.extend(samples)
    all_labels.extend(labels)
    
    # Generate coexistence samples (150 total)
    samples, labels = generate_coexistence_samples(num_samples=150)
    all_samples.extend(samples)
    all_labels.extend(labels)
    
    # Generate complex interference samples (50 total)
    samples, labels = generate_complex_interference_samples(num_samples=50)
    all_samples.extend(samples)
    all_labels.extend(labels)
    
    # Apply channel effects to some samples
    all_samples, all_labels = apply_channel_effects(all_samples, all_labels)
    
    # Save dataset
    save_dataset(all_samples, all_labels, 'rfss_validation_dataset.h5')
    
    # Generate summary statistics
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    type_counts = {}
    standard_counts = {}
    
    for label in all_labels:
        label_type = label['type']
        type_counts[label_type] = type_counts.get(label_type, 0) + 1
        
        if label_type == 'single_standard':
            std = label['standard']
            standard_counts[std] = standard_counts.get(std, 0) + 1
    
    print("Sample distribution by type:")
    for sample_type, count in type_counts.items():
        print(f"  {sample_type}: {count}")
    
    print("\nSingle standard distribution:")
    for standard, count in standard_counts.items():
        print(f"  {standard}: {count}")
    
    print(f"\nTotal samples: {len(all_samples)}")
    print(f"Sample rate: 30.72 MHz")
    print(f"Duration per sample: 10 ms")
    
    return len(all_samples)


if __name__ == "__main__":
    total_samples = main()
    print(f"\n✓ Successfully generated {total_samples} validation samples")