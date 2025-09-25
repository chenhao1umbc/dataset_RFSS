#!/usr/bin/env python3
"""
Large-scale dataset generation for deep learning training
Generate 4K-40K samples for CNN-LSTM training with MPS acceleration
"""
import sys
import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mixing.signal_mixer import SignalMixer, InterferenceGenerator
from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator
from channel_models.basic_channels import ChannelSimulator


def generate_single_mixture(args):
    """Generate a single mixed signal sample (for parallel processing)"""
    sample_id, config = args
    
    try:
        sample_rate = config['sample_rate']
        duration = config['duration']
        scenario = config['scenario']
        
        # Set random seed for reproducibility
        np.random.seed(sample_id + config.get('base_seed', 42))
        
        if scenario == 'single_standard':
            return generate_single_standard_sample(sample_id, config)
        elif scenario == 'two_standard_coexistence':
            return generate_two_standard_sample(sample_id, config)
        elif scenario == 'multi_standard_interference':
            return generate_multi_standard_sample(sample_id, config)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
            
    except Exception as e:
        return {
            'sample_id': sample_id,
            'error': str(e),
            'mixed_signal': None,
            'components': None,
            'metadata': None
        }


def generate_single_standard_sample(sample_id, config):
    """Generate single standard sample"""
    standards = ['GSM', 'UMTS', 'LTE', '5G_NR']
    standard = standards[sample_id % len(standards)]
    
    sample_rate = config['sample_rate']
    duration = config['duration']
    
    # Generate signal based on standard
    if standard == 'GSM':
        gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        signal = gen.generate_baseband()
        carrier_freq = 900e6
    elif standard == 'UMTS':
        gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
        signal = gen.generate_baseband()
        carrier_freq = 2.1e9
    elif standard == 'LTE':
        bandwidth = np.random.choice([10, 15, 20])
        modulation = np.random.choice(['16QAM', '64QAM'])
        gen = LTEGenerator(sample_rate=sample_rate, duration=duration, 
                          bandwidth=bandwidth, modulation=modulation)
        signal = gen.generate_baseband()
        carrier_freq = np.random.choice([1.8e9, 2.1e9, 2.6e9])
    else:  # 5G_NR
        bandwidth = np.random.choice([20, 50, 100])
        modulation = np.random.choice(['64QAM', '256QAM'])
        numerology = np.random.choice([0, 1, 2])
        gen = NRGenerator(sample_rate=sample_rate, duration=duration,
                         bandwidth=bandwidth, modulation=modulation, numerology=numerology)
        signal = gen.generate_baseband()
        carrier_freq = np.random.choice([3.5e9, 28e9])
    
    # Apply channel effects (30% probability)
    if np.random.random() < 0.3:
        channel = ChannelSimulator(sample_rate)
        
        # Add random effects
        effects = []
        if np.random.random() < 0.7:
            snr_db = np.random.uniform(10, 30)
            channel.add_awgn(snr_db=snr_db)
            effects.append(f'AWGN_{snr_db:.1f}dB')
        
        if np.random.random() < 0.3:
            doppler = np.random.uniform(10, 200)
            channel.add_rayleigh_fading(doppler_freq=doppler)
            effects.append(f'Rayleigh_{doppler:.1f}Hz')
        
        signal = channel.apply(signal)
    else:
        effects = []
    
    return {
        'sample_id': sample_id,
        'mixed_signal': signal,
        'components': {standard: signal},
        'metadata': {
            'type': 'single_standard',
            'primary_standard': standard,
            'carrier_freq': carrier_freq,
            'sample_rate': sample_rate,
            'duration': duration,
            'channel_effects': effects
        }
    }


def generate_two_standard_sample(sample_id, config):
    """Generate two-standard coexistence sample"""
    combinations = [
        ('GSM', 'LTE'),
        ('UMTS', 'LTE'), 
        ('LTE', '5G_NR')
    ]
    combo = combinations[sample_id % len(combinations)]
    
    sample_rate = config['sample_rate']
    duration = config['duration']
    
    mixer = SignalMixer(sample_rate)
    components = {}
    
    for i, standard in enumerate(combo):
        # Generate signal
        if standard == 'GSM':
            gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
            signal = gen.generate_baseband()
            carrier_freq = 900e6
        elif standard == 'UMTS':
            gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
            signal = gen.generate_baseband() 
            carrier_freq = 2.1e9
        elif standard == 'LTE':
            bandwidth = np.random.choice([10, 20])
            gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=bandwidth)
            signal = gen.generate_baseband()
            carrier_freq = 1.8e9 if 'GSM' in combo else 2.6e9
        else:  # 5G_NR
            bandwidth = np.random.choice([50, 100])
            gen = NRGenerator(sample_rate=sample_rate, duration=duration, bandwidth=bandwidth)
            signal = gen.generate_baseband()
            carrier_freq = 3.5e9
        
        # Random power level
        power_db = 0 if i == 0 else np.random.uniform(-10, -1)
        
        mixer.add_signal(signal, carrier_freq=carrier_freq, 
                        power_db=power_db, label=standard)
        components[standard] = signal
    
    mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
    
    return {
        'sample_id': sample_id,
        'mixed_signal': mixed_signal,
        'components': components,
        'metadata': {
            'type': 'two_standard_coexistence',
            'standards': list(combo),
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        }
    }


def generate_multi_standard_sample(sample_id, config):
    """Generate complex multi-standard interference sample"""
    sample_rate = config['sample_rate'] 
    duration = config['duration']
    
    mixer = SignalMixer(sample_rate)
    components = {}
    
    # Always include LTE as primary
    lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=20)
    lte_signal = lte_gen.generate_baseband()
    mixer.add_signal(lte_signal, carrier_freq=2.1e9, power_db=0, label='LTE')
    components['LTE'] = lte_signal
    
    # Add 2-3 additional standards
    num_additional = np.random.choice([2, 3])
    additional_standards = np.random.choice(['GSM', 'UMTS', '5G_NR'], 
                                          size=num_additional, replace=False)
    
    for standard in additional_standards:
        if standard == 'GSM':
            gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
            signal = gen.generate_baseband()
            carrier_freq = 900e6
        elif standard == 'UMTS':
            gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
            signal = gen.generate_baseband()
            carrier_freq = 1.9e9
        else:  # 5G_NR
            gen = NRGenerator(sample_rate=sample_rate, duration=duration, bandwidth=50)
            signal = gen.generate_baseband()
            carrier_freq = 3.5e9
        
        power_db = np.random.uniform(-15, -5)
        mixer.add_signal(signal, carrier_freq=carrier_freq, 
                        power_db=power_db, label=standard)
        components[standard] = signal
    
    # Add narrowband interference
    interference = InterferenceGenerator.generate_narrowband_noise(
        sample_rate, duration, center_freq=0, bandwidth=1e6, power_db=-25
    )
    mixer.add_signal(interference, carrier_freq=2.05e9, power_db=-25, 
                    label='Narrowband_Interference')
    
    mixed_signal, mixing_info = mixer.mix_signals(duration=duration)
    
    return {
        'sample_id': sample_id,
        'mixed_signal': mixed_signal,
        'components': components,
        'metadata': {
            'type': 'multi_standard_interference',
            'standards': ['LTE'] + list(additional_standards),
            'mixing_info': mixing_info,
            'sample_rate': sample_rate,
            'duration': duration
        }
    }


def save_batch_to_hdf5(batch_results, output_file, batch_id):
    """Save batch results to HDF5 file"""
    with h5py.File(output_file, 'a') as f:  # Append mode
        
        # Create batch group if it doesn't exist
        if 'batches' not in f:
            batches_group = f.create_group('batches')
        else:
            batches_group = f['batches']
        
        batch_group = batches_group.create_group(f'batch_{batch_id:04d}')
        
        # Save signals and metadata
        signals_group = batch_group.create_group('signals')
        metadata_group = batch_group.create_group('metadata')
        components_group = batch_group.create_group('components')
        
        valid_samples = 0
        
        for result in batch_results:
            if result['mixed_signal'] is not None:
                sample_id = result['sample_id']
                
                # Save mixed signal
                signals_group.create_dataset(f'mixed_{sample_id:06d}', 
                                           data=result['mixed_signal'])
                
                # Save component signals
                comp_sample_group = components_group.create_group(f'sample_{sample_id:06d}')
                for std_name, signal in result['components'].items():
                    comp_sample_group.create_dataset(std_name, data=signal)
                
                # Save metadata
                meta_sample_group = metadata_group.create_group(f'sample_{sample_id:06d}')
                for key, value in result['metadata'].items():
                    if isinstance(value, (str, int, float)):
                        meta_sample_group.attrs[key] = value
                    else:
                        meta_sample_group.attrs[key] = str(value)
                
                valid_samples += 1
        
        # Batch statistics
        batch_group.attrs['num_samples'] = valid_samples
        batch_group.attrs['batch_id'] = batch_id
        batch_group.attrs['generation_time'] = str(datetime.now())


def generate_large_dataset(num_samples, output_file, config):
    """Generate large dataset with parallel processing"""
    print(f"Generating {num_samples} samples...")
    print(f"Output file: {output_file}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing file
    if output_path.exists():
        output_path.unlink()
    
    # Initialize HDF5 file with metadata
    with h5py.File(output_file, 'w') as f:
        f.attrs['total_samples'] = num_samples
        f.attrs['sample_rate'] = config['sample_rate']
        f.attrs['duration'] = config['duration']
        f.attrs['generation_start'] = str(datetime.now())
        f.attrs['config'] = json.dumps(config)
    
    # Generate scenario distribution
    scenario_distribution = {
        'single_standard': int(0.4 * num_samples),
        'two_standard_coexistence': int(0.4 * num_samples),
        'multi_standard_interference': int(0.2 * num_samples)
    }
    
    # Adjust for rounding
    total_assigned = sum(scenario_distribution.values())
    if total_assigned < num_samples:
        scenario_distribution['two_standard_coexistence'] += (num_samples - total_assigned)
    
    print(f"Scenario distribution: {scenario_distribution}")
    
    # Create task list
    tasks = []
    sample_id = 0
    
    for scenario, count in scenario_distribution.items():
        for _ in range(count):
            task_config = config.copy()
            task_config['scenario'] = scenario
            task_config['base_seed'] = config.get('seed', 42)
            tasks.append((sample_id, task_config))
            sample_id += 1
    
    # Process in batches for memory efficiency
    batch_size = 100
    num_batches = (len(tasks) + batch_size - 1) // batch_size
    
    print(f"Processing {num_batches} batches of {batch_size} samples...")
    
    # Use process pool for CPU-intensive signal generation
    max_workers = min(mp.cpu_count() - 1, 8)  # Leave one core free
    
    total_processed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_id in range(num_batches):
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, len(tasks))
            batch_tasks = tasks[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_id + 1}/{num_batches} "
                  f"({len(batch_tasks)} samples)...")
            
            # Submit batch tasks
            batch_results = list(tqdm(
                executor.map(generate_single_mixture, batch_tasks),
                total=len(batch_tasks),
                desc=f"Batch {batch_id + 1}"
            ))
            
            # Save batch to file
            save_batch_to_hdf5(batch_results, output_file, batch_id)
            
            # Count successful samples
            successful = sum(1 for r in batch_results if r['mixed_signal'] is not None)
            total_processed += successful
            
            print(f"  Batch {batch_id + 1}: {successful}/{len(batch_tasks)} successful")
            
            # Clean up memory
            del batch_results
            gc.collect()
    
    # Update final statistics
    with h5py.File(output_file, 'a') as f:
        f.attrs['successful_samples'] = total_processed
        f.attrs['generation_end'] = str(datetime.now())
    
    # Print final statistics
    file_size = Path(output_file).stat().st_size / (1024**3)  # GB
    
    print(f"\n{'='*60}")
    print("DATASET GENERATION COMPLETED")
    print(f"{'='*60}")
    print(f"Successful samples: {total_processed}/{num_samples}")
    print(f"Success rate: {total_processed/num_samples*100:.1f}%")
    print(f"Output file: {output_file}")
    print(f"File size: {file_size:.2f} GB")
    print(f"Average per sample: {file_size/total_processed*1000:.1f} MB")
    
    return total_processed


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Generate large RF signal dataset')
    parser.add_argument('--num_samples', type=int, default=4000,
                       help='Number of samples to generate (default: 4000)')
    parser.add_argument('--output', type=str, default='data/large_dataset/rfss_large.h5',
                       help='Output HDF5 file path')
    parser.add_argument('--sample_rate', type=float, default=30.72e6,
                       help='Sample rate in Hz (default: 30.72 MHz)')
    parser.add_argument('--duration', type=float, default=0.005,
                       help='Signal duration in seconds (default: 5ms)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'sample_rate': args.sample_rate,
        'duration': args.duration,
        'seed': args.seed
    }
    
    print("Large-scale RF Dataset Generation")
    print("="*60)
    
    # Generate dataset
    successful_samples = generate_large_dataset(
        args.num_samples, args.output, config
    )
    
    print(f"\n✅ Generated {successful_samples} samples successfully!")
    
    return successful_samples


if __name__ == "__main__":
    main()