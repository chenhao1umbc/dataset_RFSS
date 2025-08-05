# Getting Started with RF Signal Source Separation Dataset

## Overview

The RF Signal Source Separation (RFSS) dataset provides realistic wireless communication signals for research in signal processing, interference mitigation, and source separation algorithms. This guide will help you get started with generating and using the dataset.

## Installation

### Prerequisites

- Python 3.13 or higher
- NumPy, SciPy, and other scientific computing libraries
- Optional: GPU support for large-scale generation

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dataset_RFSS.git
cd dataset_RFSS

# Install dependencies
pip install -e .

# Alternative: use uv for faster installation
uv pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,docs,jupyter]"

# Run tests to verify installation
pytest tests/
```

## Quick Start

### Generate Your First Signal

```python
import numpy as np
from src.signal_generation.lte_generator import LTEGenerator

# Create a 4G LTE signal generator
lte_gen = LTEGenerator(
    sample_rate=30.72e6,  # 30.72 MHz sampling rate
    duration=0.01,        # 10 ms signal duration
    bandwidth=20,         # 20 MHz LTE bandwidth
    modulation='64QAM'    # 64-QAM modulation
)

# Generate the baseband signal
signal = lte_gen.generate_baseband()
metadata = lte_gen.get_metadata()

print(f"Generated {len(signal)} samples")
print(f"Signal power: {np.mean(np.abs(signal)**2):.6f}")
```

### Create a Multi-Standard Scenario

```python
from src.signal_generation.gsm_generator import GSMGenerator
from src.signal_generation.lte_generator import LTEGenerator
from src.mixing.signal_mixer import SignalMixer

# Generate individual signals
gsm_gen = GSMGenerator(sample_rate=10e6, duration=0.005)
lte_gen = LTEGenerator(sample_rate=30.72e6, duration=0.005)

gsm_signal = gsm_gen.generate_baseband()
lte_signal = lte_gen.generate_baseband()

# Mix signals at different frequencies
mixer = SignalMixer(sample_rate=30.72e6)
mixer.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
mixer.add_signal(lte_signal, carrier_freq=1.8e9, power_db=-3, label='LTE')

# Generate mixed signal
mixed_signal, info = mixer.mix_signals(duration=0.005)
print(f"Mixed {info['num_signals']} signals")
```

### Apply Channel Effects

```python
from src.channel_models.basic_channels import ChannelSimulator

# Create realistic channel conditions
channel = ChannelSimulator(sample_rate=30.72e6)

# Urban mobile scenario: multipath + Rayleigh fading + noise
channel.add_multipath().add_rayleigh_fading(doppler_hz=200).add_awgn(snr_db=10)

# Apply channel effects
received_signal = channel.apply(mixed_signal)
```

### Validate Signal Quality

```python
from src.validation.signal_metrics import ValidationReport

# Create validation report
validator = ValidationReport()
report = validator.generate_signal_report(
    signal_data=received_signal,
    sample_rate=30.72e6,
    signal_type='LTE'
)

# Print detailed report
validator.print_report(report)

# Check standards compliance
if report['validation_results'] and report['validation_results']['overall_valid']:
    print("✓ Signal meets standards requirements")
else:
    print("✗ Signal validation failed")
```

## Dataset Structure

The generated dataset follows this organization:

```
data/
├── raw/                 # Raw generated signals
├── processed/           # Processed signals with metadata
│   ├── GSM_baseband.npy
│   ├── LTE_baseband.npy
│   ├── NR_baseband.npy
│   ├── UMTS_baseband.npy
│   ├── All_Standards_mixed.npy
│   ├── All_Standards_metadata.npy
│   └── ...
└── datasets/            # Organized dataset splits
    ├── train/
    ├── validation/
    └── test/
```

### File Formats

- **Signal Files**: NumPy binary format (`.npy`)
  - Complex float64 data type
  - Time-domain baseband representation
  - Normalized to unit average power

- **Metadata Files**: Python pickle format (`.npy` with `allow_pickle=True`)
  - Signal parameters and generation settings
  - Validation results and quality metrics
  - Channel model parameters

## Common Use Cases

### 1. Source Separation Research

```python
# Generate mixed multi-standard signal
from examples.complete_demo import generate_all_standards, create_complex_scenarios

# Generate individual standards
signals, sample_rate = generate_all_standards()

# Create coexistence scenario
scenarios = create_complex_scenarios(signals, sample_rate)
mixed_signal = scenarios['All_Standards']['signal']

# Your source separation algorithm here
separated_signals = your_separation_algorithm(mixed_signal)
```

### 2. MIMO System Testing

```python
from src.mimo.mimo_channel import MIMOSystemSimulator

# Create 4x4 MIMO system
mimo = MIMOSystemSimulator(num_tx=4, num_rx=4, correlation='medium')

# Test different processing techniques
for method in ['zf', 'mmse', 'ml']:
    rx_signals, H = mimo.simulate_transmission(
        tx_signals=mimo_input,
        precoding=method,
        snr_db=15
    )
    
    metrics = mimo.calculate_performance_metrics(mimo_input, rx_signals)
    print(f"{method.upper()}: SNR = {metrics['snr_measured']:.2f} dB")
```

### 3. Machine Learning Training Data

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Load existing dataset
signals = []
labels = []

for signal_type in ['GSM', 'UMTS', 'LTE', 'NR']:
    signal_data = np.load(f'data/processed/{signal_type}_baseband.npy')
    signals.append(signal_data)
    labels.append(signal_type)

# Prepare ML dataset
X = np.array(signals)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Your ML model training here
model.fit(X_train, y_train)
```

## Advanced Features

### Custom Signal Generation

```python
from src.signal_generation.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def __init__(self, sample_rate, duration, custom_param):
        super().__init__(sample_rate, duration)
        self.custom_param = custom_param
    
    def generate_baseband(self):
        # Your custom signal generation logic
        t = np.arange(self.num_samples) / self.sample_rate
        signal = np.exp(1j * 2 * np.pi * self.custom_param * t)
        return signal
```

### Batch Processing

```python
import os
from pathlib import Path

def generate_dataset_batch(output_dir, num_samples=1000):
    """Generate large dataset in batches"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        # Generate random parameters
        bandwidth = np.random.choice([5, 10, 15, 20])
        snr_db = np.random.uniform(0, 30)
        
        # Generate signal
        lte_gen = LTEGenerator(bandwidth=bandwidth)
        signal = lte_gen.generate_baseband()
        
        # Apply random channel
        channel = ChannelSimulator(lte_gen.sample_rate)
        channel.add_awgn(snr_db)
        received_signal = channel.apply(signal)
        
        # Save with metadata
        np.save(f'{output_dir}/signal_{i:06d}.npy', received_signal)
        
        if i % 100 == 0:
            print(f"Generated {i} samples...")
```

### Performance Optimization

```python
# Use GPU acceleration (if available)
import cupy as cp  # Optional: for GPU arrays

# Vectorized generation for multiple signals
def generate_batch_signals(params_list):
    signals = []
    for params in params_list:
        gen = LTEGenerator(**params)
        signal = gen.generate_baseband()
        signals.append(signal)
    
    return np.array(signals)

# Parallel processing
from multiprocessing import Pool

def parallel_generation(param_chunks):
    with Pool() as pool:
        results = pool.map(generate_batch_signals, param_chunks)
    
    return np.concatenate(results)
```

## Troubleshooting

### Common Issues

1. **Memory Issues with Large Datasets**
   ```python
   # Process in smaller chunks
   chunk_size = 1000  # samples per chunk
   for i in range(0, total_samples, chunk_size):
       chunk_data = process_chunk(i, min(i + chunk_size, total_samples))
       save_chunk(chunk_data, f'chunk_{i//chunk_size:04d}.npy')
   ```

2. **Validation Failures**
   ```python
   # Check signal parameters
   print(f"Signal length: {len(signal)}")
   print(f"Sample rate: {sample_rate} Hz")
   print(f"Duration: {len(signal) / sample_rate:.6f} s")
   print(f"Power: {np.mean(np.abs(signal)**2):.6f}")
   ```

3. **Inconsistent Results**
   ```python
   # Set random seeds for reproducibility
   np.random.seed(42)
   import random
   random.seed(42)
   ```

### Performance Tips

- Use appropriate sample rates for each signal type
- Process signals in batches for large datasets
- Consider using GPU acceleration for intensive computations
- Save intermediate results to avoid recomputation

## Next Steps

- Explore the [API Reference](../api/api_reference.md) for detailed function documentation
- Check out [Advanced Usage](advanced_usage.md) for complex scenarios
- See [Examples](../../examples/) directory for complete working code
- Read the [Technical Specifications](../api/technical_specifications.md) for implementation details

## Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Look at example code in the `examples/` directory
- Report issues on the project's GitHub repository
- Join the discussion in our research community