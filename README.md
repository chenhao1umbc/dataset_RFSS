# RF Signal Source Separation Dataset (RFSS)

Multi-standard RF signal generation framework for machine learning research.

## What is this?

Framework for generating realistic multi-standard RF signals (2G/3G/4G/5G) with channel effects and MIMO. Designed for:
- RF signal source separation research
- Deep learning on wireless communications
- Spectrum sharing studies
- Multi-standard coexistence analysis

## Installation

```bash
git clone https://github.com/yourusername/dataset_RFSS.git
cd dataset_RFSS
pip install -e .
```

## Quick Start

### 1. Generate Dataset

```bash
# Generate small dataset (1000 train, 100 val, 50 test samples)
python scripts/generate_dataset.py --train 1000 --val 100 --test 50

# Generate larger dataset
python scripts/generate_dataset.py --train 10000 --val 1000 --test 500
```

Dataset saved to `data/` with PyTorch .pt format:
```
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
```

### 2. Validate Data Quality

```bash
# Validate signal generators are working correctly
python scripts/validate_data_quality.py

# Inspect individual signals
python scripts/inspect_data.py --standard lte

# Compare all standards
python scripts/inspect_data.py --compare
```

Check quality reports in `data/quality_reports/`

### 3. Load Data in PyTorch

```python
from src.data import RFSSDataset, create_dataloaders

# Load dataset
dataset = RFSSDataset('data/train')

# Get a sample
sample = dataset[0]
mixed_signal = sample['mixed_signal']  # Complex tensor
labels = sample['labels']  # e.g., ['LTE', 'GSM']
metadata = sample['metadata']  # Generation parameters

# Create dataloaders
dataloaders = create_dataloaders('data', batch_size=16)
train_loader = dataloaders['train']
```

## Data Format

Each `.pt` file contains:
```python
{
    'mixed_signal': torch.ComplexTensor,  # Mixed RF signal
    'source_signals': {
        'LTE': torch.ComplexTensor,       # Individual sources
        'GSM': torch.ComplexTensor,
        ...
    },
    'labels': ['LTE', 'GSM', ...],        # Standards present
    'metadata': {
        'scenario': 'two_standard_coexistence',
        'standards': [...],
        'carrier_freq': ...,
        ...
    }
}
```

## Supported Standards

- **2G (GSM)**: GMSK modulation, 200 kHz bandwidth
- **3G (UMTS)**: CDMA with spreading codes, 5 MHz bandwidth
- **4G (LTE)**: OFDM, configurable bandwidth (10/15/20 MHz), QAM modulation
- **5G (NR)**: OFDM with flexible numerology, up to 100 MHz bandwidth

## Features

### Signal Generation
- 3GPP-compliant signal generators
- Configurable parameters (bandwidth, modulation, power, etc.)
- Deterministic generation with seed control
- Realistic power normalization

### Channel Models
- Rayleigh/Rician fading
- Multipath propagation
- AWGN (Additive White Gaussian Noise)
- MIMO channel simulation (2×2, 4×4, 6×6, 8×8)

### Scenarios
- **Single Standard**: One RF technology
- **Two Standard Coexistence**: Two technologies sharing spectrum
- **Multi-Standard Interference**: 3-4 technologies + interference

### Quality Assurance
- Automated 3GPP compliance checking
- Signal quality metrics (EVM, PAPR, SNR)
- Spectral purity validation
- Statistical property verification

## Performance

Signal generation speed (on typical workstation):
- GSM: ~16 ms per 10ms signal
- UMTS: ~4 ms per 10ms signal
- LTE: ~16 ms per 10ms signal
- 5G NR: ~16 ms per 10ms signal

Memory: ~4.7 MB per 10ms signal (all standards)

## Project Structure

```
dataset_RFSS/
├── src/
│   ├── signal_generation/    # Standard-specific generators
│   ├── channel_models/        # Channel effects
│   ├── mimo/                  # MIMO processing
│   ├── mixing/                # Signal mixing
│   ├── validation/            # Quality metrics
│   └── data/                  # PyTorch dataset
├── scripts/
│   ├── generate_dataset.py    # Main dataset generation
│   ├── validate_data_quality.py  # Quality validation
│   └── inspect_data.py        # Data visualization
├── examples/                  # Usage examples
├── tests/                     # Unit tests
└── config/                    # YAML configurations
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format
black src/ scripts/ tests/

# Type check
mypy src/

# Lint
flake8 src/
```

## Data Quality Checklist

Before using generated data for ML:

1. **Run validation**: `python scripts/validate_data_quality.py`
2. **Check quality scores**: Should be >80 for all standards
3. **Inspect samples**: `python scripts/inspect_data.py`
4. **Verify power normalization**: Mean power should be ~1.0
5. **Check spectral purity**: No unexpected spurious emissions

If quality scores are low, check signal generators and parameters.

## License

CC BY 4.0

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{rfss2025,
    title={RF Signal Source Separation Dataset Generation Framework},
    author={Your Name},
    year={2025},
    url={https://github.com/yourusername/dataset_RFSS}
}
```

## Contributing

Issues and pull requests welcome.
