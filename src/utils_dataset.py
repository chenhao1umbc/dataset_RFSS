"""
Dataset generation utilities for RFSS project.

Implements parameter sampling and dataset generation. Signal parameters
are derived from 3GPP specifications:
- 5G NR sample rates: 3GPP TS 38.211 §4.3.1, TS 38.104 §5.4
- LTE bandwidths and sample rates: 3GPP TS 36.211 §5.6, TS 36.104 §5.6
- TDL channel models: 3GPP TR 38.901 §7.7.2 Tables 7.7.2-1 to 7.7.2-5

Dataset specifications per paper/dataset_parameters.md:
- 100k samples total
- 1 ms signal duration
- Correlated CFO/SFO impairments
- 5G NR: μ=1, μ=3 only
- HDF5 storage with gzip compression
"""

import warnings
import torch
import h5py
import numpy as np
from typing import Dict, Any, List, Tuple
from pathlib import Path
import json
from datetime import datetime


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


# Parameter distributions per paper/dataset_parameters.md

# Section 1: Signal Generation Parameters
STANDARDS = ['GSM', 'UMTS', 'LTE', '5G_NR']
STANDARD_WEIGHTS = [0.125, 0.125, 0.375, 0.375]  # GSM:1, UMTS:1, LTE:24, 5G:40

# 3GPP TS 36.211 §5.6 and TS 36.104 §5.6
LTE_BANDWIDTHS = [1.4, 3.0, 5.0, 10.0, 15.0, 20.0]
LTE_BW_WEIGHTS = [0.05, 0.10, 0.20, 0.30, 0.10, 0.25]

LTE_MODULATIONS = ['QPSK', '16QAM', '64QAM', '256QAM']
LTE_MOD_WEIGHTS = [0.20, 0.30, 0.35, 0.15]

NR_NUMEROLOGIES = [1, 3]  # μ=1 (30 kHz), μ=3 (120 kHz)
NR_NUM_WEIGHTS = [0.75, 0.25]

NR_BANDWIDTHS = {
    1: [10, 20, 50, 100],
    3: [50, 100]  # Restricted to sample rates <= 122.88 MHz
}
NR_BW_WEIGHTS = [0.20, 0.25, 0.30, 0.25]  # Used for μ=1 (4 options); first 2 entries used for μ=3

# Actual sample rates per (numerology, bandwidth_mhz): fft_size * subcarrier_spacing
# 3GPP TS 38.211 §4.3.1 (subcarrier spacings) and TS 38.104 §5.4 (channel bandwidths)
NR_SAMPLE_RATES = {
    (1, 10): 15.36e6,
    (1, 20): 30.72e6,
    (1, 50): 61.44e6,
    (1, 100): 122.88e6,
    (3, 50): 61.44e6,
    (3, 100): 122.88e6,
}

NR_MODULATIONS = ['QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']
NR_MOD_WEIGHTS = [0.15, 0.20, 0.30, 0.25, 0.10]

# Section 2: Channel Model Parameters
# 3GPP TR 38.901 §7.7.2 Tables 7.7.2-1 to 7.7.2-5
TDL_MODELS = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E']
TDL_WEIGHTS = [0.25, 0.20, 0.15, 0.20, 0.20]

DOPPLER_RANGES = [(0, 10), (30, 120), (150, 300), (400, 700)]  # Hz
DOPPLER_WEIGHTS = [0.30, 0.40, 0.20, 0.10]  # Pedestrian, urban, highway, high-speed rail

# Section 3: Hardware Impairment Parameters
# Oscillator error (ppm) - affects both CFO and SFO (correlated)
OSC_ERROR_RANGES = [(0.05, 0.1), (0.1, 0.5), (0.5, 2.0), (2.0, 5.0)]
OSC_ERROR_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

# I/Q imbalance
IQ_AMP_RANGES = [(0.1, 0.5), (0.5, 1.5), (1.5, 3.0)]  # dB
IQ_PHASE_RANGES = [(1, 3), (3, 6), (6, 10)]  # degrees
IQ_WEIGHTS = [0.50, 0.30, 0.20]

# DC offset
DC_OFFSET_RANGES = [(-40, -35), (-35, -32), (-32, -30)]  # dBc
DC_OFFSET_WEIGHTS = [0.50, 0.30, 0.20]

# Phase noise
PHASE_NOISE_RANGES = [(-110, -105), (-105, -100), (-100, -90)]  # dBc/Hz
PHASE_NOISE_WEIGHTS = [0.30, 0.50, 0.20]

# PA nonlinearity back-off
PA_BACKOFF_RANGES = [(7, 9), (5, 7), (3, 5)]  # dB
PA_BACKOFF_WEIGHTS = [0.40, 0.35, 0.25]

# Impairment application strategy
IMPAIRMENT_MODES = ['clean', 'single', 'multiple']
IMPAIRMENT_WEIGHTS = [0.20, 0.30, 0.50]

SINGLE_IMPAIRMENTS = ['CFO_SFO', 'IQ', 'DC', 'PN', 'PA']
SINGLE_IMP_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]

# Section 4: SNR Ranges
SNR_RANGES = [(-10, 0), (0, 10), (10, 20), (20, 30), (30, 40)]  # dB
SNR_WEIGHTS = [0.15, 0.25, 0.35, 0.20, 0.05]

# Section 5: Mixed Signal Scenarios
SOURCE_COUNTS = [2, 3, 4]
SOURCE_COUNT_WEIGHTS = [0.50, 0.35, 0.15]

MIXING_MODES = ['co-channel', 'adjacent-channel']
MIXING_MODE_WEIGHTS = [0.40, 0.60]

# Power ratio categories (SIR in dB)
POWER_RATIO_CATEGORIES = ['equal', 'moderate', 'near-far', 'extreme']
POWER_RATIO_WEIGHTS = [0.20, 0.40, 0.30, 0.10]

# Section 6: MIMO Configuration
MIMO_CONFIGS = [(1, 1), (2, 2), (4, 4)]  # (num_tx, num_rx)
MIMO_WEIGHTS = [0.50, 0.30, 0.20]

SPATIAL_CORRELATION_LEVELS = ['low', 'medium', 'high']
SPATIAL_CORR_WEIGHTS = [0.40, 0.40, 0.20]


class ParameterSampler:
    """Sample parameters according to dataset specifications."""

    def __init__(self, seed: int = 42):
        """
        Initialize parameter sampler.

        Args:
            seed: Master random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.master_seed = seed

    def set_seed(self, seed: int):
        """Set random seed for reproducible sampling."""
        self.rng = np.random.RandomState(seed)

    def sample_from_range(self, ranges: List[Tuple], weights: List[float]) -> float:
        """Sample uniformly from weighted range categories."""
        range_idx = self.rng.choice(len(ranges), p=weights)
        low, high = ranges[range_idx]
        return self.rng.uniform(low, high)

    def sample_standard(self) -> str:
        """Sample wireless standard."""
        return self.rng.choice(STANDARDS, p=STANDARD_WEIGHTS)

    def sample_signal_params(self, standard: str) -> Dict[str, Any]:
        """Sample signal generation parameters for given standard."""
        params = {'standard': standard}

        if standard == 'GSM':
            params['bandwidth_mhz'] = 0.2
            params['modulation'] = 'GMSK'
            params['sample_rate'] = 2.166e6

        elif standard == 'UMTS':
            params['bandwidth_mhz'] = 5.0
            params['modulation'] = 'WCDMA'
            params['sample_rate'] = 7.68e6

        elif standard == 'LTE':
            params['bandwidth_mhz'] = self.rng.choice(LTE_BANDWIDTHS, p=LTE_BW_WEIGHTS)
            params['modulation'] = self.rng.choice(LTE_MODULATIONS, p=LTE_MOD_WEIGHTS)
            # Sample rate per 3GPP TS 36.211 §5.6 and TS 36.104 §5.6
            bw_to_rate = {1.4: 1.92e6, 3.0: 3.84e6, 5.0: 7.68e6,
                         10.0: 15.36e6, 15.0: 23.04e6, 20.0: 30.72e6}
            params['sample_rate'] = bw_to_rate[params['bandwidth_mhz']]

        elif standard == '5G_NR':
            params['numerology'] = self.rng.choice(NR_NUMEROLOGIES, p=NR_NUM_WEIGHTS)
            bandwidths = NR_BANDWIDTHS[params['numerology']]
            bw_weights = NR_BW_WEIGHTS[:len(bandwidths)]
            bw_weights = np.array(bw_weights) / sum(bw_weights)
            params['bandwidth_mhz'] = self.rng.choice(bandwidths, p=bw_weights)
            params['modulation'] = self.rng.choice(NR_MODULATIONS, p=NR_MOD_WEIGHTS)
            params['sample_rate'] = NR_SAMPLE_RATES[(params['numerology'], params['bandwidth_mhz'])]

        return params

    def sample_channel_params(self) -> Dict[str, Any]:
        """Sample channel model parameters."""
        params = {}
        params['tdl_model'] = self.rng.choice(TDL_MODELS, p=TDL_WEIGHTS)
        params['doppler_hz'] = self.sample_from_range(DOPPLER_RANGES, DOPPLER_WEIGHTS)
        return params

    def sample_impairment_params(self) -> Dict[str, Any]:
        """Sample hardware impairment parameters."""
        params = {}

        # Determine impairment mode
        mode = self.rng.choice(IMPAIRMENT_MODES, p=IMPAIRMENT_WEIGHTS)
        params['mode'] = mode

        if mode == 'clean':
            # No impairments
            params['cfo_ppm'] = 0.0
            params['sfo_ppm'] = 0.0
            params['iq_amp_db'] = 0.0
            params['iq_phase_deg'] = 0.0
            params['dc_offset_dbc'] = -100.0  # Very low
            params['phase_noise_dbc_hz'] = -120.0  # Very low
            params['pa_backoff_db'] = 10.0  # High linearity

        else:
            # Sample oscillator error (affects both CFO and SFO - correlated!)
            osc_error = self.sample_from_range(OSC_ERROR_RANGES, OSC_ERROR_WEIGHTS)
            sign = self.rng.choice([-1, 1])
            params['cfo_ppm'] = sign * osc_error
            params['sfo_ppm'] = sign * osc_error  # Same as CFO (correlated)

            if mode == 'single':
                # Apply only one impairment type (but CFO/SFO count as one)
                imp_type = self.rng.choice(SINGLE_IMPAIRMENTS, p=SINGLE_IMP_WEIGHTS)

                if imp_type == 'CFO_SFO':
                    # Already set above
                    params['iq_amp_db'] = 0.0
                    params['iq_phase_deg'] = 0.0
                    params['dc_offset_dbc'] = -100.0
                    params['phase_noise_dbc_hz'] = -120.0
                    params['pa_backoff_db'] = 10.0

                elif imp_type == 'IQ':
                    params['cfo_ppm'] = 0.0
                    params['sfo_ppm'] = 0.0
                    params['iq_amp_db'] = self.sample_from_range(IQ_AMP_RANGES, IQ_WEIGHTS)
                    params['iq_phase_deg'] = self.sample_from_range(IQ_PHASE_RANGES, IQ_WEIGHTS)
                    params['dc_offset_dbc'] = -100.0
                    params['phase_noise_dbc_hz'] = -120.0
                    params['pa_backoff_db'] = 10.0

                elif imp_type == 'DC':
                    params['cfo_ppm'] = 0.0
                    params['sfo_ppm'] = 0.0
                    params['iq_amp_db'] = 0.0
                    params['iq_phase_deg'] = 0.0
                    params['dc_offset_dbc'] = self.sample_from_range(DC_OFFSET_RANGES, DC_OFFSET_WEIGHTS)
                    params['phase_noise_dbc_hz'] = -120.0
                    params['pa_backoff_db'] = 10.0

                elif imp_type == 'PN':
                    params['cfo_ppm'] = 0.0
                    params['sfo_ppm'] = 0.0
                    params['iq_amp_db'] = 0.0
                    params['iq_phase_deg'] = 0.0
                    params['dc_offset_dbc'] = -100.0
                    params['phase_noise_dbc_hz'] = self.sample_from_range(PHASE_NOISE_RANGES, PHASE_NOISE_WEIGHTS)
                    params['pa_backoff_db'] = 10.0

                elif imp_type == 'PA':
                    params['cfo_ppm'] = 0.0
                    params['sfo_ppm'] = 0.0
                    params['iq_amp_db'] = 0.0
                    params['iq_phase_deg'] = 0.0
                    params['dc_offset_dbc'] = -100.0
                    params['phase_noise_dbc_hz'] = -120.0
                    params['pa_backoff_db'] = self.sample_from_range(PA_BACKOFF_RANGES, PA_BACKOFF_WEIGHTS)

            else:  # mode == 'multiple'
                # Apply multiple impairments (CFO/SFO already set)
                params['iq_amp_db'] = self.sample_from_range(IQ_AMP_RANGES, IQ_WEIGHTS)
                params['iq_phase_deg'] = self.sample_from_range(IQ_PHASE_RANGES, IQ_WEIGHTS)
                params['dc_offset_dbc'] = self.sample_from_range(DC_OFFSET_RANGES, DC_OFFSET_WEIGHTS)
                params['phase_noise_dbc_hz'] = self.sample_from_range(PHASE_NOISE_RANGES, PHASE_NOISE_WEIGHTS)
                params['pa_backoff_db'] = self.sample_from_range(PA_BACKOFF_RANGES, PA_BACKOFF_WEIGHTS)

        return params

    def sample_snr(self) -> float:
        """Sample SNR in dB."""
        return self.sample_from_range(SNR_RANGES, SNR_WEIGHTS)

    def sample_mimo_config(self) -> Dict[str, Any]:
        """Sample MIMO configuration."""
        params = {}
        num_tx, num_rx = MIMO_CONFIGS[self.rng.choice(len(MIMO_CONFIGS), p=MIMO_WEIGHTS)]
        params['num_tx'] = num_tx
        params['num_rx'] = num_rx

        if num_tx > 1 or num_rx > 1:
            corr_level = self.rng.choice(SPATIAL_CORRELATION_LEVELS, p=SPATIAL_CORR_WEIGHTS)
            # Map correlation level to coefficient
            corr_map = {'low': 0.1, 'medium': 0.5, 'high': 0.8}
            params['spatial_correlation'] = corr_map[corr_level]
        else:
            params['spatial_correlation'] = 0.0

        return params

    def sample_mixing_params(self, num_sources: int) -> Dict[str, Any]:
        """Sample mixing parameters for multi-source scenario."""
        params = {}
        params['num_sources'] = num_sources
        params['mixing_mode'] = self.rng.choice(MIXING_MODES, p=MIXING_MODE_WEIGHTS)

        # Sample power ratios based on category
        power_cat = self.rng.choice(POWER_RATIO_CATEGORIES, p=POWER_RATIO_WEIGHTS)

        if power_cat == 'equal':
            params['power_ratios_db'] = [self.rng.uniform(-2, 2) for _ in range(num_sources)]
        elif power_cat == 'moderate':
            params['power_ratios_db'] = [self.rng.uniform(-10, 10) for _ in range(num_sources)]
        elif power_cat == 'near-far':
            params['power_ratios_db'] = []
            for _ in range(num_sources):
                if self.rng.rand() < 0.5:
                    params['power_ratios_db'].append(self.rng.uniform(-20, -10))
                else:
                    params['power_ratios_db'].append(self.rng.uniform(10, 20))
        else:  # extreme
            params['power_ratios_db'] = []
            for _ in range(num_sources):
                if self.rng.rand() < 0.5:
                    params['power_ratios_db'].append(self.rng.uniform(-30, -20))
                else:
                    params['power_ratios_db'].append(self.rng.uniform(20, 30))

        # Normalize so average is 0 dB
        avg_power = np.mean(params['power_ratios_db'])
        params['power_ratios_db'] = [p - avg_power for p in params['power_ratios_db']]

        # Frequency offsets for adjacent-channel mixing
        if params['mixing_mode'] == 'adjacent-channel':
            spacing = 2e6  # 2 MHz spacing
            params['frequency_offsets_hz'] = [i * spacing for i in range(-(num_sources // 2), num_sources - num_sources // 2)]
        else:
            params['frequency_offsets_hz'] = [0.0] * num_sources

        return params

    def sample_source_count(self) -> int:
        """Sample number of sources for mixing."""
        return self.rng.choice(SOURCE_COUNTS, p=SOURCE_COUNT_WEIGHTS)

    def generate_single_source_config(self, standard: str, sample_id: int) -> Dict[str, Any]:
        """
        Generate configuration for a single-source sample of a specific standard.

        Uses a seed offset of 2,000,000 to avoid collision with the multi-source
        dataset (seeds 42 … 42+99,999).

        Args:
            standard: Wireless standard ('GSM', 'UMTS', 'LTE', '5G_NR')
            sample_id: Sample index within the single-source dataset

        Returns:
            Configuration dict compatible with generate_single_source()
        """
        self.set_seed(self.master_seed + 2000000 + sample_id)
        return {
            'sample_id': sample_id,
            'seed': self.master_seed + 2000000 + sample_id,
            'num_sources': 1,
            'sources': [{
                'standard': standard,
                'signal_params': self.sample_signal_params(standard),
                'channel_params': self.sample_channel_params(),
                'impairment_params': self.sample_impairment_params(),
            }],
            'mixing_params': {
                'num_sources': 1,
                'mixing_mode': 'single',
                'power_ratios_db': [0.0],
                'frequency_offsets_hz': [0.0],
            },
            'snr_db': self.sample_snr(),
            'mimo_config': {'num_tx': 1, 'num_rx': 1, 'spatial_correlation': 0.0},
            'generation_time': datetime.utcnow().isoformat(),
        }

    def generate_sample_config(self, sample_id: int) -> Dict[str, Any]:
        """
        Generate complete configuration for one sample.

        Args:
            sample_id: Unique sample identifier (used for seeding)

        Returns:
            Dictionary with all parameters for generating the sample
        """
        # Set seed based on sample_id for reproducibility
        self.set_seed(self.master_seed + sample_id)

        config = {}
        config['sample_id'] = sample_id
        config['seed'] = self.master_seed + sample_id

        # Determine number of sources
        num_sources = self.sample_source_count()
        config['num_sources'] = num_sources

        # Sample parameters for each source
        config['sources'] = []
        for i in range(num_sources):
            source = {}
            source['standard'] = self.sample_standard()
            source['signal_params'] = self.sample_signal_params(source['standard'])
            source['channel_params'] = self.sample_channel_params()
            source['impairment_params'] = self.sample_impairment_params()
            config['sources'].append(source)

        # Mixing parameters
        config['mixing_params'] = self.sample_mixing_params(num_sources)

        # Global parameters
        config['snr_db'] = self.sample_snr()
        config['mimo_config'] = self.sample_mimo_config()

        # Generation timestamp
        config['generation_time'] = datetime.utcnow().isoformat()

        return config


class DatasetWriter:
    """Write samples to HDF5 dataset with compression."""

    # 1 ms at 122.88 MHz (max sample rate across all supported configurations)
    MAX_SIGNAL_LEN = 122880

    def __init__(self, output_path: Path, max_samples: int = 100000, resume: bool = False):
        """
        Initialize dataset writer.

        Args:
            output_path: Path to output HDF5 file
            max_samples: Maximum number of samples (for pre-allocation)
            resume: If True and file exists, open in append mode to continue generation
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples

        if resume and self.output_path.exists():
            self.h5file = h5py.File(self.output_path, 'r+')
            self.current_idx = int(self.h5file.attrs.get('actual_samples', 0))
            self.mixed_signals = self.h5file['mixed_signals']
            self.source_signals = self.h5file['source_signals']
            self.signal_lengths = self.h5file['signal_lengths']
            self.metadata = self.h5file['metadata']
        else:
            self.current_idx = 0
            self.h5file = h5py.File(self.output_path, 'w')
            max_signal_len = self.MAX_SIGNAL_LEN

            self.mixed_signals = self.h5file.create_dataset(
                'mixed_signals',
                shape=(max_samples, max_signal_len),
                dtype=np.complex64,
                compression='gzip',
                compression_opts=6,
                chunks=(1, max_signal_len)
            )
            self.source_signals = self.h5file.create_dataset(
                'source_signals',
                shape=(max_samples, 4, max_signal_len),
                dtype=np.complex64,
                compression='gzip',
                compression_opts=6,
                chunks=(1, 1, max_signal_len)
            )
            self.signal_lengths = self.h5file.create_dataset(
                'signal_lengths',
                shape=(max_samples,),
                dtype=np.int32
            )
            self.metadata = self.h5file.create_dataset(
                'metadata',
                shape=(max_samples,),
                dtype=h5py.string_dtype(encoding='utf-8')
            )
            self.h5file.attrs['version'] = '1.0'
            self.h5file.attrs['creation_time'] = datetime.utcnow().isoformat()
            self.h5file.attrs['max_samples'] = max_samples
            self.h5file.attrs['signal_duration_ms'] = 1.0
            self.h5file.attrs['format'] = 'complex64'

    def write_sample(self, mixed_signal: torch.Tensor, sources: List[torch.Tensor],
                    config: Dict[str, Any]):
        """
        Write one sample to dataset.

        Args:
            mixed_signal: Mixed received signal (complex)
            sources: List of source signals (complex)
            config: Sample configuration/metadata
        """
        if self.current_idx >= self.max_samples:
            raise ValueError(f"Dataset full: {self.max_samples} samples")

        # Convert to numpy
        mixed_np = mixed_signal.cpu().numpy()
        max_len = self.mixed_signals.shape[1]
        signal_len = min(len(mixed_np), max_len)
        if len(mixed_np) > max_len:
            warnings.warn(f"Signal length {len(mixed_np)} exceeds buffer {max_len}; truncating.")

        # Write mixed signal
        self.mixed_signals[self.current_idx, :signal_len] = mixed_np[:signal_len]
        self.signal_lengths[self.current_idx] = signal_len

        # Write source signals (pad to 4 sources with zeros)
        for i, source in enumerate(sources):
            if i < 4:
                source_np = source.cpu().numpy()
                src_len = min(len(source_np), max_len)
                self.source_signals[self.current_idx, i, :src_len] = source_np[:src_len]

        # Write metadata as JSON (convert numpy types first)
        config_serializable = convert_to_serializable(config)
        self.metadata[self.current_idx] = json.dumps(config_serializable)

        self.current_idx += 1

    def flush(self):
        """Flush HDF5 buffers to disk. Call at checkpoints to prevent data loss on crash."""
        self.h5file.attrs['actual_samples'] = self.current_idx
        self.h5file.flush()

    def close(self):
        """Close HDF5 file and finalize dataset."""
        self.h5file.attrs['actual_samples'] = self.current_idx
        self.h5file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RFSSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading RFSS HDF5 data."""

    def __init__(self, h5_path: Path, split: str = 'train'):
        """
        Initialize RFSS Dataset.

        Args:
            h5_path: Path to HDF5 dataset file
            split: Dataset split ('train', 'val', or 'test')
        """
        self.h5_path = Path(h5_path)
        self.split = split

        # Open HDF5 file (read-only)
        self.h5file = h5py.File(self.h5_path, 'r')

        # Get dataset info
        total_samples = self.h5file.attrs.get('actual_samples', self.h5file.attrs['max_samples'])

        # Calculate split indices (70/15/15 train/val/test)
        train_end = int(0.70 * total_samples)
        val_end = int(0.85 * total_samples)

        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_end
        elif split == 'val':
            self.start_idx = train_end
            self.end_idx = val_end
        else:  # test
            self.start_idx = val_end
            self.end_idx = total_samples

        self.num_samples = self.end_idx - self.start_idx

    def __len__(self) -> int:
        """Return number of samples in split."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by index.

        Args:
            idx: Sample index within split

        Returns:
            Dictionary containing mixed signal, sources, and metadata
        """
        # Map to global index
        global_idx = self.start_idx + idx

        # Load data
        signal_len = self.h5file['signal_lengths'][global_idx]
        mixed_signal = self.h5file['mixed_signals'][global_idx, :signal_len]

        # Load source signals (may have zeros for padding)
        source_signals = []
        for i in range(4):
            source = self.h5file['source_signals'][global_idx, i, :signal_len]
            if torch.any(torch.from_numpy(source) != 0):  # Non-zero source
                source_signals.append(torch.from_numpy(source))

        # Load metadata
        metadata_str = self.h5file['metadata'][global_idx]
        metadata = json.loads(metadata_str)

        return {
            'mixed_signal': torch.from_numpy(mixed_signal),
            'source_signals': source_signals,
            'metadata': metadata,
            'sample_id': metadata['sample_id']
        }

    def __del__(self):
        """Close HDF5 file on deletion."""
        if hasattr(self, 'h5file'):
            self.h5file.close()


def _collate_rfss_batch(batch):
    """Collate variable-length RFSS samples into padded batched tensors."""
    n = len(batch)
    signal_lens = [item['mixed_signal'].shape[0] for item in batch]
    max_len = max(signal_lens)
    max_sources = max(len(item['source_signals']) for item in batch)

    mixed_signals = torch.zeros(n, max_len, dtype=torch.complex64)
    source_signals = torch.zeros(n, max_sources, max_len, dtype=torch.complex64)

    for i, item in enumerate(batch):
        sig = item['mixed_signal']
        mixed_signals[i, :sig.shape[0]] = sig
        for j, source in enumerate(item['source_signals']):
            source_signals[i, j, :source.shape[0]] = source

    return {
        'mixed_signals': mixed_signals,
        'source_signals': source_signals,
        'signal_lengths': torch.tensor(signal_lens, dtype=torch.int32),
        'metadata': [item['metadata'] for item in batch],
        'sample_ids': [item['sample_id'] for item in batch]
    }


def create_dataloader(
    h5_path: Path,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader for RFSS dataset.

    Args:
        h5_path: Path to HDF5 dataset file
        split: Dataset split ('train', 'val', or 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        PyTorch DataLoader
    """
    dataset = RFSSDataset(h5_path, split=split)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_rfss_batch
    )


def validate_dataset(h5_path: Path) -> Dict[str, Any]:
    """
    Validate dataset integrity and compute statistics.

    Args:
        h5_path: Path to HDF5 dataset file

    Returns:
        Dictionary with validation results and statistics
    """
    h5file = h5py.File(h5_path, 'r')

    total_samples = h5file.attrs.get('actual_samples', h5file.attrs['max_samples'])

    stats = {
        'total_samples': total_samples,
        'signal_duration_ms': h5file.attrs['signal_duration_ms'],
        'format': h5file.attrs['format'],
        'version': h5file.attrs['version'],
        'standards': {},
        'num_sources': {},
        'mixing_modes': {},
        'mimo_configs': {},
        'snr_range': [float('inf'), float('-inf')],
        'signal_length_range': [float('inf'), float('-inf')]
    }

    # Analyze all samples
    for i in range(total_samples):
        metadata_str = h5file['metadata'][i]
        metadata = json.loads(metadata_str)

        # Count standards
        for source in metadata['sources']:
            std = source['standard']
            stats['standards'][std] = stats['standards'].get(std, 0) + 1

        # Count source counts
        num_sources = metadata['num_sources']
        stats['num_sources'][num_sources] = stats['num_sources'].get(num_sources, 0) + 1

        # Count mixing modes
        if num_sources > 1:
            mode = metadata['mixing_params']['mixing_mode']
            stats['mixing_modes'][mode] = stats['mixing_modes'].get(mode, 0) + 1

        # Count MIMO configs
        mimo = f"{metadata['mimo_config']['num_tx']}x{metadata['mimo_config']['num_rx']}"
        stats['mimo_configs'][mimo] = stats['mimo_configs'].get(mimo, 0) + 1

        # SNR range
        snr = metadata['snr_db']
        stats['snr_range'][0] = min(stats['snr_range'][0], snr)
        stats['snr_range'][1] = max(stats['snr_range'][1], snr)

        # Signal length range
        signal_len = h5file['signal_lengths'][i]
        stats['signal_length_range'][0] = min(stats['signal_length_range'][0], signal_len)
        stats['signal_length_range'][1] = max(stats['signal_length_range'][1], signal_len)

    h5file.close()

    return stats


def inspect_sample(h5_path: Path, sample_id: int) -> Dict[str, Any]:
    """
    Inspect a specific sample from dataset.

    Args:
        h5_path: Path to HDF5 dataset file
        sample_id: Sample ID to inspect

    Returns:
        Dictionary with sample data and metadata
    """
    h5file = h5py.File(h5_path, 'r')

    signal_len = h5file['signal_lengths'][sample_id]
    mixed_signal = h5file['mixed_signals'][sample_id, :signal_len]

    source_signals = []
    for i in range(4):
        source = h5file['source_signals'][sample_id, i, :signal_len]
        if torch.any(torch.from_numpy(source) != 0):
            source_signals.append(source)

    metadata_str = h5file['metadata'][sample_id]
    metadata = json.loads(metadata_str)

    h5file.close()

    return {
        'mixed_signal': mixed_signal,
        'source_signals': source_signals,
        'metadata': metadata,
        'signal_length': signal_len
    }
