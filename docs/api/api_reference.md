# RF Signal Source Separation Dataset - API Reference

## Core Signal Generators

### BaseGenerator

Base class for all signal generators providing common functionality.

```python
class BaseGenerator:
    def __init__(self, sample_rate: float, duration: float)
    def generate_baseband(self) -> np.ndarray
    def get_metadata(self) -> dict
    def validate_parameters(self) -> bool
```

**Parameters:**
- `sample_rate` (float): Sampling rate in Hz
- `duration` (float): Signal duration in seconds

**Methods:**
- `generate_baseband()`: Generate complex baseband signal
- `get_metadata()`: Return signal parameters and metadata
- `validate_parameters()`: Check parameter validity

### GSMGenerator

Generate 2G GSM signals with GMSK modulation.

```python
class GSMGenerator(BaseGenerator):
    def __init__(
        self,
        sample_rate: float = 10e6,
        duration: float = 0.01,
        bt_product: float = 0.3,
        symbol_rate: float = 270833.0,
        power_db: float = 0.0
    )
```

**Parameters:**
- `bt_product` (float): Gaussian filter BT product (default: 0.3)
- `symbol_rate` (float): Symbol rate in symbols/second (default: 270.833 ksps)
- `power_db` (float): Signal power in dB (default: 0 dB)

**Example:**
```python
from signal_generation.gsm_generator import GSMGenerator

# Generate 10ms GSM signal
gsm_gen = GSMGenerator(sample_rate=10e6, duration=0.01)
signal = gsm_gen.generate_baseband()
metadata = gsm_gen.get_metadata()
```

### UMTSGenerator

Generate 3G UMTS signals with CDMA spreading.

```python
class UMTSGenerator(BaseGenerator):
    def __init__(
        self,
        sample_rate: float = 15.36e6,
        duration: float = 0.01,
        spreading_factor: int = 128,
        num_users: int = 1,
        modulation: str = 'QPSK',
        power_db: float = 0.0
    )
```

**Parameters:**
- `spreading_factor` (int): CDMA spreading factor (4-512)
- `num_users` (int): Number of simultaneous users
- `modulation` (str): Modulation scheme ('QPSK', '16QAM')
- `power_db` (float): Signal power in dB

**Example:**
```python
from signal_generation.umts_generator import UMTSGenerator

# Generate multi-user UMTS signal
umts_gen = UMTSGenerator(
    sample_rate=15.36e6,
    duration=0.01,
    spreading_factor=128,
    num_users=4
)
signal = umts_gen.generate_baseband()
```

### LTEGenerator

Generate 4G LTE signals with OFDM.

```python
class LTEGenerator(BaseGenerator):
    def __init__(
        self,
        sample_rate: float = 30.72e6,
        duration: float = 0.01,
        bandwidth: int = 20,
        modulation: str = '64QAM',
        num_antennas: int = 1,
        power_db: float = 0.0
    )
```

**Parameters:**
- `bandwidth` (int): LTE bandwidth in MHz (1.4, 3, 5, 10, 15, 20)
- `modulation` (str): Modulation scheme ('QPSK', '16QAM', '64QAM', '256QAM')
- `num_antennas` (int): Number of antenna ports
- `power_db` (float): Signal power in dB

**Example:**
```python
from signal_generation.lte_generator import LTEGenerator

# Generate 20MHz LTE signal with 256-QAM
lte_gen = LTEGenerator(
    sample_rate=30.72e6,
    bandwidth=20,
    modulation='256QAM'
)
signal = lte_gen.generate_baseband()
```

### NRGenerator

Generate 5G NR signals with flexible numerology.

```python
class NRGenerator(BaseGenerator):
    def __init__(
        self,
        sample_rate: float = 122.88e6,
        duration: float = 0.01,
        bandwidth: int = 100,
        numerology: int = 1,
        modulation: str = '256QAM',
        num_layers: int = 1,
        power_db: float = 0.0
    )
```

**Parameters:**
- `bandwidth` (int): NR bandwidth in MHz (5-100)
- `numerology` (int): Subcarrier spacing index μ (0-3)
- `modulation` (str): Modulation scheme ('QPSK', '16QAM', '64QAM', '256QAM', '1024QAM')
- `num_layers` (int): Number of spatial layers
- `power_db` (float): Signal power in dB

**Example:**
```python
from signal_generation.nr_generator import NRGenerator

# Generate 100MHz 5G NR signal
nr_gen = NRGenerator(
    sample_rate=122.88e6,
    bandwidth=100,
    numerology=1,
    modulation='1024QAM'
)
signal = nr_gen.generate_baseband()
```

## Channel Models

### ChannelSimulator

Apply realistic channel effects to signals.

```python
class ChannelSimulator:
    def __init__(self, sample_rate: float)
    def add_awgn(self, snr_db: float) -> 'ChannelSimulator'
    def add_multipath(self, delays: list = None, gains: list = None) -> 'ChannelSimulator'
    def add_rayleigh_fading(self, doppler_hz: float) -> 'ChannelSimulator'
    def add_rician_fading(self, doppler_hz: float, k_factor_db: float) -> 'ChannelSimulator'
    def apply(self, signal: np.ndarray) -> np.ndarray
```

**Methods:**
- `add_awgn()`: Add white Gaussian noise
- `add_multipath()`: Add multipath propagation
- `add_rayleigh_fading()`: Add Rayleigh fading
- `add_rician_fading()`: Add Rician fading with K-factor
- `apply()`: Apply all configured channel effects

**Example:**
```python
from channel_models.basic_channels import ChannelSimulator

# Create urban mobile channel
channel = ChannelSimulator(sample_rate=30.72e6)
channel.add_multipath().add_rayleigh_fading(200).add_awgn(10)

# Apply to signal
faded_signal = channel.apply(original_signal)
```

## MIMO Systems

### MIMOSystemSimulator

Simulate MIMO wireless systems with multiple antennas.

```python
class MIMOSystemSimulator:
    def __init__(
        self,
        num_tx: int,
        num_rx: int,
        correlation: str = 'none'
    )
    def simulate_transmission(
        self,
        tx_signals: np.ndarray,
        precoding: str = 'none',
        snr_db: float = 20
    ) -> tuple
    def calculate_performance_metrics(
        self,
        tx_signals: np.ndarray,
        rx_signals: np.ndarray
    ) -> dict
```

**Parameters:**
- `num_tx` (int): Number of transmit antennas
- `num_rx` (int): Number of receive antennas  
- `correlation` (str): Spatial correlation ('none', 'low', 'medium', 'high')

**Methods:**
- `simulate_transmission()`: Simulate MIMO transmission
- `calculate_performance_metrics()`: Compute SNR, BER, capacity

**Example:**
```python
from mimo.mimo_channel import MIMOSystemSimulator

# Create 4x4 MIMO system
mimo = MIMOSystemSimulator(num_tx=4, num_rx=4, correlation='medium')

# Simulate transmission with MMSE processing
rx_signals, channel_matrix = mimo.simulate_transmission(
    tx_signals=mimo_input,
    precoding='mmse',
    snr_db=15
)

# Calculate metrics
metrics = mimo.calculate_performance_metrics(mimo_input, rx_signals)
print(f"SNR: {metrics['snr_measured']:.2f} dB")
```

## Signal Mixing

### SignalMixer

Combine multiple signals with different carriers and power levels.

```python
class SignalMixer:
    def __init__(self, sample_rate: float)
    def add_signal(
        self,
        signal: np.ndarray,
        carrier_freq: float,
        power_db: float,
        label: str = None
    )
    def mix_signals(self, duration: float = None) -> tuple
    def clear_signals(self)
```

**Methods:**
- `add_signal()`: Add signal to mixer with carrier frequency and power
- `mix_signals()`: Combine all added signals
- `clear_signals()`: Remove all signals from mixer

**Example:**
```python
from mixing.signal_mixer import SignalMixer

# Create multi-standard scenario
mixer = SignalMixer(sample_rate=30.72e6)

# Add different standards at different frequencies
mixer.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
mixer.add_signal(lte_signal, carrier_freq=1.8e9, power_db=-3, label='LTE')
mixer.add_signal(nr_signal, carrier_freq=3.5e9, power_db=-2, label='5G')

# Generate mixed signal
mixed_signal, metadata = mixer.mix_signals(duration=0.01)
```

### InterferenceGenerator

Generate various types of interference signals.

```python
class InterferenceGenerator:
    @staticmethod
    def generate_cw_tone(
        sample_rate: float,
        duration: float,
        frequency: float,
        power_db: float
    ) -> np.ndarray
    
    @staticmethod
    def generate_narrowband_noise(
        sample_rate: float,
        duration: float,
        center_freq: float,
        bandwidth: float,
        power_db: float
    ) -> np.ndarray
    
    @staticmethod
    def generate_wideband_interference(
        sample_rate: float,
        duration: float,
        power_db: float,
        spectrum_shape: str = 'flat'
    ) -> np.ndarray
```

## Validation Framework

### SignalAnalyzer

Analyze signal quality and characteristics.

```python
class SignalAnalyzer:
    def __init__(self, sample_rate: float)
    def calculate_power_metrics(self, signal_data: np.ndarray) -> dict
    def calculate_bandwidth(
        self,
        signal_data: np.ndarray,
        threshold_db: float = -3,
        method: str = 'psd'
    ) -> dict
    def calculate_evm(
        self,
        tx_symbols: np.ndarray,
        rx_symbols: np.ndarray
    ) -> dict
    def calculate_snr(
        self,
        signal_data: np.ndarray,
        noise_floor: float = None
    ) -> dict
```

**Methods:**
- `calculate_power_metrics()`: Power, PAPR, RMS calculations
- `calculate_bandwidth()`: Signal bandwidth using PSD or RMS methods
- `calculate_evm()`: Error Vector Magnitude
- `calculate_snr()`: Signal-to-noise ratio estimation

### StandardsValidator

Validate signals against cellular standards.

```python
class StandardsValidator:
    def validate_gsm_signal(
        self,
        signal_data: np.ndarray,
        sample_rate: float
    ) -> dict
    
    def validate_lte_signal(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        expected_bw_mhz: int = 20
    ) -> dict
    
    def validate_umts_signal(
        self,
        signal_data: np.ndarray,
        sample_rate: float
    ) -> dict
    
    def validate_nr_signal(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        expected_bw_mhz: int = 100
    ) -> dict
```

### ValidationReport

Generate comprehensive validation reports.

```python
class ValidationReport:
    def generate_signal_report(
        self,
        signal_data: np.ndarray,
        sample_rate: float,
        signal_type: str = 'Unknown'
    ) -> dict
    
    def print_report(self, report: dict)
    
    def save_report(self, report: dict, filename: str)
```

## Configuration Management

### ConfigLoader

Load and validate configuration files.

```python
class ConfigLoader:
    @staticmethod
    def load_signal_specs(config_file: str) -> dict
    
    @staticmethod
    def validate_config(config: dict) -> bool
    
    @staticmethod
    def get_default_config(signal_type: str) -> dict
```

## Utility Functions

### Data Processing
```python
def normalize_power(signal: np.ndarray, target_power_db: float = 0) -> np.ndarray
def add_frequency_offset(signal: np.ndarray, offset_hz: float, sample_rate: float) -> np.ndarray
def apply_time_delay(signal: np.ndarray, delay_samples: int) -> np.ndarray
def calculate_correlation(signal1: np.ndarray, signal2: np.ndarray) -> float
```

### File I/O
```python
def save_signal_dataset(
    signals: dict,
    metadata: dict,
    output_dir: str,
    compress: bool = True
) -> None

def load_signal_dataset(dataset_path: str) -> tuple
```

## Error Handling

### Custom Exceptions
```python
class SignalGenerationError(Exception):
    """Raised when signal generation fails"""
    pass

class ValidationError(Exception):
    """Raised when signal validation fails"""
    pass

class ParameterError(Exception):
    """Raised when invalid parameters are provided"""
    pass
```

## Examples and Tutorials

Complete examples are available in the `examples/` directory:

- `complete_demo.py`: Full demonstration of all features
- `demo_dataset_generation.py`: Basic dataset generation
- Individual generator examples in respective modules

For detailed tutorials and advanced usage, see the User Guide documentation.