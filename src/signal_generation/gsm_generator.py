"""
GSM (2G) signal generator
"""
import numpy as np
from src.signal_generation.base_generator import BaseSignalGenerator
from src.utils.config_loader import get_standard_specs


class GSMGenerator(BaseSignalGenerator):
    """GSM signal generator"""
    
    def __init__(self, sample_rate=1e6, duration=1.0, band='GSM900'):
        """
        Initialize GSM signal generator
        
        Args:
            sample_rate: Sampling rate in Hz (default 1 MHz)
            duration: Signal duration in seconds
            band: GSM frequency band
        """
        super().__init__(sample_rate, duration)
        
        # Load GSM specifications
        self.specs = get_standard_specs('2g')
        self.band = band
        self.bandwidth = self.specs['channel_bandwidth'] * 1e6  # Convert to Hz
        self.symbol_rate = self.specs['symbol_rate'] * 1e3  # Convert to Hz
        
        # GSM-specific parameters
        self.bt = 0.3  # BT product for GMSK
        self.samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        
    def generate_data_bits(self, num_bits=None):
        """Generate random data bits"""
        if num_bits is None:
            # Generate enough bits for the duration
            bits_per_second = self.symbol_rate  # 1 bit per symbol for GMSK
            num_bits = int(bits_per_second * self.duration)
        
        return np.random.randint(0, 2, num_bits)
    
    def gaussian_filter(self, data_bits):
        """Apply Gaussian filter for GMSK modulation"""
        # Simple approximation of Gaussian filter
        # In practice, this would be more sophisticated
        filter_length = 4 * self.samples_per_symbol
        sigma = self.bt * self.samples_per_symbol / (2 * np.sqrt(2 * np.log(2)))
        
        # Create Gaussian filter
        t = np.arange(-filter_length//2, filter_length//2)
        h = np.exp(-t**2 / (2 * sigma**2))
        h = h / np.sum(h)
        
        # Upsample data bits
        upsampled = np.repeat(data_bits, self.samples_per_symbol)
        
        # Apply filter
        filtered = np.convolve(upsampled, h, mode='same')
        
        return filtered
    
    def generate_baseband(self, num_bits=None, **kwargs):
        """Generate GSM GMSK baseband signal"""
        # Generate data bits
        data_bits = self.generate_data_bits(num_bits)
        
        # Convert bits to NRZ format (-1, +1)
        nrz_data = 2 * data_bits - 1
        
        # Apply Gaussian filtering
        filtered_data = self.gaussian_filter(nrz_data)
        
        # Generate GMSK signal (frequency modulation)
        # Integrate filtered data to get phase
        dt = 1 / self.sample_rate
        phase = np.cumsum(filtered_data) * dt * np.pi * self.symbol_rate / 2
        
        # Generate complex baseband signal
        signal = np.exp(1j * phase)
        
        # Truncate to desired length
        if len(signal) > self.num_samples:
            signal = signal[:self.num_samples]
        elif len(signal) < self.num_samples:
            # Pad with zeros if needed
            signal = np.pad(signal, (0, self.num_samples - len(signal)))
        
        return signal
    
    def get_carrier_frequencies(self):
        """Get carrier frequencies for the selected band"""
        band_info = self.specs['frequency_bands'][self.band]
        dl_start, dl_end, ul_start, ul_end = band_info
        
        # Return center frequencies in MHz
        dl_center = (dl_start + dl_end) / 2
        ul_center = (ul_start + ul_end) / 2
        
        return {
            'downlink': dl_center * 1e6,  # Convert to Hz
            'uplink': ul_center * 1e6
        }


if __name__ == "__main__":
    # Test GSM generator
    print("Testing GSM generator...")
    
    gen = GSMGenerator(sample_rate=1e6, duration=0.1)  # 100ms signal
    
    # Generate baseband signal
    signal = gen.generate_baseband()
    print(f"Generated signal length: {len(signal)}")
    print(f"Signal power: {np.mean(np.abs(signal)**2):.4f}")
    
    # Get carrier frequencies
    freqs = gen.get_carrier_frequencies()
    print(f"Carrier frequencies: {freqs}")
    
    # Generate with carrier
    carrier_freq = freqs['downlink']
    rf_signal = gen.add_carrier(signal, carrier_freq)
    
    print(f"RF signal power: {np.mean(np.abs(rf_signal)**2):.4f}")
    print("GSM generator test completed")