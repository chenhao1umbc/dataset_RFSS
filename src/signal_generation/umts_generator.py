"""
UMTS (3G) signal generator with CDMA spreading
"""
import numpy as np
from src.signal_generation.base_generator import BaseSignalGenerator
from src.utils.config_loader import get_standard_specs
from src.utils.modulation import ModulationSchemes
from src.utils.signal_utils import normalize_power


class UMTSGenerator(BaseSignalGenerator):
    """UMTS signal generator with CDMA"""
    
    def __init__(self, sample_rate=15.36e6, duration=1.0, spreading_factor=256, 
                 band='Band1', num_users=1):
        """
        Initialize UMTS signal generator
        
        Args:
            sample_rate: Sampling rate in Hz (default 15.36 MHz for 4x chip rate)
            duration: Signal duration in seconds
            spreading_factor: Spreading factor (4-512)
            band: UMTS frequency band
            num_users: Number of simultaneous users
        """
        super().__init__(sample_rate, duration)
        
        # Load UMTS specifications
        self.specs = get_standard_specs('3g')
        self.spreading_factor = spreading_factor
        self.band = band
        self.num_users = num_users
        
        # UMTS parameters
        self.chip_rate = self.specs['chip_rate'] * 1e6  # 3.84 Mcps
        self.channel_bandwidth = 5e6  # 5 MHz
        self.samples_per_chip = int(self.sample_rate / self.chip_rate)
        
        # Frame structure
        self.frame_duration = self.specs['frame_duration'] / 1000  # 10 ms
        self.slots_per_frame = self.specs['slots_per_frame']  # 15
        self.slot_duration = self.frame_duration / self.slots_per_frame
        
    def generate_spreading_codes(self, sf):
        """Generate OVSF (Orthogonal Variable Spreading Factor) codes"""
        # Start with base code
        codes = np.array([[1]], dtype=int)
        
        # Generate OVSF tree
        for level in range(int(np.log2(sf))):
            new_codes = []
            for code in codes:
                # Each code generates two new codes
                new_codes.append(np.concatenate([code, code]))
                new_codes.append(np.concatenate([code, -code]))
            codes = np.array(new_codes)
        
        return codes
    
    def generate_scrambling_code(self, length):
        """Generate Gold scrambling code (simplified)"""
        # Simplified Gold code generation
        # In practice, this uses specific polynomials and initial states
        
        # Use LFSR-like sequence (simplified)
        np.random.seed(42)  # Fixed seed for reproducibility
        scrambling = np.random.choice([-1, 1], length)
        
        return scrambling
    
    def generate_data_symbols(self, num_symbols):
        """Generate QPSK data symbols using shared modulation utilities"""
        # Generate random bits
        bits = np.random.randint(0, 2, 2 * num_symbols)
        
        # Use shared QPSK constellation
        qpsk_constellation = ModulationSchemes.generate_qam_constellation(4)
        
        # Map bit pairs to QPSK symbols
        symbols = []
        for i in range(0, len(bits), 2):
            bit_pair = bits[i] * 2 + bits[i+1]  # Convert to 0,1,2,3
            symbols.append(qpsk_constellation[bit_pair])
        
        return np.array(symbols)
    
    def apply_spreading(self, data_symbols, spreading_codes, user_index=0):
        """Apply spreading to data symbols"""
        # Get spreading code for this user
        spreading_code = spreading_codes[user_index % len(spreading_codes)]
        
        # Spread each symbol
        spread_signal = []
        for symbol in data_symbols:
            # Multiply symbol by spreading code
            spread_chips = symbol * spreading_code
            spread_signal.extend(spread_chips)
        
        return np.array(spread_signal)
    
    def apply_scrambling(self, spread_signal):
        """Apply scrambling to spread signal"""
        # Generate scrambling code
        scrambling_code = self.generate_scrambling_code(len(spread_signal))
        
        # Apply scrambling (complex multiplication)
        scrambled_signal = spread_signal * scrambling_code
        
        return scrambled_signal
    
    def pulse_shape(self, chip_signal):
        """Apply pulse shaping (root raised cosine approximation)"""
        # Simple pulse shaping - repeat each chip
        if self.samples_per_chip > 1:
            shaped_signal = np.repeat(chip_signal, self.samples_per_chip)
        else:
            shaped_signal = chip_signal
        
        # Apply simple low-pass filtering
        if len(shaped_signal) > 10:
            # Simple moving average filter
            filter_length = min(10, len(shaped_signal) // 10)
            if filter_length > 1:
                kernel = np.ones(filter_length) / filter_length
                shaped_signal = np.convolve(shaped_signal, kernel, mode='same')
        
        return shaped_signal
    
    def generate_baseband(self, **kwargs):
        """Generate UMTS CDMA baseband signal"""
        # Calculate number of symbols needed
        symbol_rate = self.chip_rate / self.spreading_factor
        total_symbols = int(symbol_rate * self.duration)
        
        # Generate spreading codes
        spreading_codes = self.generate_spreading_codes(self.spreading_factor)
        
        # Initialize combined signal
        combined_signal = np.zeros(0, dtype=complex)
        
        # Generate signal for each user
        for user in range(self.num_users):
            # Generate data symbols for this user
            data_symbols = self.generate_data_symbols(total_symbols)
            
            # Apply spreading
            spread_signal = self.apply_spreading(data_symbols, spreading_codes, user)
            
            # Apply scrambling
            scrambled_signal = self.apply_scrambling(spread_signal)
            
            # Apply pulse shaping
            shaped_signal = self.pulse_shape(scrambled_signal)
            
            # Add to combined signal
            if len(combined_signal) == 0:
                combined_signal = shaped_signal
            else:
                # Ensure same length
                min_len = min(len(combined_signal), len(shaped_signal))
                combined_signal = combined_signal[:min_len] + shaped_signal[:min_len]
        
        # Normalize by number of users and apply power normalization
        if self.num_users > 1:
            combined_signal = combined_signal / np.sqrt(self.num_users)
        
        # Truncate or pad to desired length
        if len(combined_signal) > self.num_samples:
            combined_signal = combined_signal[:self.num_samples]
        elif len(combined_signal) < self.num_samples:
            combined_signal = np.pad(combined_signal, (0, self.num_samples - len(combined_signal)))
        
        # Apply power normalization using shared utilities
        combined_signal = normalize_power(combined_signal, target_power=1.0)
        
        return combined_signal
    
    def get_carrier_frequencies(self):
        """Get carrier frequencies for the selected band"""
        band_info = self.specs['frequency_bands'][self.band]
        ul_start, ul_end, dl_start, dl_end = band_info
        
        # Return center frequencies
        dl_center = (dl_start + dl_end) / 2
        ul_center = (ul_start + ul_end) / 2
        
        return {
            'downlink': dl_center * 1e6,  # Convert to Hz
            'uplink': ul_center * 1e6
        }


if __name__ == "__main__":
    # Test UMTS generator
    print("Testing UMTS generator...")
    
    # Test with different spreading factors
    for sf in [64, 128, 256]:
        print(f"\nTesting SF={sf}...")
        gen = UMTSGenerator(sample_rate=15.36e6, duration=0.01, 
                           spreading_factor=sf, num_users=2)
        
        # Generate baseband signal
        signal = gen.generate_baseband()
        print(f"  Signal length: {len(signal)}")
        print(f"  Signal power: {np.mean(np.abs(signal)**2):.6f}")
        print(f"  Samples per chip: {gen.samples_per_chip}")
        
        # Get carrier frequencies
        freqs = gen.get_carrier_frequencies()
        print(f"  Carrier frequencies: {freqs}")
    
    # Test spreading codes
    gen = UMTSGenerator(sample_rate=15.36e6, duration=0.001, spreading_factor=16)
    codes = gen.generate_spreading_codes(16)
    print(f"\nSpreading codes for SF=16:")
    print(f"  Number of codes: {len(codes)}")
    print(f"  Code length: {len(codes[0])}")
    print(f"  First few codes: {codes[:4]}")
    
    # Test orthogonality
    if len(codes) >= 2:
        correlation = np.dot(codes[0], codes[1])
        print(f"  Correlation between codes 0 and 1: {correlation}")
    
    print("UMTS generator test completed")