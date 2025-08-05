"""
LTE (4G) signal generator
"""
import numpy as np
from src.signal_generation.base_generator import BaseSignalGenerator
from src.utils.config_loader import get_standard_specs


class LTEGenerator(BaseSignalGenerator):
    """LTE signal generator with OFDM"""
    
    def __init__(self, sample_rate=30.72e6, duration=1.0, bandwidth=20, 
                 modulation='64QAM', band='Band1'):
        """
        Initialize LTE signal generator
        
        Args:
            sample_rate: Sampling rate in Hz (default 30.72 MHz for 20MHz BW)
            duration: Signal duration in seconds
            bandwidth: Channel bandwidth in MHz
            modulation: Modulation scheme ('QPSK', '16QAM', '64QAM')
            band: LTE frequency band
        """
        super().__init__(sample_rate, duration)
        
        # Load LTE specifications
        self.specs = get_standard_specs('4g')
        self.bandwidth_mhz = bandwidth
        self.modulation = modulation
        self.band = band
        
        # LTE OFDM parameters
        self.subcarrier_spacing = self.specs['subcarrier_spacing'] * 1e3  # 15 kHz
        self.num_resource_blocks = self.specs['resource_blocks'][f'{bandwidth}MHz']
        self.num_subcarriers = self.num_resource_blocks * 12  # 12 subcarriers per RB
        self.fft_size = self._get_fft_size()
        self.cp_length = self._get_cp_length()
        
        # Symbol timing - will be calculated per symbol due to varying CP
        self.symbols_per_slot = 7  # Normal CP
        self.slots_per_subframe = 2
        self.subframes_per_frame = 10
        
    def _get_fft_size(self):
        """Get FFT size based on bandwidth (3GPP TS 36.211 Table 5.6-1)"""
        fft_sizes = {
            1.4: 128,   # 1.4 MHz -> 128-point FFT
            3: 256,     # 3 MHz -> 256-point FFT
            5: 512,     # 5 MHz -> 512-point FFT
            10: 1024,   # 10 MHz -> 1024-point FFT
            15: 1536,   # 15 MHz -> 1536-point FFT
            20: 2048    # 20 MHz -> 2048-point FFT
        }
        return fft_sizes.get(self.bandwidth_mhz, 2048)
    
    def _get_cp_length(self, symbol_idx=0):
        """
        Get cyclic prefix length according to 3GPP TS 36.211
        
        Args:
            symbol_idx: OFDM symbol index in slot (0-6 for normal CP)
        """
        if symbol_idx == 0:
            # First symbol in slot has extended CP
            # Normal CP: 160 samples at 30.72 MHz for 20 MHz BW
            cp_samples = int(160 * (self.sample_rate / 30.72e6))
        else:
            # Other symbols have normal CP
            # Normal CP: 144 samples at 30.72 MHz for 20 MHz BW
            cp_samples = int(144 * (self.sample_rate / 30.72e6))
        
        return cp_samples
    
    def generate_qam_symbols(self, num_symbols):
        """
        Generate QAM symbols based on modulation scheme
        Following 3GPP TS 36.211 Section 7.1 for constellation mapping
        """
        if self.modulation == 'QPSK':
            # 3GPP TS 36.211 Table 7.1.2-1: QPSK modulation
            constellation = np.array([
                1+1j, 1-1j, -1+1j, -1-1j
            ]) / np.sqrt(2)
            
        elif self.modulation == '16QAM':
            # 3GPP TS 36.211 Table 7.1.3-1: 16QAM modulation
            constellation = np.array([
                1+1j, 1+3j, 3+1j, 3+3j,
                1-1j, 1-3j, 3-1j, 3-3j,
                -1+1j, -1+3j, -3+1j, -3+3j,
                -1-1j, -1-3j, -3-1j, -3-3j
            ]) / np.sqrt(10)
            
        elif self.modulation == '64QAM':
            # 3GPP TS 36.211 Table 7.1.4-1: 64QAM modulation
            constellation = []
            for i in [-7, -5, -3, -1, 1, 3, 5, 7]:
                for q in [-7, -5, -3, -1, 1, 3, 5, 7]:
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation) / np.sqrt(42)
            
        elif self.modulation == '256QAM':
            # 3GPP TS 36.211 Table 7.1.5-1: 256QAM modulation
            constellation = []
            for i in [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]:
                for q in [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]:
                    constellation.append(i + 1j*q)
            constellation = np.array(constellation) / np.sqrt(170)
            
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        
        # Generate random symbols
        symbol_indices = np.random.randint(0, len(constellation), num_symbols)
        return constellation[symbol_indices]
    
    def generate_ofdm_symbol(self, data_symbols, symbol_idx=0):
        """
        Generate one OFDM symbol following 3GPP TS 36.211
        
        Args:
            data_symbols: Data symbols for active subcarriers
            symbol_idx: Symbol index in slot (0-6) for proper CP length
        """
        # Create frequency domain signal
        freq_signal = np.zeros(self.fft_size, dtype=complex)
        
        # Map data symbols to subcarriers following 3GPP resource grid
        # Active subcarriers are centered around DC with guard bands
        num_active = self.num_resource_blocks * 12  # 12 subcarriers per RB
        
        # Calculate subcarrier indices (symmetric around DC, skip DC itself)
        if num_active % 2 == 0:
            # Even number of subcarriers
            neg_indices = np.arange(-num_active//2, 0)
            pos_indices = np.arange(1, num_active//2 + 1)
        else:
            # Odd number of subcarriers (includes DC)
            neg_indices = np.arange(-(num_active//2), 0)
            pos_indices = np.arange(1, num_active//2 + 1)
        
        all_indices = np.concatenate([neg_indices, pos_indices])
        
        # Map to FFT indices
        fft_indices = all_indices % self.fft_size
        
        # Ensure we don't exceed data_symbols length
        num_mapped = min(len(fft_indices), len(data_symbols))
        freq_signal[fft_indices[:num_mapped]] = data_symbols[:num_mapped]
        
        # IFFT to time domain (3GPP uses IFFT with proper scaling)
        time_signal = np.fft.ifft(freq_signal) * np.sqrt(self.fft_size)
        
        # Add cyclic prefix with correct length for this symbol
        cp_length = self._get_cp_length(symbol_idx)
        cp_signal = np.concatenate([time_signal[-cp_length:], time_signal])
        
        return cp_signal
    
    def generate_baseband(self, **kwargs):
        """
        Generate LTE OFDM baseband signal following 3GPP frame structure
        """
        signal = []
        symbol_count = 0
        
        # Calculate symbols needed to fill duration
        # LTE frame structure: 10ms frame = 10 subframes = 20 slots = 140 symbols (normal CP)
        total_samples_needed = self.num_samples
        current_samples = 0
        
        while current_samples < total_samples_needed:
            # Generate one slot (7 OFDM symbols with normal CP)
            for symbol_idx in range(self.symbols_per_slot):
                # Generate data symbols for this OFDM symbol
                data_symbols = self.generate_qam_symbols(self.num_subcarriers)
                
                # Generate OFDM symbol with appropriate CP
                ofdm_symbol = self.generate_ofdm_symbol(data_symbols, symbol_idx)
                signal.extend(ofdm_symbol)
                
                current_samples += len(ofdm_symbol)
                symbol_count += 1
                
                # Break if we have enough samples
                if current_samples >= total_samples_needed:
                    break
            
            if current_samples >= total_samples_needed:
                break
        
        signal = np.array(signal)
        
        # Truncate to exact desired length
        if len(signal) > self.num_samples:
            signal = signal[:self.num_samples]
        elif len(signal) < self.num_samples:
            # Pad with zeros if needed
            signal = np.pad(signal, (0, self.num_samples - len(signal)), mode='constant')
        
        # Normalize power
        signal_power = np.mean(np.abs(signal)**2)
        if signal_power > 0:
            signal = signal / np.sqrt(signal_power)
            
        return signal
    
    def get_carrier_frequencies(self):
        """Get carrier frequencies for the selected band"""
        band_info = self.specs['frequency_bands'][self.band]
        ul_start, ul_end, dl_start, dl_end = band_info
        
        # Return center frequencies in MHz
        dl_center = (dl_start + dl_end) / 2
        ul_center = (ul_start + ul_end) / 2
        
        return {
            'downlink': dl_center * 1e6,  # Convert to Hz
            'uplink': ul_center * 1e6
        }


if __name__ == "__main__":
    # Test LTE generator
    print("Testing LTE generator...")
    
    gen = LTEGenerator(sample_rate=30.72e6, duration=0.01, bandwidth=20)  # 10ms signal
    
    # Generate baseband signal
    signal = gen.generate_baseband()
    print(f"Generated signal length: {len(signal)}")
    print(f"Signal power: {np.mean(np.abs(signal)**2):.4f}")
    print(f"FFT size: {gen.fft_size}")
    print(f"Number of subcarriers: {gen.num_subcarriers}")
    
    # Get carrier frequencies
    freqs = gen.get_carrier_frequencies()
    print(f"Carrier frequencies: {freqs}")
    
    # Test different modulations
    for mod in ['QPSK', '16QAM', '64QAM']:
        gen_mod = LTEGenerator(sample_rate=30.72e6, duration=0.001, 
                              bandwidth=20, modulation=mod)
        sig_mod = gen_mod.generate_baseband()
        print(f"{mod} signal power: {np.mean(np.abs(sig_mod)**2):.4f}")
    
    print("LTE generator test completed")