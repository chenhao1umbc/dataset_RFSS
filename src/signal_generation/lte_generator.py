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
        
        # Symbol timing
        self.ofdm_symbol_time = (self.fft_size + self.cp_length) / self.sample_rate
        self.symbols_per_slot = 7  # Normal CP
        
    def _get_fft_size(self):
        """Get FFT size based on bandwidth"""
        fft_sizes = {
            1.4: 128, 3: 256, 5: 512, 10: 1024, 15: 1536, 20: 2048
        }
        return fft_sizes.get(self.bandwidth_mhz, 2048)
    
    def _get_cp_length(self):
        """Get cyclic prefix length"""
        # Simplified - first symbol has longer CP
        return self.fft_size // 8  # Approximately 7% of FFT size
    
    def generate_qam_symbols(self, num_symbols):
        """Generate QAM symbols based on modulation scheme"""
        if self.modulation == 'QPSK':
            bits_per_symbol = 2
            constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            bits_per_symbol = 4
            # Simplified 16-QAM constellation
            real_part = np.array([-3, -1, 1, 3])
            imag_part = np.array([-3, -1, 1, 3])
            constellation = []
            for r in real_part:
                for i in imag_part:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation) / np.sqrt(10)
        elif self.modulation == '64QAM':
            bits_per_symbol = 6
            # Simplified 64-QAM constellation
            real_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            imag_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = []
            for r in real_part:
                for i in imag_part:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation) / np.sqrt(42)
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        
        # Generate random symbols
        symbol_indices = np.random.randint(0, len(constellation), num_symbols)
        return constellation[symbol_indices]
    
    def generate_ofdm_symbol(self, data_symbols):
        """Generate one OFDM symbol"""
        # Create frequency domain signal
        freq_signal = np.zeros(self.fft_size, dtype=complex)
        
        # Map data symbols to subcarriers (skip DC and guard bands)
        used_subcarriers = len(data_symbols)
        start_idx = (self.fft_size - used_subcarriers) // 2
        end_idx = start_idx + used_subcarriers
        freq_signal[start_idx:end_idx] = data_symbols
        
        # IFFT to time domain
        time_signal = np.fft.ifft(freq_signal) * np.sqrt(self.fft_size)
        
        # Add cyclic prefix
        cp_signal = np.concatenate([time_signal[-self.cp_length:], time_signal])
        
        return cp_signal
    
    def generate_baseband(self, **kwargs):
        """Generate LTE OFDM baseband signal"""
        # Calculate number of OFDM symbols needed
        symbol_duration = self.ofdm_symbol_time
        num_ofdm_symbols = int(self.duration / symbol_duration)
        
        # Generate signal
        signal = []
        
        for _ in range(num_ofdm_symbols):
            # Generate data symbols for this OFDM symbol
            data_symbols = self.generate_qam_symbols(self.num_subcarriers)
            
            # Generate OFDM symbol
            ofdm_symbol = self.generate_ofdm_symbol(data_symbols)
            signal.extend(ofdm_symbol)
        
        signal = np.array(signal)
        
        # Truncate or pad to desired length
        if len(signal) > self.num_samples:
            signal = signal[:self.num_samples]
        elif len(signal) < self.num_samples:
            signal = np.pad(signal, (0, self.num_samples - len(signal)))
        
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