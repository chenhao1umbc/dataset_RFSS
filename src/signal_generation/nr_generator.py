"""
5G NR (New Radio) signal generator with flexible numerology
"""
import numpy as np
from src.signal_generation.base_generator import BaseSignalGenerator
from src.utils.config_loader import get_standard_specs


class NRGenerator(BaseSignalGenerator):
    """5G NR signal generator with OFDM and flexible numerology"""
    
    def __init__(self, sample_rate=61.44e6, duration=1.0, bandwidth=100, 
                 numerology=1, modulation='256QAM', band='n78'):
        """
        Initialize 5G NR signal generator
        
        Args:
            sample_rate: Sampling rate in Hz (default 61.44 MHz for 100MHz BW)
            duration: Signal duration in seconds  
            bandwidth: Channel bandwidth in MHz
            numerology: Numerology μ (0-4, determines subcarrier spacing)
            modulation: Modulation scheme
            band: 5G NR frequency band
        """
        super().__init__(sample_rate, duration)
        
        # Load 5G NR specifications
        self.specs = get_standard_specs('5g')
        self.bandwidth_mhz = bandwidth
        self.numerology = numerology
        self.modulation = modulation
        self.band = band
        
        # NR OFDM parameters based on numerology
        self.subcarrier_spacing = self.specs['subcarrier_spacing'][numerology] * 1e3  # Hz
        self.slots_per_subframe = self.specs['slots_per_subframe'][numerology]
        self.symbols_per_slot = self.specs['symbols_per_slot']  # 14 for normal CP
        
        # Calculate derived parameters
        self.fft_size = self._calculate_fft_size()
        self.cp_length = self._calculate_cp_length()
        self.num_resource_blocks = self._calculate_num_rbs()
        self.num_subcarriers = self.num_resource_blocks * 12  # 12 subcarriers per RB
        
        # Timing parameters
        self.symbol_duration = (self.fft_size + self.cp_length) / self.sample_rate
        self.slot_duration = self.symbols_per_slot * self.symbol_duration
        
    def _calculate_fft_size(self):
        """Calculate FFT size based on bandwidth and numerology"""
        # Standard FFT sizes for different bandwidths and numerologies
        fft_size_table = {
            # numerology: {bandwidth: fft_size}
            0: {5: 512, 10: 1024, 15: 1536, 20: 2048, 25: 2048, 30: 2048, 
                40: 4096, 50: 4096, 60: 4096, 80: 4096, 90: 4096, 100: 4096},
            1: {5: 512, 10: 1024, 15: 1536, 20: 2048, 25: 2048, 30: 2048,
                40: 2048, 50: 4096, 60: 4096, 80: 4096, 90: 4096, 100: 4096},
            2: {10: 512, 15: 1024, 20: 1024, 25: 1024, 30: 1024, 40: 2048,
                50: 2048, 60: 2048, 80: 4096, 90: 4096, 100: 4096},
            3: {50: 512, 100: 1024, 200: 2048, 400: 4096},
            4: {50: 256, 100: 512, 200: 1024, 400: 2048}
        }
        
        return fft_size_table.get(self.numerology, {}).get(self.bandwidth_mhz, 4096)
    
    def _calculate_cp_length(self):
        """Calculate cyclic prefix length"""
        # Simplified CP length calculation
        # First symbol in slot has longer CP
        return self.fft_size // 16  # Approximately 6.25% of FFT size
    
    def _calculate_num_rbs(self):
        """Calculate number of resource blocks"""
        # Simplified calculation based on bandwidth
        # Each RB is 12 subcarriers * subcarrier_spacing
        rb_bandwidth = 12 * self.subcarrier_spacing
        total_bandwidth = self.bandwidth_mhz * 1e6
        
        # Account for guard bands (simplified)
        usable_bandwidth = total_bandwidth * 0.9  # 90% utilization
        num_rbs = int(usable_bandwidth / rb_bandwidth)
        
        return min(num_rbs, 275)  # Max 275 RBs for NR
    
    def generate_qam_symbols(self, num_symbols):
        """Generate QAM symbols based on modulation scheme"""
        if self.modulation == 'QPSK':
            bits_per_symbol = 2
            constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        elif self.modulation == '16QAM':
            bits_per_symbol = 4
            real_part = np.array([-3, -1, 1, 3])
            imag_part = np.array([-3, -1, 1, 3])
            constellation = []
            for r in real_part:
                for i in imag_part:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation) / np.sqrt(10)
        elif self.modulation == '64QAM':
            bits_per_symbol = 6
            real_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            imag_part = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = []
            for r in real_part:
                for i in imag_part:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation) / np.sqrt(42)
        elif self.modulation == '256QAM':
            bits_per_symbol = 8
            real_part = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 
                                 1, 3, 5, 7, 9, 11, 13, 15])
            imag_part = np.array([-15, -13, -11, -9, -7, -5, -3, -1,
                                 1, 3, 5, 7, 9, 11, 13, 15])
            constellation = []
            for r in real_part:
                for i in imag_part:
                    constellation.append(r + 1j*i)
            constellation = np.array(constellation) / np.sqrt(170)
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
        
        # Generate random symbols
        symbol_indices = np.random.randint(0, len(constellation), num_symbols)
        return constellation[symbol_indices]
    
    def generate_reference_signals(self, symbol_index, slot_index):
        """Generate reference signals (simplified DMRS)"""
        # Simplified demodulation reference signals
        # In practice, this would follow 3GPP specifications
        
        ref_signal = np.exp(1j * 2 * np.pi * np.random.rand(self.num_subcarriers))
        
        # Apply reference signal pattern (every 4th subcarrier)
        ref_pattern = np.zeros(self.num_subcarriers, dtype=complex)
        ref_pattern[::4] = ref_signal[::4]
        
        return ref_pattern
    
    def generate_ofdm_symbol(self, data_symbols, symbol_index=0, slot_index=0):
        """Generate one OFDM symbol"""
        # Create frequency domain signal
        freq_signal = np.zeros(self.fft_size, dtype=complex)
        
        # Check if this is a reference symbol
        is_ref_symbol = (symbol_index % 4 == 1)  # Simplified pattern
        
        if is_ref_symbol:
            # Generate reference signals
            ref_signals = self.generate_reference_signals(symbol_index, slot_index)
            # Mix data and reference signals
            used_subcarriers = min(len(data_symbols), len(ref_signals))
            mixed_signals = np.zeros(used_subcarriers, dtype=complex)
            mixed_signals[::4] = ref_signals[::4]  # Reference on every 4th subcarrier
            mixed_signals[1::4] = data_symbols[1::4] if len(data_symbols) > 1 else 0
            mixed_signals[2::4] = data_symbols[2::4] if len(data_symbols) > 2 else 0  
            mixed_signals[3::4] = data_symbols[3::4] if len(data_symbols) > 3 else 0
            data_to_use = mixed_signals
        else:
            data_to_use = data_symbols
        
        # Map data symbols to subcarriers (skip DC and guard bands)
        used_subcarriers = min(len(data_to_use), self.num_subcarriers)
        start_idx = (self.fft_size - used_subcarriers) // 2
        end_idx = start_idx + used_subcarriers
        freq_signal[start_idx:end_idx] = data_to_use[:used_subcarriers]
        
        # IFFT to time domain
        time_signal = np.fft.ifft(freq_signal) * np.sqrt(self.fft_size)
        
        # Add cyclic prefix
        if symbol_index == 0:  # First symbol in slot has longer CP
            cp_len = int(self.cp_length * 1.25)  # 25% longer
        else:
            cp_len = self.cp_length
        
        cp_signal = np.concatenate([time_signal[-cp_len:], time_signal])
        
        return cp_signal
    
    def generate_baseband(self, **kwargs):
        """Generate 5G NR OFDM baseband signal"""
        # Calculate number of slots needed
        num_slots = int(self.duration / self.slot_duration)
        if num_slots == 0:
            num_slots = 1
        
        # Generate signal
        signal = []
        
        for slot_idx in range(num_slots):
            for symbol_idx in range(self.symbols_per_slot):
                # Generate data symbols for this OFDM symbol
                data_symbols = self.generate_qam_symbols(self.num_subcarriers)
                
                # Generate OFDM symbol
                ofdm_symbol = self.generate_ofdm_symbol(data_symbols, symbol_idx, slot_idx)
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
        band_info = self.specs['frequency_bands'].get(self.band)
        if not band_info:
            # Default frequencies for n78 band
            band_info = [3300, 3800, 3300, 3800]
        
        # Handle TDD bands (same UL/DL frequencies)
        if len(band_info) == 4:
            ul_start, ul_end, dl_start, dl_end = band_info
        else:
            # TDD band
            ul_start = ul_end = dl_start = dl_end = band_info[0]
        
        # Return center frequencies
        dl_center = (dl_start + dl_end) / 2
        ul_center = (ul_start + ul_end) / 2
        
        return {
            'downlink': dl_center * 1e6,  # Convert to Hz
            'uplink': ul_center * 1e6
        }


if __name__ == "__main__":
    # Test 5G NR generator
    print("Testing 5G NR generator...")
    
    # Test different numerologies
    for mu in [0, 1, 2]:
        if mu <= 2:  # Only test sub-6 numerologies
            print(f"\nTesting numerology μ={mu}...")
            gen = NRGenerator(sample_rate=61.44e6, duration=0.001,  # 1ms
                             bandwidth=100, numerology=mu, modulation='64QAM')
            
            # Generate baseband signal
            signal = gen.generate_baseband()
            print(f"  Signal length: {len(signal)}")
            print(f"  Signal power: {np.mean(np.abs(signal)**2):.6f}")
            print(f"  Subcarrier spacing: {gen.subcarrier_spacing/1e3:.0f} kHz")
            print(f"  FFT size: {gen.fft_size}")
            print(f"  Number of RBs: {gen.num_resource_blocks}")
            print(f"  Symbol duration: {gen.symbol_duration*1e6:.2f} μs")
    
    # Test different modulations
    print(f"\nTesting different modulations...")
    for mod in ['QPSK', '16QAM', '64QAM', '256QAM']:
        gen_mod = NRGenerator(sample_rate=30.72e6, duration=0.001,
                             bandwidth=50, numerology=1, modulation=mod)
        sig_mod = gen_mod.generate_baseband()
        power = np.mean(np.abs(sig_mod)**2)
        papr = 10 * np.log10(np.max(np.abs(sig_mod)**2) / power)
        print(f"  {mod}: power={power:.6f}, PAPR={papr:.2f} dB")
    
    # Test carrier frequencies
    gen = NRGenerator(sample_rate=61.44e6, duration=0.001, bandwidth=100)
    freqs = gen.get_carrier_frequencies()
    print(f"\nCarrier frequencies: {freqs}")
    
    print("5G NR generator test completed")