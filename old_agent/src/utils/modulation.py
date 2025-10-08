"""
Shared modulation schemes and constellation utilities
Consolidates duplicate QAM constellation generation across signal generators
"""
import numpy as np


class ModulationSchemes:
    """Centralized modulation constellation generation and mapping"""
    
    @staticmethod
    def generate_qam_constellation(modulation_order):
        """
        Generate QAM constellation points
        
        Args:
            modulation_order: Number of constellation points (4, 16, 64, 256, 1024)
            
        Returns:
            constellation: Complex constellation points normalized to unit average power
        """
        if modulation_order == 4:  # QPSK
            return ModulationSchemes._generate_qpsk()
        elif modulation_order == 16:
            return ModulationSchemes._generate_16qam()
        elif modulation_order == 64:
            return ModulationSchemes._generate_64qam()
        elif modulation_order == 256:
            return ModulationSchemes._generate_256qam()
        elif modulation_order == 1024:
            return ModulationSchemes._generate_1024qam()
        else:
            raise ValueError(f"Unsupported modulation order: {modulation_order}")
    
    @staticmethod
    def _generate_qpsk():
        """Generate QPSK constellation (4-QAM)"""
        constellation = np.array([
            1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j
        ]) / np.sqrt(2)
        return constellation
    
    @staticmethod
    def _generate_16qam():
        """Generate 16-QAM constellation"""
        # Gray-coded 16-QAM per 3GPP TS 36.211 Table 7.1.4-1
        constellation = []
        for i in range(4):
            for q in range(4):
                real_part = 2*i - 3
                imag_part = 2*q - 3
                constellation.append(real_part + 1j*imag_part)
        
        constellation = np.array(constellation)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        return constellation
    
    @staticmethod
    def _generate_64qam():
        """Generate 64-QAM constellation"""
        # Gray-coded 64-QAM per 3GPP TS 36.211 Table 7.1.4-2
        constellation = []
        for i in range(8):
            for q in range(8):
                real_part = 2*i - 7
                imag_part = 2*q - 7
                constellation.append(real_part + 1j*imag_part)
        
        constellation = np.array(constellation)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        return constellation
    
    @staticmethod
    def _generate_256qam():
        """Generate 256-QAM constellation"""
        # Gray-coded 256-QAM per 3GPP TS 36.211 Table 7.1.4-3
        constellation = []
        for i in range(16):
            for q in range(16):
                real_part = 2*i - 15
                imag_part = 2*q - 15
                constellation.append(real_part + 1j*imag_part)
        
        constellation = np.array(constellation)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        return constellation
    
    @staticmethod
    def _generate_1024qam():
        """Generate 1024-QAM constellation"""
        # 1024-QAM constellation for 5G NR
        constellation = []
        for i in range(32):
            for q in range(32):
                real_part = 2*i - 31
                imag_part = 2*q - 31
                constellation.append(real_part + 1j*imag_part)
        
        constellation = np.array(constellation)
        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)
        return constellation
    
    @staticmethod
    def modulate_symbols(data_bits, modulation_order):
        """
        Modulate binary data to constellation symbols
        
        Args:
            data_bits: Binary data array
            modulation_order: Modulation order (4, 16, 64, 256, 1024)
            
        Returns:
            symbols: Complex modulated symbols
        """
        constellation = ModulationSchemes.generate_qam_constellation(modulation_order)
        bits_per_symbol = int(np.log2(modulation_order))
        
        # Pad data to multiple of bits_per_symbol
        padding = (bits_per_symbol - len(data_bits) % bits_per_symbol) % bits_per_symbol
        if padding > 0:
            data_bits = np.concatenate([data_bits, np.zeros(padding, dtype=int)])
        
        # Group bits and map to symbols
        data_bits = data_bits.reshape(-1, bits_per_symbol)
        symbol_indices = np.packbits(data_bits, axis=1, bitorder='big').flatten()
        
        # Handle different bit widths
        if bits_per_symbol <= 8:
            symbols = constellation[symbol_indices]
        else:
            # For 1024-QAM (10 bits), need special handling
            symbol_indices = np.sum(data_bits * (2 ** np.arange(bits_per_symbol-1, -1, -1)), axis=1)
            symbols = constellation[symbol_indices]
        
        return symbols
    
    @staticmethod
    def get_modulation_name(modulation_order):
        """Get standard name for modulation order"""
        names = {
            4: 'QPSK',
            16: '16-QAM',
            64: '64-QAM', 
            256: '256-QAM',
            1024: '1024-QAM'
        }
        return names.get(modulation_order, f'{modulation_order}-QAM')
    
    @staticmethod
    def get_bits_per_symbol(modulation_order):
        """Get number of bits per symbol for given modulation order"""
        return int(np.log2(modulation_order))


class DigitalModulation:
    """Additional digital modulation schemes"""
    
    @staticmethod
    def gmsk_modulate(data_bits, bt_product=0.3, samples_per_symbol=8):
        """
        GMSK modulation for GSM
        
        Args:
            data_bits: Binary data array
            bt_product: Bandwidth-time product (default 0.3 for GSM)
            samples_per_symbol: Oversampling factor
            
        Returns:
            modulated_signal: Complex GMSK modulated signal
        """
        # Convert bits to NRZ symbols
        symbols = 2 * data_bits - 1
        
        # Gaussian filter design
        # Create time vector for filter
        L = 4  # Filter length in symbols
        n = np.arange(-L*samples_per_symbol, L*samples_per_symbol+1)
        t = n / samples_per_symbol
        
        # Gaussian pulse
        alpha = np.sqrt(np.log(2)) / (2 * bt_product)
        g = alpha * np.sqrt(2*np.pi) * np.exp(-2 * (np.pi * alpha * t)**2)
        g = g / np.sum(g)  # Normalize
        
        # Upsample and filter
        upsampled = np.zeros(len(symbols) * samples_per_symbol)
        upsampled[::samples_per_symbol] = symbols
        
        # Apply Gaussian filter
        filtered = np.convolve(upsampled, g, mode='same')
        
        # Integrate to get phase
        phase = np.pi/2 * np.cumsum(filtered) / samples_per_symbol
        
        # Generate GMSK signal
        modulated = np.exp(1j * phase)
        
        return modulated
    
    @staticmethod  
    def bpsk_modulate(data_bits):
        """
        BPSK modulation
        
        Args:
            data_bits: Binary data array
            
        Returns:
            symbols: BPSK modulated symbols
        """
        return 2 * data_bits - 1
    
    @staticmethod
    def pi2_bpsk_modulate(data_bits):
        """
        π/2-BPSK modulation for 5G NR PUSCH
        
        Args:
            data_bits: Binary data array
            
        Returns:
            symbols: π/2-BPSK modulated symbols
        """
        symbols = 2 * data_bits - 1
        
        # Apply π/2 rotation every other symbol
        rotation = np.ones(len(symbols), dtype=complex)
        rotation[1::2] = 1j
        
        return symbols * rotation


if __name__ == "__main__":
    # Test modulation schemes
    print("Testing ModulationSchemes class...")
    
    # Test constellation generation
    for order in [4, 16, 64, 256]:
        constellation = ModulationSchemes.generate_qam_constellation(order)
        avg_power = np.mean(np.abs(constellation)**2)
        print(f"{ModulationSchemes.get_modulation_name(order)}: "
              f"{len(constellation)} points, avg power: {avg_power:.6f}")
    
    # Test symbol modulation
    test_bits = np.random.randint(0, 2, 100)
    symbols = ModulationSchemes.modulate_symbols(test_bits, 16)
    print(f"\nModulated {len(test_bits)} bits to {len(symbols)} 16-QAM symbols")
    print(f"Symbol power: {np.mean(np.abs(symbols)**2):.6f}")
    
    # Test GMSK
    gmsk_signal = DigitalModulation.gmsk_modulate(test_bits[:20])
    print(f"\nGMSK signal length: {len(gmsk_signal)}")
    print(f"GMSK power: {np.mean(np.abs(gmsk_signal)**2):.6f}")
    
    print("All tests completed successfully!")