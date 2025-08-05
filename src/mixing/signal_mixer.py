"""
Signal mixing engine for combining multiple RF signals
"""
import numpy as np
from typing import List, Dict, Tuple


class SignalMixer:
    """Mixes multiple RF signals with different parameters"""
    
    def __init__(self, sample_rate):
        """
        Initialize signal mixer
        
        Args:
            sample_rate: Common sampling rate for all signals
        """
        self.sample_rate = sample_rate
        self.signals = []
        self.metadata = []
    
    def add_signal(self, signal, carrier_freq, power_db=0, label=None, **metadata):
        """
        Add a signal to the mixer
        
        Args:
            signal: Complex baseband signal
            carrier_freq: Carrier frequency in Hz
            power_db: Signal power in dB (relative to reference)
            label: Signal label/type (e.g., 'LTE', 'GSM')
            **metadata: Additional metadata about the signal
        """
        signal_info = {
            'signal': signal,
            'carrier_freq': carrier_freq,
            'power_db': power_db,
            'label': label,
            'metadata': metadata
        }
        
        self.signals.append(signal_info)
        self.metadata.append(signal_info)
    
    def set_signal_power(self, index, power_db):
        """Set power level for a specific signal"""
        if 0 <= index < len(self.signals):
            self.signals[index]['power_db'] = power_db
    
    def frequency_shift(self, signal, carrier_freq, duration=None):
        """Shift signal to carrier frequency"""
        if duration is None:
            duration = len(signal) / self.sample_rate
        
        t = np.arange(len(signal)) / self.sample_rate
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        return signal * carrier
    
    def normalize_power(self, signal, power_db):
        """Normalize signal to specified power level"""
        current_power = np.mean(np.abs(signal)**2)
        target_power = 10**(power_db/10)
        scale_factor = np.sqrt(target_power / current_power)
        
        return signal * scale_factor
    
    def mix_signals(self, duration=None, normalize_output=True):
        """
        Mix all added signals
        
        Args:
            duration: Output duration in seconds (uses longest signal if None)
            normalize_output: Whether to normalize the mixed output
            
        Returns:
            mixed_signal: Combined RF signal
            mixing_info: Information about the mixing process
        """
        if not self.signals:
            raise ValueError("No signals added to mixer")
        
        # Determine output length
        if duration is None:
            max_length = max(len(sig['signal']) for sig in self.signals)
        else:
            max_length = int(duration * self.sample_rate)
        
        # Initialize output
        mixed_signal = np.zeros(max_length, dtype=complex)
        
        # Mix each signal
        for i, sig_info in enumerate(self.signals):
            signal = sig_info['signal']
            carrier_freq = sig_info['carrier_freq']
            power_db = sig_info['power_db']
            
            # Extend or truncate signal to match output length
            if len(signal) < max_length:
                # Repeat signal if too short
                repeats = int(np.ceil(max_length / len(signal)))
                extended_signal = np.tile(signal, repeats)[:max_length]
            else:
                extended_signal = signal[:max_length]
            
            # Apply carrier frequency
            rf_signal = self.frequency_shift(extended_signal, carrier_freq)
            
            # Apply power scaling
            powered_signal = self.normalize_power(rf_signal, power_db)
            
            # Add to mixed signal
            mixed_signal += powered_signal
        
        # Normalize output if requested
        if normalize_output:
            mixed_signal = self.normalize_power(mixed_signal, 0)  # Normalize to 0 dB
        
        # Create mixing information
        mixing_info = {
            'sample_rate': self.sample_rate,
            'duration': max_length / self.sample_rate,
            'num_signals': len(self.signals),
            'signals': [
                {
                    'label': sig['label'],
                    'carrier_freq': sig['carrier_freq'],
                    'power_db': sig['power_db'],
                    'metadata': sig['metadata']
                }
                for sig in self.signals
            ]
        }
        
        return mixed_signal, mixing_info
    
    def clear_signals(self):
        """Clear all signals from mixer"""
        self.signals = []
        self.metadata = []
    
    def get_signal_info(self):
        """Get information about all added signals"""
        return [
            {
                'index': i,
                'label': sig['label'],
                'carrier_freq': sig['carrier_freq'],
                'power_db': sig['power_db'],
                'signal_length': len(sig['signal']),
                'metadata': sig['metadata']
            }
            for i, sig in enumerate(self.signals)
        ]


class InterferenceGenerator:
    """Generate common interference signals"""
    
    @staticmethod
    def generate_cw_tone(sample_rate, duration, freq, power_db=0):
        """Generate continuous wave (CW) tone"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = np.exp(1j * 2 * np.pi * freq * t)
        
        # Apply power scaling
        target_power = 10**(power_db/10)
        current_power = np.mean(np.abs(signal)**2)
        scale_factor = np.sqrt(target_power / current_power)
        
        return signal * scale_factor
    
    @staticmethod
    def generate_chirp(sample_rate, duration, f_start, f_end, power_db=0):
        """Generate linear chirp signal"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        
        # Linear chirp from f_start to f_end
        chirp_rate = (f_end - f_start) / duration
        phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2)
        signal = np.exp(1j * phase)
        
        # Apply power scaling
        target_power = 10**(power_db/10)
        current_power = np.mean(np.abs(signal)**2)
        scale_factor = np.sqrt(target_power / current_power)
        
        return signal * scale_factor
    
    @staticmethod
    def generate_narrowband_noise(sample_rate, duration, center_freq, bandwidth, power_db=0):
        """Generate narrowband noise"""
        num_samples = int(sample_rate * duration)
        
        # Generate white noise
        noise = np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
        
        # Filter to desired bandwidth (simple rectangular filter in frequency domain)
        freq_noise = np.fft.fft(noise)
        freqs = np.fft.fftfreq(num_samples, 1/sample_rate)
        
        # Create filter mask
        mask = np.abs(freqs - center_freq) <= bandwidth/2
        freq_noise[~mask] = 0
        
        # Transform back to time domain
        filtered_noise = np.fft.ifft(freq_noise)
        
        # Apply power scaling
        target_power = 10**(power_db/10)
        current_power = np.mean(np.abs(filtered_noise)**2)
        if current_power > 0:
            scale_factor = np.sqrt(target_power / current_power)
            filtered_noise *= scale_factor
        
        return filtered_noise


if __name__ == "__main__":
    # Test signal mixer
    print("Testing signal mixer...")
    
    sample_rate = 10e6  # 10 MHz
    duration = 0.001    # 1 ms
    
    # Create mixer
    mixer = SignalMixer(sample_rate)
    
    # Generate test signals
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Signal 1: LTE-like OFDM signal
    signal1 = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    mixer.add_signal(signal1, carrier_freq=2e9, power_db=0, label='LTE')
    
    # Signal 2: GSM-like signal
    signal2 = np.exp(1j * 2 * np.pi * 1e3 * t)  # 1 kHz tone
    mixer.add_signal(signal2, carrier_freq=900e6, power_db=-10, label='GSM')
    
    # Signal 3: Interference
    interference = InterferenceGenerator.generate_cw_tone(
        sample_rate, duration, 0, power_db=-20
    )
    mixer.add_signal(interference, carrier_freq=1.8e9, power_db=-20, label='Interference')
    
    # Mix signals
    mixed_signal, mixing_info = mixer.mix_signals()
    
    print(f"Mixed signal length: {len(mixed_signal)}")
    print(f"Mixed signal power: {np.mean(np.abs(mixed_signal)**2):.4f}")
    print(f"Number of mixed signals: {mixing_info['num_signals']}")
    
    # Test interference generators
    cw_tone = InterferenceGenerator.generate_cw_tone(sample_rate, duration, 1e6)
    chirp = InterferenceGenerator.generate_chirp(sample_rate, duration, 1e6, 2e6)
    nb_noise = InterferenceGenerator.generate_narrowband_noise(
        sample_rate, duration, 0, 100e3
    )
    
    print(f"CW tone power: {np.mean(np.abs(cw_tone)**2):.4f}")
    print(f"Chirp power: {np.mean(np.abs(chirp)**2):.4f}")
    print(f"Narrowband noise power: {np.mean(np.abs(nb_noise)**2):.4f}")
    
    print("Signal mixer test completed")