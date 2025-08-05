"""
Base signal generator class for all cellular standards
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseSignalGenerator(ABC):
    """Base class for all signal generators"""
    
    def __init__(self, sample_rate=None, duration=1.0):
        """
        Initialize base signal generator
        
        Args:
            sample_rate: Sampling rate in Hz
            duration: Signal duration in seconds
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration) if sample_rate else None
        self.time_vector = None
        
        if self.sample_rate:
            self.time_vector = np.arange(self.num_samples) / self.sample_rate
    
    @abstractmethod
    def generate_baseband(self, **kwargs):
        """Generate baseband signal - must be implemented by subclasses"""
        pass
    
    def add_carrier(self, baseband_signal, carrier_freq):
        """Add carrier frequency to baseband signal"""
        if self.time_vector is None:
            raise ValueError("Time vector not initialized - need sample_rate")
        
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * self.time_vector)
        return baseband_signal * carrier
    
    def add_noise(self, signal, snr_db):
        """Add AWGN noise to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(snr_db/10))
        
        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                            1j * np.random.randn(len(signal)))
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        return signal + noise
    
    def normalize_power(self, signal, target_power=1.0):
        """Normalize signal to target power"""
        current_power = np.mean(np.abs(signal)**2)
        scale_factor = np.sqrt(target_power / current_power)
        return signal * scale_factor
    
    def get_signal_info(self):
        """Get basic signal information"""
        return {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'num_samples': self.num_samples,
            'bandwidth': getattr(self, 'bandwidth', None),
            'carrier_freq': getattr(self, 'carrier_freq', None)
        }