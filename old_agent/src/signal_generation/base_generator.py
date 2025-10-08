"""
Base signal generator class for all cellular standards
"""
import numpy as np
from abc import ABC, abstractmethod
from ..utils.signal_utils import normalize_power, add_awgn_noise, add_carrier_frequency, generate_time_vector


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
            self.time_vector = generate_time_vector(self.num_samples, self.sample_rate)
    
    @abstractmethod
    def generate_baseband(self, **kwargs):
        """Generate baseband signal - must be implemented by subclasses"""
        pass
    
    def add_carrier(self, baseband_signal, carrier_freq):
        """Add carrier frequency to baseband signal"""
        if self.sample_rate is None:
            raise ValueError("Sample rate not initialized")
        
        return add_carrier_frequency(baseband_signal, carrier_freq, self.sample_rate)
    
    def add_noise(self, signal, snr_db):
        """Add AWGN noise to signal"""
        return add_awgn_noise(signal, snr_db)
    
    def normalize_power(self, signal, target_power=1.0):
        """Normalize signal to target power"""
        return normalize_power(signal, target_power)
    
    def get_signal_info(self):
        """Get basic signal information"""
        return {
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'num_samples': self.num_samples,
            'bandwidth': getattr(self, 'bandwidth', None),
            'carrier_freq': getattr(self, 'carrier_freq', None)
        }