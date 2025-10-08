"""
Basic channel models for RF signal simulation
"""
import numpy as np
from src.utils.config_loader import get_channel_models


class AWGNChannel:
    """Additive White Gaussian Noise channel"""
    
    def __init__(self, snr_db=10):
        """
        Initialize AWGN channel
        
        Args:
            snr_db: Signal-to-noise ratio in dB
        """
        self.snr_db = snr_db
    
    def apply(self, signal):
        """Apply AWGN to signal"""
        signal_power = np.mean(np.abs(signal)**2)
        noise_power = signal_power / (10**(self.snr_db/10))
        
        if np.iscomplexobj(signal):
            noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                            1j * np.random.randn(len(signal)))
        else:
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        return signal + noise


class RayleighChannel:
    """Rayleigh fading channel"""
    
    def __init__(self, sample_rate, doppler_freq=50):
        """
        Initialize Rayleigh fading channel
        
        Args:
            sample_rate: Sample rate in Hz
            doppler_freq: Maximum Doppler frequency in Hz
        """
        self.sample_rate = sample_rate
        self.doppler_freq = doppler_freq
    
    def generate_fading(self, num_samples):
        """Generate Rayleigh fading coefficients"""
        # Generate complex Gaussian random process
        # Simple approximation using Jakes' model
        
        # Number of sinusoids for Jakes' model
        N = 8
        
        # Generate fading coefficients
        h = np.zeros(num_samples, dtype=complex)
        
        for n in range(N):
            phase_n = 2 * np.pi * n / N
            alpha_n = np.pi * (2*n - 1) / (4*N)
            
            t = np.arange(num_samples) / self.sample_rate
            
            h += np.exp(1j * (2 * np.pi * self.doppler_freq * np.cos(alpha_n) * t + phase_n))
        
        # Normalize
        h = h / np.sqrt(N)
        
        return h
    
    def apply(self, signal):
        """Apply Rayleigh fading to signal"""
        fading_coeff = self.generate_fading(len(signal))
        return signal * fading_coeff


class RicianChannel:
    """Rician fading channel"""
    
    def __init__(self, sample_rate, doppler_freq=50, k_factor_db=6):
        """
        Initialize Rician fading channel
        
        Args:
            sample_rate: Sample rate in Hz
            doppler_freq: Maximum Doppler frequency in Hz
            k_factor_db: Rician K-factor in dB
        """
        self.sample_rate = sample_rate
        self.doppler_freq = doppler_freq
        self.k_factor_db = k_factor_db
        self.k_factor = 10**(k_factor_db/10)
    
    def generate_fading(self, num_samples):
        """Generate Rician fading coefficients"""
        # Line-of-sight component
        los_component = np.sqrt(self.k_factor / (self.k_factor + 1))
        
        # Scattered component (Rayleigh)
        rayleigh_channel = RayleighChannel(self.sample_rate, self.doppler_freq)
        scattered_component = rayleigh_channel.generate_fading(num_samples)
        scattered_component *= np.sqrt(1 / (self.k_factor + 1))
        
        # Combine LOS and scattered components
        h = los_component + scattered_component
        
        return h
    
    def apply(self, signal):
        """Apply Rician fading to signal"""
        fading_coeff = self.generate_fading(len(signal))
        return signal * fading_coeff


class MultipathChannel:
    """Multipath channel with multiple taps"""
    
    def __init__(self, sample_rate, tap_delays_ns=None, tap_powers_db=None):
        """
        Initialize multipath channel
        
        Args:
            sample_rate: Sample rate in Hz
            tap_delays_ns: Tap delays in nanoseconds
            tap_powers_db: Tap powers in dB
        """
        self.sample_rate = sample_rate
        
        # Use default ITU Pedestrian A model if not specified
        if tap_delays_ns is None:
            tap_delays_ns = [0, 50, 120, 200, 230, 500]
        if tap_powers_db is None:
            tap_powers_db = [0, -9.7, -19.2, -22.8, -26.2, -26.7]
        
        self.tap_delays_ns = np.array(tap_delays_ns)
        self.tap_powers_db = np.array(tap_powers_db)
        
        # Convert to sample delays
        self.tap_delays_samples = np.round(self.tap_delays_ns * 1e-9 * self.sample_rate).astype(int)
        
        # Convert tap powers from dB to linear
        self.tap_powers = 10**(self.tap_powers_db / 10)
        self.tap_powers = self.tap_powers / np.sum(self.tap_powers)  # Normalize
    
    def apply(self, signal):
        """Apply multipath channel to signal"""
        max_delay = np.max(self.tap_delays_samples)
        output_length = len(signal) + max_delay
        output = np.zeros(output_length, dtype=signal.dtype)
        
        for delay, power in zip(self.tap_delays_samples, self.tap_powers):
            # Apply delay and power scaling
            delayed_signal = np.sqrt(power) * signal
            output[delay:delay+len(signal)] += delayed_signal
        
        # Return original length (truncate)
        return output[:len(signal)]


class ChannelSimulator:
    """Combines multiple channel effects"""
    
    def __init__(self, sample_rate):
        """Initialize channel simulator"""
        self.sample_rate = sample_rate
        self.channels = []
    
    def add_awgn(self, snr_db):
        """Add AWGN channel"""
        self.channels.append(AWGNChannel(snr_db))
        return self
    
    def add_rayleigh_fading(self, doppler_freq):
        """Add Rayleigh fading"""
        self.channels.append(RayleighChannel(self.sample_rate, doppler_freq))
        return self
    
    def add_rician_fading(self, doppler_freq, k_factor_db):
        """Add Rician fading"""
        self.channels.append(RicianChannel(self.sample_rate, doppler_freq, k_factor_db))
        return self
    
    def add_multipath(self, tap_delays_ns=None, tap_powers_db=None):
        """Add multipath channel"""
        self.channels.append(MultipathChannel(self.sample_rate, tap_delays_ns, tap_powers_db))
        return self
    
    def apply(self, signal):
        """Apply all channel effects in sequence"""
        output = signal.copy()
        for channel in self.channels:
            output = channel.apply(output)
        return output


if __name__ == "__main__":
    # Test channel models
    print("Testing channel models...")
    
    # Generate test signal
    sample_rate = 1e6
    duration = 0.001  # 1ms
    t = np.arange(int(sample_rate * duration)) / sample_rate
    test_signal = np.exp(1j * 2 * np.pi * 1e3 * t)  # 1 kHz complex sinusoid
    
    print(f"Original signal power: {np.mean(np.abs(test_signal)**2):.4f}")
    
    # Test AWGN
    awgn = AWGNChannel(snr_db=10)
    noisy_signal = awgn.apply(test_signal)
    print(f"AWGN (10dB) signal power: {np.mean(np.abs(noisy_signal)**2):.4f}")
    
    # Test Rayleigh fading
    rayleigh = RayleighChannel(sample_rate, doppler_freq=50)
    faded_signal = rayleigh.apply(test_signal)
    print(f"Rayleigh faded signal power: {np.mean(np.abs(faded_signal)**2):.4f}")
    
    # Test Rician fading
    rician = RicianChannel(sample_rate, doppler_freq=50, k_factor_db=6)
    rician_signal = rician.apply(test_signal)
    print(f"Rician faded signal power: {np.mean(np.abs(rician_signal)**2):.4f}")
    
    # Test multipath
    multipath = MultipathChannel(sample_rate)
    multipath_signal = multipath.apply(test_signal)
    print(f"Multipath signal power: {np.mean(np.abs(multipath_signal)**2):.4f}")
    
    # Test combined channel
    combined = ChannelSimulator(sample_rate)
    combined.add_multipath().add_rayleigh_fading(50).add_awgn(10)
    combined_signal = combined.apply(test_signal)
    print(f"Combined channel signal power: {np.mean(np.abs(combined_signal)**2):.4f}")
    
    print("Channel models test completed")