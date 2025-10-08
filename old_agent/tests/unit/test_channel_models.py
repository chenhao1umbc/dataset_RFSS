"""
Unit tests for channel models
Tests AWGN, multipath, fading, and combined channel effects
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from channel_models.basic_channels import ChannelSimulator


class TestChannelSimulator:
    """Test ChannelSimulator class"""
    
    def setup_method(self):
        """Set up test signals and channel simulator"""
        self.sample_rate = 10e6
        self.duration = 0.01
        
        # Create test signal - simple sinusoid
        t = np.arange(int(self.sample_rate * self.duration)) / self.sample_rate
        self.test_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        
        # Create channel simulator
        self.channel = ChannelSimulator(self.sample_rate)
    
    def test_channel_initialization(self):
        """Test channel simulator initialization"""
        channel = ChannelSimulator(self.sample_rate)
        assert channel.sample_rate == self.sample_rate
        assert hasattr(channel, 'effects')
        assert len(channel.effects) == 0  # Should start with no effects
    
    def test_awgn_addition(self):
        """Test AWGN addition"""
        snr_db = 20
        channel = ChannelSimulator(self.sample_rate)
        channel.add_awgn(snr_db)
        
        # Apply AWGN
        noisy_signal = channel.apply(self.test_signal)
        
        # Check basic properties
        assert len(noisy_signal) == len(self.test_signal)
        assert np.iscomplexobj(noisy_signal)
        assert np.all(np.isfinite(noisy_signal))
        
        # Signal should be different from original (noise added)
        assert not np.allclose(noisy_signal, self.test_signal)
        
        # Power should be approximately preserved at high SNR
        original_power = np.mean(np.abs(self.test_signal)**2)
        noisy_power = np.mean(np.abs(noisy_signal)**2)
        power_ratio_db = 10 * np.log10(noisy_power / original_power)
        
        # At 20 dB SNR, power increase should be small
        assert power_ratio_db < 1.0  # Less than 1 dB increase
    
    def test_awgn_snr_levels(self):
        """Test AWGN at different SNR levels"""
        snr_levels = [0, 10, 20, 30]
        original_power = np.mean(np.abs(self.test_signal)**2)
        
        for snr_db in snr_levels:
            channel = ChannelSimulator(self.sample_rate)
            channel.add_awgn(snr_db)
            noisy_signal = channel.apply(self.test_signal)
            
            # Calculate actual noise power
            noise = noisy_signal - self.test_signal
            noise_power = np.mean(np.abs(noise)**2)
            actual_snr_db = 10 * np.log10(original_power / noise_power)
            
            # Should be close to target SNR (within 1 dB)
            assert abs(actual_snr_db - snr_db) < 1.0
    
    def test_multipath_addition(self):
        """Test multipath channel addition"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath()
        
        multipath_signal = channel.apply(self.test_signal)
        
        assert len(multipath_signal) == len(self.test_signal)
        assert np.iscomplexobj(multipath_signal)
        assert np.all(np.isfinite(multipath_signal))
        
        # Multipath should change the signal
        assert not np.allclose(multipath_signal, self.test_signal)
    
    def test_multipath_custom_params(self):
        """Test multipath with custom parameters"""
        delays = [0, 1e-6, 2e-6]  # 0, 1, 2 microsecond delays
        gains = [1.0, 0.5, 0.25]  # Linear gains
        
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath(delays=delays, gains=gains)
        
        multipath_signal = channel.apply(self.test_signal)
        
        assert len(multipath_signal) == len(self.test_signal)
        assert np.all(np.isfinite(multipath_signal))
    
    def test_rayleigh_fading(self):
        """Test Rayleigh fading"""
        doppler_hz = 100
        channel = ChannelSimulator(self.sample_rate)
        channel.add_rayleigh_fading(doppler_hz)
        
        faded_signal = channel.apply(self.test_signal)
        
        assert len(faded_signal) == len(self.test_signal)
        assert np.iscomplexobj(faded_signal)
        assert np.all(np.isfinite(faded_signal))
        
        # Fading should change signal amplitude over time
        original_amplitude = np.abs(self.test_signal)
        faded_amplitude = np.abs(faded_signal)
        
        # Amplitude should vary (not constant)
        assert np.std(faded_amplitude) > np.std(original_amplitude)
    
    def test_rician_fading(self):
        """Test Rician fading"""
        doppler_hz = 50
        k_factor_db = 10
        
        channel = ChannelSimulator(self.sample_rate)
        channel.add_rician_fading(doppler_hz, k_factor_db)
        
        faded_signal = channel.apply(self.test_signal)
        
        assert len(faded_signal) == len(self.test_signal)
        assert np.iscomplexobj(faded_signal)
        assert np.all(np.isfinite(faded_signal))
        
        # Rician fading should have less variation than Rayleigh (due to LOS component)
        faded_amplitude = np.abs(faded_signal)
        amplitude_variation = np.std(faded_amplitude) / np.mean(faded_amplitude)
        
        # Should have some variation but not too much due to strong LOS
        assert 0.1 < amplitude_variation < 1.0
    
    def test_combined_effects(self):
        """Test multiple channel effects combined"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath().add_rayleigh_fading(100).add_awgn(15)
        
        processed_signal = channel.apply(self.test_signal)
        
        assert len(processed_signal) == len(self.test_signal)
        assert np.iscomplexobj(processed_signal)
        assert np.all(np.isfinite(processed_signal))
        
        # Combined effects should significantly change the signal
        correlation = np.abs(np.corrcoef(self.test_signal.real, processed_signal.real)[0, 1])
        assert correlation < 0.9  # Should be decorrelated due to channel effects
    
    def test_channel_chaining(self):
        """Test method chaining for channel effects"""
        # Should be able to chain method calls
        processed_signal = (ChannelSimulator(self.sample_rate)
                          .add_awgn(20)
                          .add_multipath()
                          .add_rayleigh_fading(50)
                          .apply(self.test_signal))
        
        assert len(processed_signal) == len(self.test_signal)
        assert np.all(np.isfinite(processed_signal))
    
    def test_empty_channel(self):
        """Test channel with no effects (should pass signal through)"""
        channel = ChannelSimulator(self.sample_rate)
        output_signal = channel.apply(self.test_signal)
        
        # Should be identical to input (no effects applied)
        np.testing.assert_array_almost_equal(output_signal, self.test_signal)
    
    def test_channel_effects_order(self):
        """Test that channel effects are applied in correct order"""
        # Create two channels with same effects in different order
        channel1 = ChannelSimulator(self.sample_rate)
        channel1.add_awgn(20).add_multipath()
        
        channel2 = ChannelSimulator(self.sample_rate)  
        channel2.add_multipath().add_awgn(20)
        
        # Set same random seed for reproducibility
        np.random.seed(42)
        signal1 = channel1.apply(self.test_signal)
        
        np.random.seed(42)
        signal2 = channel2.apply(self.test_signal)
        
        # Results should be different due to different effect order
        assert not np.allclose(signal1, signal2, rtol=1e-10)
    
    def test_power_preservation(self):
        """Test power preservation through channel (approximately)"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath().add_rayleigh_fading(50)  # No AWGN
        
        output_signal = channel.apply(self.test_signal)
        
        original_power = np.mean(np.abs(self.test_signal)**2)
        output_power = np.mean(np.abs(output_signal)**2)
        power_ratio_db = 10 * np.log10(output_power / original_power)
        
        # Power should be approximately preserved (within 3 dB)
        assert abs(power_ratio_db) < 3.0
    
    def test_signal_length_preservation(self):
        """Test that signal length is preserved through all channel effects"""
        test_lengths = [100, 1000, 10000]
        
        for length in test_lengths:
            test_signal = np.random.randn(length) + 1j * np.random.randn(length)
            
            channel = ChannelSimulator(self.sample_rate)
            channel.add_multipath().add_rayleigh_fading(100).add_awgn(20)
            
            output_signal = channel.apply(test_signal)
            
            assert len(output_signal) == length
    
    def test_zero_doppler_fading(self):
        """Test fading with zero Doppler (static channel)"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_rayleigh_fading(doppler_hz=0)
        
        faded_signal = channel.apply(self.test_signal)
        
        # With zero Doppler, fading should be constant (but random)
        faded_amplitude = np.abs(faded_signal)
        original_amplitude = np.abs(self.test_signal)
        
        # Amplitude should be scaled by constant factor
        scale_factor = faded_amplitude[0] / original_amplitude[0]
        expected_amplitude = original_amplitude * scale_factor
        
        # Should be approximately constant scaling
        np.testing.assert_allclose(faded_amplitude, expected_amplitude, rtol=0.1)
    
    def test_high_snr_awgn(self):
        """Test AWGN at very high SNR"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_awgn(snr_db=60)  # Very high SNR
        
        noisy_signal = channel.apply(self.test_signal)
        
        # At very high SNR, signal should be nearly unchanged
        correlation = np.abs(np.corrcoef(self.test_signal.real, noisy_signal.real)[0, 1])
        assert correlation > 0.99
    
    def test_low_snr_awgn(self):
        """Test AWGN at very low SNR"""
        channel = ChannelSimulator(self.sample_rate)
        channel.add_awgn(snr_db=-10)  # Very low SNR
        
        noisy_signal = channel.apply(self.test_signal)
        
        # At very low SNR, signal should be heavily corrupted
        correlation = np.abs(np.corrcoef(self.test_signal.real, noisy_signal.real)[0, 1])
        assert correlation < 0.5
    
    def test_extreme_parameters(self):
        """Test channel with extreme parameters"""
        # Very high Doppler
        channel = ChannelSimulator(self.sample_rate)
        channel.add_rayleigh_fading(doppler_hz=1000)
        
        faded_signal = channel.apply(self.test_signal)
        assert np.all(np.isfinite(faded_signal))
        
        # Very low K-factor
        channel2 = ChannelSimulator(self.sample_rate)
        channel2.add_rician_fading(doppler_hz=100, k_factor_db=-20)
        
        faded_signal2 = channel2.apply(self.test_signal)
        assert np.all(np.isfinite(faded_signal2))
    
    def test_channel_reproducibility(self):
        """Test channel reproducibility with same random seed"""
        np.random.seed(12345)
        channel1 = ChannelSimulator(self.sample_rate)
        channel1.add_awgn(15).add_rayleigh_fading(100)
        signal1 = channel1.apply(self.test_signal)
        
        np.random.seed(12345)
        channel2 = ChannelSimulator(self.sample_rate)
        channel2.add_awgn(15).add_rayleigh_fading(100)
        signal2 = channel2.apply(self.test_signal)
        
        # Should be identical with same seed
        np.testing.assert_array_almost_equal(signal1, signal2, decimal=10)


class TestChannelEffectsRealism:
    """Test channel effects for realistic behavior"""
    
    def setup_method(self):
        """Set up realistic test scenarios"""
        self.sample_rate = 30.72e6  # LTE sample rate
        self.duration = 0.01
        
        # Create OFDM-like test signal
        t = np.arange(int(self.sample_rate * self.duration)) / self.sample_rate
        num_subcarriers = 1024
        subcarrier_data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        ofdm_symbol = np.fft.ifft(subcarrier_data)
        self.ofdm_signal = np.tile(ofdm_symbol, len(t) // len(ofdm_symbol) + 1)[:len(t)]
        self.ofdm_signal = self.ofdm_signal / np.sqrt(np.mean(np.abs(self.ofdm_signal)**2))
    
    def test_urban_mobile_channel(self):
        """Test urban mobile channel scenario"""
        # Urban: high Doppler, multipath, Rayleigh fading
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath().add_rayleigh_fading(200).add_awgn(5)
        
        urban_signal = channel.apply(self.ofdm_signal)
        
        # Should have significant signal variation
        original_power = np.mean(np.abs(self.ofdm_signal)**2)
        urban_power = np.mean(np.abs(urban_signal)**2)
        
        # Power should increase due to noise
        assert urban_power > original_power
        
        # Signal should be heavily affected
        correlation = np.abs(np.corrcoef(self.ofdm_signal.real, urban_signal.real)[0, 1])
        assert correlation < 0.7  # Significant decorrelation
    
    def test_rural_los_channel(self):
        """Test rural line-of-sight channel scenario"""
        # Rural LOS: low Doppler, Rician fading, higher SNR
        channel = ChannelSimulator(self.sample_rate)
        channel.add_rician_fading(50, k_factor_db=10).add_awgn(15)
        
        rural_signal = channel.apply(self.ofdm_signal)
        
        # Should preserve more of original signal structure
        correlation = np.abs(np.corrcoef(self.ofdm_signal.real, rural_signal.real)[0, 1])
        assert correlation > 0.5  # Better preservation than urban
    
    def test_indoor_nlos_channel(self):
        """Test indoor non-line-of-sight channel scenario"""
        # Indoor NLOS: low Doppler, dense multipath, moderate SNR
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath().add_rayleigh_fading(10).add_awgn(8)
        
        indoor_signal = channel.apply(self.ofdm_signal)
        
        # Should have moderate correlation
        correlation = np.abs(np.corrcoef(self.ofdm_signal.real, indoor_signal.real)[0, 1])
        assert 0.3 < correlation < 0.8
    
    def test_frequency_selective_fading(self):
        """Test frequency-selective fading effects"""
        # Long delay spread should cause frequency selectivity
        delays = np.array([0, 0.5e-6, 1.0e-6, 2.0e-6, 5.0e-6])  # Up to 5 μs
        gains = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
        
        channel = ChannelSimulator(self.sample_rate)
        channel.add_multipath(delays=delays, gains=gains)
        
        multipath_signal = channel.apply(self.ofdm_signal)
        
        # Should cause inter-symbol interference
        assert not np.allclose(multipath_signal, self.ofdm_signal, rtol=1e-3)
        
        # Power should be preserved (approximately)
        original_power = np.mean(np.abs(self.ofdm_signal)**2)
        multipath_power = np.mean(np.abs(multipath_signal)**2)
        power_ratio_db = 10 * np.log10(multipath_power / original_power)
        assert abs(power_ratio_db) < 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])