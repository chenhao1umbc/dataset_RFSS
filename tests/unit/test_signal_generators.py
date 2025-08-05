"""
Unit tests for signal generators
Tests basic functionality, parameter validation, and signal quality
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.nr_generator import NRGenerator
from validation.signal_metrics import SignalAnalyzer, StandardsValidator


class TestGSMGenerator:
    """Test GSM signal generation"""
    
    def test_basic_generation(self):
        """Test basic GSM signal generation"""
        gen = GSMGenerator(sample_rate=10e6, duration=0.001)
        signal = gen.generate_baseband()
        
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex128
        assert len(signal) == int(10e6 * 0.001)
        assert np.all(np.isfinite(signal))
    
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Valid parameters
        gen = GSMGenerator(sample_rate=10e6, duration=0.001, bt_product=0.3)
        assert gen.validate_parameters()
        
        # Invalid sample rate
        with pytest.raises((ValueError, AssertionError)):
            gen = GSMGenerator(sample_rate=-1, duration=0.001)
            gen.validate_parameters()
        
        # Invalid duration
        with pytest.raises((ValueError, AssertionError)):
            gen = GSMGenerator(sample_rate=10e6, duration=-1)
            gen.validate_parameters()
    
    def test_signal_properties(self):
        """Test GSM signal properties"""
        gen = GSMGenerator(sample_rate=10e6, duration=0.01)
        signal = gen.generate_baseband()
        
        # Check power normalization
        avg_power = np.mean(np.abs(signal)**2)
        assert 0.5 < avg_power < 2.0  # Allow some tolerance
        
        # Check signal is complex
        assert np.iscomplexobj(signal)
        
        # Check finite values
        assert np.all(np.isfinite(signal))
    
    def test_metadata(self):
        """Test metadata generation"""
        gen = GSMGenerator(sample_rate=10e6, duration=0.001, bt_product=0.3)
        metadata = gen.get_metadata()
        
        assert isinstance(metadata, dict)
        assert 'signal_type' in metadata
        assert 'sample_rate' in metadata
        assert 'duration' in metadata
        assert metadata['signal_type'] == 'GSM'
        assert metadata['sample_rate'] == 10e6
        assert metadata['duration'] == 0.001
    
    def test_reproducibility(self):
        """Test signal generation reproducibility"""
        np.random.seed(42)
        gen1 = GSMGenerator(sample_rate=10e6, duration=0.001)
        signal1 = gen1.generate_baseband()
        
        np.random.seed(42)
        gen2 = GSMGenerator(sample_rate=10e6, duration=0.001)
        signal2 = gen2.generate_baseband()
        
        np.testing.assert_array_almost_equal(signal1, signal2, decimal=10)


class TestLTEGenerator:
    """Test LTE signal generation"""
    
    def test_basic_generation(self):
        """Test basic LTE signal generation"""
        gen = LTEGenerator(sample_rate=30.72e6, duration=0.001, bandwidth=20)
        signal = gen.generate_baseband()
        
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex128
        assert len(signal) == int(30.72e6 * 0.001)
        assert np.all(np.isfinite(signal))
    
    def test_different_bandwidths(self):
        """Test different LTE bandwidths"""
        sample_rate = 30.72e6
        duration = 0.001
        
        for bw in [5, 10, 15, 20]:
            gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=bw)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_different_modulations(self):
        """Test different modulation schemes"""
        sample_rate = 30.72e6
        duration = 0.001
        
        for mod in ['QPSK', '16QAM', '64QAM', '256QAM']:
            gen = LTEGenerator(sample_rate=sample_rate, duration=duration, 
                             bandwidth=20, modulation=mod)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_signal_properties(self):
        """Test LTE signal properties"""
        gen = LTEGenerator(sample_rate=30.72e6, duration=0.01, bandwidth=20)
        signal = gen.generate_baseband()
        
        # Check power normalization
        avg_power = np.mean(np.abs(signal)**2)
        assert 0.5 < avg_power < 2.0
        
        # Check PAPR is reasonable for OFDM
        peak_power = np.max(np.abs(signal)**2)
        papr_db = 10 * np.log10(peak_power / avg_power)
        assert 3 < papr_db < 20  # OFDM should have higher PAPR than single carrier
    
    def test_standards_compliance(self):
        """Test LTE standards compliance"""
        gen = LTEGenerator(sample_rate=30.72e6, duration=0.01, bandwidth=20)
        signal = gen.generate_baseband()
        
        validator = StandardsValidator()
        results = validator.validate_lte_signal(signal, 30.72e6, expected_bw_mhz=20)
        
        # Check bandwidth is approximately correct (allow some tolerance)
        bw_error = abs(results['measured_bandwidth'] - results['expected_bandwidth'])
        assert bw_error / results['expected_bandwidth'] < 0.5  # 50% tolerance


class TestUMTSGenerator:
    """Test UMTS signal generation"""
    
    def test_basic_generation(self):
        """Test basic UMTS signal generation"""
        gen = UMTSGenerator(sample_rate=15.36e6, duration=0.001, spreading_factor=128)
        signal = gen.generate_baseband()
        
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex128
        assert len(signal) == int(15.36e6 * 0.001)
        assert np.all(np.isfinite(signal))
    
    def test_different_spreading_factors(self):
        """Test different spreading factors"""
        sample_rate = 15.36e6
        duration = 0.001
        
        for sf in [4, 8, 16, 32, 64, 128, 256]:
            gen = UMTSGenerator(sample_rate=sample_rate, duration=duration, 
                              spreading_factor=sf)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_multi_user(self):
        """Test multi-user UMTS generation"""
        gen = UMTSGenerator(sample_rate=15.36e6, duration=0.001, 
                          spreading_factor=128, num_users=4)
        signal = gen.generate_baseband()
        
        assert len(signal) == int(15.36e6 * 0.001)
        assert np.all(np.isfinite(signal))
    
    def test_signal_properties(self):
        """Test UMTS signal properties"""
        gen = UMTSGenerator(sample_rate=15.36e6, duration=0.01, spreading_factor=128)
        signal = gen.generate_baseband()
        
        # Check power normalization
        avg_power = np.mean(np.abs(signal)**2)
        assert 0.5 < avg_power < 2.0
        
        # CDMA should have moderate PAPR
        peak_power = np.max(np.abs(signal)**2)
        papr_db = 10 * np.log10(peak_power / avg_power)
        assert 2 < papr_db < 12


class TestNRGenerator:
    """Test 5G NR signal generation"""
    
    def test_basic_generation(self):
        """Test basic NR signal generation"""
        gen = NRGenerator(sample_rate=122.88e6, duration=0.001, bandwidth=100)
        signal = gen.generate_baseband()
        
        assert isinstance(signal, np.ndarray)
        assert signal.dtype == np.complex128
        assert len(signal) == int(122.88e6 * 0.001)
        assert np.all(np.isfinite(signal))
    
    def test_different_numerologies(self):
        """Test different numerologies"""
        sample_rate = 61.44e6
        duration = 0.001
        
        for numerology in [0, 1, 2]:
            gen = NRGenerator(sample_rate=sample_rate, duration=duration, 
                            bandwidth=50, numerology=numerology)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_different_bandwidths(self):
        """Test different NR bandwidths"""
        sample_rate = 122.88e6
        duration = 0.001
        
        for bw in [10, 20, 50, 100]:
            gen = NRGenerator(sample_rate=sample_rate, duration=duration, bandwidth=bw)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_high_order_modulation(self):
        """Test high-order modulation schemes"""
        sample_rate = 61.44e6
        duration = 0.001
        
        for mod in ['QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']:
            gen = NRGenerator(sample_rate=sample_rate, duration=duration, 
                            bandwidth=50, modulation=mod)
            signal = gen.generate_baseband()
            
            assert len(signal) == int(sample_rate * duration)
            assert np.all(np.isfinite(signal))
    
    def test_signal_properties(self):
        """Test NR signal properties"""
        gen = NRGenerator(sample_rate=61.44e6, duration=0.01, bandwidth=50)
        signal = gen.generate_baseband()
        
        # Check power normalization
        avg_power = np.mean(np.abs(signal)**2)
        assert 0.5 < avg_power < 2.0
        
        # 5G NR with high-order QAM should have high PAPR
        peak_power = np.max(np.abs(signal)**2)
        papr_db = 10 * np.log10(peak_power / avg_power)
        assert 5 < papr_db < 25


class TestSignalQuality:
    """Test signal quality across all generators"""
    
    @pytest.mark.parametrize("generator_class,params", [
        (GSMGenerator, {'sample_rate': 10e6, 'duration': 0.01}),
        (LTEGenerator, {'sample_rate': 30.72e6, 'duration': 0.01, 'bandwidth': 20}),
        (UMTSGenerator, {'sample_rate': 15.36e6, 'duration': 0.01, 'spreading_factor': 128}),
        (NRGenerator, {'sample_rate': 61.44e6, 'duration': 0.01, 'bandwidth': 50})
    ])
    def test_signal_quality_metrics(self, generator_class, params):
        """Test signal quality metrics for all generators"""
        gen = generator_class(**params)
        signal = gen.generate_baseband()
        
        analyzer = SignalAnalyzer(params['sample_rate'])
        
        # Test power metrics
        power_metrics = analyzer.calculate_power_metrics(signal)
        assert np.isfinite(power_metrics['avg_power'])
        assert np.isfinite(power_metrics['peak_power'])
        assert np.isfinite(power_metrics['papr'])
        assert power_metrics['avg_power'] > 0
        assert power_metrics['peak_power'] >= power_metrics['avg_power']
        
        # Test bandwidth calculation
        bw_metrics = analyzer.calculate_bandwidth(signal)
        assert np.isfinite(bw_metrics['bandwidth'])
        assert bw_metrics['bandwidth'] > 0
        
        # Test SNR estimation (should be high for clean generated signals)
        snr_metrics = analyzer.calculate_snr(signal)
        assert np.isfinite(snr_metrics['snr_db'])
        assert snr_metrics['snr_db'] > 20  # Clean generated signal should have high SNR
    
    def test_signal_length_consistency(self):
        """Test that all generators produce expected signal lengths"""
        duration = 0.005  # 5 ms
        
        test_cases = [
            (GSMGenerator, {'sample_rate': 10e6}),
            (LTEGenerator, {'sample_rate': 30.72e6, 'bandwidth': 20}),
            (UMTSGenerator, {'sample_rate': 15.36e6, 'spreading_factor': 128}),
            (NRGenerator, {'sample_rate': 61.44e6, 'bandwidth': 50})
        ]
        
        for generator_class, params in test_cases:
            params['duration'] = duration
            gen = generator_class(**params)
            signal = gen.generate_baseband()
            
            expected_length = int(params['sample_rate'] * duration)
            assert len(signal) == expected_length, \
                f"{generator_class.__name__} produced {len(signal)} samples, expected {expected_length}"
    
    def test_power_consistency(self):
        """Test power consistency across different signal types"""
        duration = 0.01
        target_power_db = 0  # 0 dB (unit power)
        
        generators = [
            GSMGenerator(sample_rate=10e6, duration=duration, power_db=target_power_db),
            LTEGenerator(sample_rate=30.72e6, duration=duration, bandwidth=20, power_db=target_power_db),
            UMTSGenerator(sample_rate=15.36e6, duration=duration, spreading_factor=128, power_db=target_power_db),
            NRGenerator(sample_rate=61.44e6, duration=duration, bandwidth=50, power_db=target_power_db)
        ]
        
        for gen in generators:
            signal = gen.generate_baseband()
            avg_power_db = 10 * np.log10(np.mean(np.abs(signal)**2))
            
            # Allow 1 dB tolerance for power normalization
            assert abs(avg_power_db - target_power_db) < 1.0, \
                f"{gen.__class__.__name__} power error: {avg_power_db:.2f} dB vs {target_power_db} dB"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])