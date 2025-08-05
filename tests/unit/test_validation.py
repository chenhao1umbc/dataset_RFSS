"""
Unit tests for signal validation framework
Tests signal analysis, standards validation, and reporting
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from validation.signal_metrics import SignalAnalyzer, StandardsValidator, ValidationReport


class TestSignalAnalyzer:
    """Test SignalAnalyzer class"""
    
    def setup_method(self):
        """Set up test signals"""
        self.sample_rate = 10e6
        self.duration = 0.01
        self.analyzer = SignalAnalyzer(self.sample_rate)
        
        # Create test signals
        t = np.arange(int(self.sample_rate * self.duration)) / self.sample_rate
        
        # Simple sinusoid
        self.sine_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
        
        # Noisy signal
        self.noisy_signal = self.sine_signal + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        
        # Wideband signal
        self.wideband_signal = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    
    def test_power_metrics_sine(self):
        """Test power metrics with sine wave"""
        metrics = self.analyzer.calculate_power_metrics(self.sine_signal)
        
        # Check all metrics are finite
        assert np.isfinite(metrics['avg_power'])
        assert np.isfinite(metrics['peak_power'])
        assert np.isfinite(metrics['papr'])
        assert np.isfinite(metrics['rms_power'])
        
        # For constant amplitude signal, peak and average should be similar
        assert abs(metrics['papr_db']) < 1.0  # PAPR should be very low for sine wave
        
        # Power should be approximately 1 (unit power)
        assert 0.8 < metrics['avg_power'] < 1.2
    
    def test_power_metrics_noise(self):
        """Test power metrics with noise signal"""
        metrics = self.analyzer.calculate_power_metrics(self.wideband_signal)
        
        # Gaussian noise should have higher PAPR than sine wave
        assert metrics['papr_db'] > 3.0
        assert metrics['papr_db'] < 15.0  # But not unreasonably high
        
        # All metrics should be finite and positive
        assert np.isfinite(metrics['avg_power']) and metrics['avg_power'] > 0
        assert np.isfinite(metrics['peak_power']) and metrics['peak_power'] > 0
    
    def test_bandwidth_calculation_psd(self):
        """Test bandwidth calculation using PSD method"""
        # Test with sine wave (should have narrow bandwidth)
        bw_metrics = self.analyzer.calculate_bandwidth(self.sine_signal, method='psd')
        
        assert np.isfinite(bw_metrics['bandwidth'])
        assert bw_metrics['bandwidth'] > 0
        assert bw_metrics['method'] == 'psd'
        
        # Sine wave should have narrow bandwidth
        assert bw_metrics['bandwidth'] < self.sample_rate / 10
    
    def test_bandwidth_calculation_rms(self):
        """Test bandwidth calculation using RMS method"""
        bw_metrics = self.analyzer.calculate_bandwidth(self.wideband_signal, method='rms')
        
        assert np.isfinite(bw_metrics['bandwidth'])
        assert bw_metrics['bandwidth'] > 0
        assert bw_metrics['method'] == 'rms'
        
        # Wideband signal should have larger bandwidth
        assert bw_metrics['bandwidth'] > self.sample_rate / 100
    
    def test_bandwidth_threshold_sensitivity(self):
        """Test bandwidth calculation with different thresholds"""
        # Test different threshold values
        for threshold in [-1, -3, -6, -10]:
            bw_metrics = self.analyzer.calculate_bandwidth(self.sine_signal, threshold_db=threshold)
            assert np.isfinite(bw_metrics['bandwidth'])
            assert bw_metrics['bandwidth'] > 0
    
    def test_evm_calculation(self):
        """Test Error Vector Magnitude calculation"""
        # Create reference and received symbols
        tx_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j] * 100)  # QPSK symbols
        
        # Add small error
        rx_symbols = tx_symbols + 0.05 * (np.random.randn(len(tx_symbols)) + 1j * np.random.randn(len(tx_symbols)))
        
        evm_metrics = self.analyzer.calculate_evm(tx_symbols, rx_symbols)
        
        assert np.isfinite(evm_metrics['evm_rms'])
        assert np.isfinite(evm_metrics['evm_rms_percent'])
        assert np.isfinite(evm_metrics['evm_rms_db'])
        
        # EVM should be reasonable for added error
        assert 0 < evm_metrics['evm_rms_percent'] < 50
        assert evm_metrics['evm_rms'] > 0
    
    def test_evm_perfect_symbols(self):
        """Test EVM with perfect symbols (should be zero)"""
        symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j] * 100)
        evm_metrics = self.analyzer.calculate_evm(symbols, symbols)
        
        # EVM should be very small for identical symbols
        assert evm_metrics['evm_rms_percent'] < 0.1
    
    def test_evm_length_mismatch(self):
        """Test EVM handles length mismatch"""
        tx_symbols = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        rx_symbols = np.array([1+1j, 1-1j])  # Shorter
        
        evm_metrics = self.analyzer.calculate_evm(tx_symbols, rx_symbols)
        
        # Should handle gracefully and return finite results
        assert np.isfinite(evm_metrics['evm_rms'])
    
    def test_snr_estimation(self):
        """Test SNR estimation"""
        # Test with clean signal (high SNR)
        snr_metrics = self.analyzer.calculate_snr(self.sine_signal)
        
        assert np.isfinite(snr_metrics['snr_db'])
        assert snr_metrics['snr_db'] > 20  # Clean signal should have high SNR
        
        # Test with noisy signal (lower SNR)
        snr_noisy = self.analyzer.calculate_snr(self.noisy_signal)
        
        assert np.isfinite(snr_noisy['snr_db'])
        assert snr_noisy['snr_db'] < snr_metrics['snr_db']  # Noisy signal should have lower SNR
    
    def test_snr_with_known_noise_floor(self):
        """Test SNR with known noise floor"""
        noise_power = 0.01
        snr_metrics = self.analyzer.calculate_snr(self.sine_signal, noise_floor=noise_power)
        
        expected_snr = 10 * np.log10(1.0 / noise_power)  # Assuming unit signal power
        
        # Should be close to expected value (within 3 dB tolerance)
        assert abs(snr_metrics['snr_db'] - expected_snr) < 3.0
    
    def test_peak_detection(self):
        """Test spectral peak detection"""
        peak_metrics = self.analyzer.detect_peaks(self.sine_signal)
        
        assert 'peak_frequencies' in peak_metrics
        assert 'peak_powers' in peak_metrics
        assert 'num_peaks' in peak_metrics
        
        # Should detect at least one peak
        assert peak_metrics['num_peaks'] > 0
        
        # Peak frequency should be around 1 MHz (our test frequency)
        if peak_metrics['num_peaks'] > 0:
            main_peak_freq = peak_metrics['peak_frequencies'][0]
            assert abs(main_peak_freq - 1e6) < 100e3  # Within 100 kHz


class TestStandardsValidator:
    """Test StandardsValidator class"""
    
    def setup_method(self):
        """Set up validator"""
        self.validator = StandardsValidator()
    
    def create_test_signal(self, signal_type, sample_rate, duration=0.01):
        """Create test signals with approximate characteristics"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        
        if signal_type == 'GSM':
            # Narrow bandwidth, low PAPR signal
            freq = 100e3  # 100 kHz tone
            signal = np.exp(1j * 2 * np.pi * freq * t)
            # Add some bandwidth by modulating
            mod_data = np.random.randint(0, 2, len(t)) * 2 - 1
            signal *= mod_data
            
        elif signal_type == 'LTE':
            # Wide bandwidth, higher PAPR OFDM-like signal
            num_subcarriers = 1024
            subcarrier_data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
            ofdm_symbol = np.fft.ifft(subcarrier_data)
            # Repeat to fill duration
            signal = np.tile(ofdm_symbol, len(t) // len(ofdm_symbol) + 1)[:len(t)]
            
        elif signal_type == 'UMTS':
            # Moderate bandwidth, moderate PAPR CDMA-like signal
            signal = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
            # Apply some spreading
            spreading_code = np.random.choice([-1, 1], 128)
            spread_signal = np.repeat(spreading_code, len(t) // len(spreading_code) + 1)[:len(t)]
            signal *= spread_signal
            
        elif signal_type == 'NR':
            # Wide bandwidth, high PAPR 5G-like signal
            num_subcarriers = 2048
            subcarrier_data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
            # Higher order QAM increases PAPR
            subcarrier_data = np.sign(subcarrier_data) * (1 + 0.5 * np.abs(subcarrier_data))
            ofdm_symbol = np.fft.ifft(subcarrier_data)
            signal = np.tile(ofdm_symbol, len(t) // len(ofdm_symbol) + 1)[:len(t)]
        
        # Normalize power
        signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
        return signal
    
    def test_gsm_validation(self):
        """Test GSM signal validation"""
        sample_rate = 10e6
        signal = self.create_test_signal('GSM', sample_rate)
        
        results = self.validator.validate_gsm_signal(signal, sample_rate)
        
        assert isinstance(results, dict)
        assert 'bandwidth_valid' in results
        assert 'papr_valid' in results
        assert 'overall_valid' in results
        assert 'measured_bandwidth' in results
        assert 'measured_papr' in results
        
        # Results should be boolean or numeric
        assert isinstance(results['bandwidth_valid'], bool)
        assert isinstance(results['papr_valid'], bool)
        assert isinstance(results['overall_valid'], bool)
        assert np.isfinite(results['measured_bandwidth'])
        assert np.isfinite(results['measured_papr'])
    
    def test_lte_validation(self):
        """Test LTE signal validation"""
        sample_rate = 30.72e6
        signal = self.create_test_signal('LTE', sample_rate)
        
        results = self.validator.validate_lte_signal(signal, sample_rate, expected_bw_mhz=20)
        
        assert isinstance(results, dict)
        assert 'bandwidth_valid' in results
        assert 'papr_valid' in results
        assert 'overall_valid' in results
        
        # LTE should have higher PAPR than GSM
        assert results['measured_papr'] > 3.0
    
    def test_umts_validation(self):
        """Test UMTS signal validation"""
        sample_rate = 15.36e6
        signal = self.create_test_signal('UMTS', sample_rate)
        
        results = self.validator.validate_umts_signal(signal, sample_rate)
        
        assert isinstance(results, dict)
        assert 'bandwidth_valid' in results
        assert 'papr_valid' in results
        assert np.isfinite(results['measured_bandwidth'])
        assert np.isfinite(results['measured_papr'])
    
    def test_nr_validation(self):
        """Test 5G NR signal validation"""
        sample_rate = 122.88e6
        signal = self.create_test_signal('NR', sample_rate)
        
        results = self.validator.validate_nr_signal(signal, sample_rate, expected_bw_mhz=100)
        
        assert isinstance(results, dict)
        assert 'bandwidth_valid' in results
        assert 'papr_valid' in results
        
        # 5G NR should have high PAPR
        assert results['measured_papr'] > 6.0
    
    def test_validation_edge_cases(self):
        """Test validation with edge cases"""
        sample_rate = 10e6
        
        # Zero signal
        zero_signal = np.zeros(1000, dtype=complex)
        results = self.validator.validate_gsm_signal(zero_signal, sample_rate)
        assert isinstance(results, dict)
        
        # Very short signal
        short_signal = np.random.randn(10) + 1j * np.random.randn(10)
        results = self.validator.validate_gsm_signal(short_signal, sample_rate)
        assert isinstance(results, dict)


class TestValidationReport:
    """Test ValidationReport class"""
    
    def setup_method(self):
        """Set up validation report generator"""
        self.reporter = ValidationReport()
        self.sample_rate = 10e6
        
        # Create test signal
        t = np.arange(int(self.sample_rate * 0.01)) / self.sample_rate
        self.test_signal = np.exp(1j * 2 * np.pi * 1e6 * t)
    
    def test_generate_signal_report(self):
        """Test signal report generation"""
        report = self.reporter.generate_signal_report(
            self.test_signal, 
            self.sample_rate, 
            'GSM'
        )
        
        assert isinstance(report, dict)
        
        # Check required keys
        required_keys = [
            'signal_type', 'sample_rate', 'signal_length', 'duration',
            'power_metrics', 'bandwidth_metrics', 'snr_metrics', 'peak_metrics'
        ]
        for key in required_keys:
            assert key in report
        
        # Check data types and values
        assert report['signal_type'] == 'GSM'
        assert report['sample_rate'] == self.sample_rate
        assert report['signal_length'] == len(self.test_signal)
        assert report['duration'] == len(self.test_signal) / self.sample_rate
        
        # Check nested dictionaries exist
        assert isinstance(report['power_metrics'], dict)
        assert isinstance(report['bandwidth_metrics'], dict)
        assert isinstance(report['snr_metrics'], dict)
        assert isinstance(report['peak_metrics'], dict)
    
    def test_report_with_validation(self):
        """Test report generation with standards validation"""
        report = self.reporter.generate_signal_report(
            self.test_signal,
            self.sample_rate,
            'GSM'
        )
        
        # Should include validation results for known signal type
        assert 'validation_results' in report
        if report['validation_results'] is not None:
            assert isinstance(report['validation_results'], dict)
            assert 'overall_valid' in report['validation_results']
    
    def test_report_unknown_signal_type(self):
        """Test report generation with unknown signal type"""
        report = self.reporter.generate_signal_report(
            self.test_signal,
            self.sample_rate,
            'Unknown'
        )
        
        # Should not include validation for unknown type
        assert report['validation_results'] is None
    
    def test_print_report(self):
        """Test report printing (should not raise exceptions)"""
        report = self.reporter.generate_signal_report(
            self.test_signal,
            self.sample_rate,
            'LTE'
        )
        
        # Should not raise any exceptions
        try:
            self.reporter.print_report(report)
        except Exception as e:
            pytest.fail(f"print_report raised an exception: {e}")
    
    def test_report_metrics_validity(self):
        """Test that all metrics in report are valid"""
        report = self.reporter.generate_signal_report(
            self.test_signal,
            self.sample_rate,
            'GSM'
        )
        
        # Check power metrics
        pm = report['power_metrics']
        assert np.isfinite(pm['avg_power_db'])
        assert np.isfinite(pm['peak_power_db'])
        assert np.isfinite(pm['papr_db'])
        
        # Check bandwidth metrics
        bm = report['bandwidth_metrics']
        assert np.isfinite(bm['bandwidth'])
        assert bm['bandwidth'] > 0
        
        # Check SNR metrics
        sm = report['snr_metrics']
        assert np.isfinite(sm['snr_db'])
        
        # Check peak metrics
        pem = report['peak_metrics']
        assert isinstance(pem['num_peaks'], int)
        assert pem['num_peaks'] >= 0


class TestValidationIntegration:
    """Integration tests for validation framework"""
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline"""
        # Create analyzer and validator
        sample_rate = 30.72e6
        analyzer = SignalAnalyzer(sample_rate)
        validator = StandardsValidator()
        reporter = ValidationReport()
        
        # Create test signal
        duration = 0.01
        t = np.arange(int(sample_rate * duration)) / sample_rate
        
        # OFDM-like signal for LTE
        num_subcarriers = 1024
        subcarrier_data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        ofdm_symbol = np.fft.ifft(subcarrier_data)
        signal = np.tile(ofdm_symbol, len(t) // len(ofdm_symbol) + 1)[:len(t)]
        signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
        
        # Run complete pipeline
        power_metrics = analyzer.calculate_power_metrics(signal)
        bw_metrics = analyzer.calculate_bandwidth(signal)
        validation_results = validator.validate_lte_signal(signal, sample_rate, 20)
        report = reporter.generate_signal_report(signal, sample_rate, 'LTE')
        
        # All should succeed without exceptions
        assert isinstance(power_metrics, dict)
        assert isinstance(bw_metrics, dict)
        assert isinstance(validation_results, dict)
        assert isinstance(report, dict)
        
        # Report should contain all analysis results
        assert 'power_metrics' in report
        assert 'bandwidth_metrics' in report
        assert 'validation_results' in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])