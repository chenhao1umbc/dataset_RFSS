"""
Signal quality metrics and validation tools
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Dict, Tuple, Optional


class SignalAnalyzer:
    """Analyze RF signal quality and characteristics"""
    
    def __init__(self, sample_rate):
        """
        Initialize signal analyzer
        
        Args:
            sample_rate: Sampling rate in Hz
        """
        self.sample_rate = sample_rate
    
    def calculate_power_metrics(self, signal_data):
        """Calculate power-related metrics"""
        # Average power
        avg_power = np.mean(np.abs(signal_data)**2)
        avg_power_db = 10 * np.log10(avg_power) if avg_power > 0 else -np.inf
        
        # Peak power
        peak_power = np.max(np.abs(signal_data)**2)
        peak_power_db = 10 * np.log10(peak_power) if peak_power > 0 else -np.inf
        
        # PAPR (Peak-to-Average Power Ratio)
        papr = peak_power / avg_power if avg_power > 0 else np.inf
        papr_db = 10 * np.log10(papr) if papr > 0 else -np.inf
        
        # RMS power
        rms_power = np.sqrt(avg_power)
        
        return {
            'avg_power': avg_power,
            'avg_power_db': avg_power_db,
            'peak_power': peak_power,
            'peak_power_db': peak_power_db,
            'papr': papr,
            'papr_db': papr_db,
            'rms_power': rms_power
        }
    
    def calculate_bandwidth(self, signal_data, threshold_db=-3, method='psd'):
        """
        Calculate signal bandwidth using multiple methods
        
        Args:
            signal_data: Complex baseband signal
            threshold_db: Power threshold below peak (default -3dB)
            method: 'psd' (power spectral density) or 'rms' (RMS bandwidth)
        """
        if method == 'psd':
            # Use Welch's method for better spectral estimation
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate, 
                                    nperseg=min(1024, len(signal_data)//4))
            
            # Convert to dB, handle complex signals properly
            if np.iscomplexobj(signal_data):
                psd_linear = np.abs(psd)
            else:
                psd_linear = psd
                
            psd_db = 10 * np.log10(psd_linear + 1e-12)
            
            # Find peak (center around DC for baseband)
            center_idx = len(freqs) // 2
            peak_idx = np.argmax(psd_db)
            peak_power = psd_db[peak_idx]
            
            # Find bandwidth at threshold below peak
            threshold_power = peak_power + threshold_db
            
            # Find frequencies where power is above threshold
            above_threshold = psd_db >= threshold_power
            
            if np.any(above_threshold):
                freq_indices = np.where(above_threshold)[0]
                bw_start_freq = freqs[freq_indices[0]]
                bw_end_freq = freqs[freq_indices[-1]]
                bandwidth = bw_end_freq - bw_start_freq
                center_freq = freqs[peak_idx]
            else:
                bandwidth = 0
                bw_start_freq = 0
                bw_end_freq = 0
                center_freq = 0
                
        elif method == 'rms':
            # RMS bandwidth calculation
            freqs, psd = signal.welch(signal_data, fs=self.sample_rate,
                                    nperseg=min(1024, len(signal_data)//4))
            
            if np.iscomplexobj(signal_data):
                psd_linear = np.abs(psd)
            else:
                psd_linear = psd
                
            # Calculate RMS bandwidth
            total_power = np.trapz(psd_linear, freqs)
            if total_power > 0:
                center_freq = np.trapz(freqs * psd_linear, freqs) / total_power
                second_moment = np.trapz((freqs - center_freq)**2 * psd_linear, freqs) / total_power
                bandwidth = 2 * np.sqrt(second_moment)  # RMS bandwidth
                bw_start_freq = center_freq - bandwidth/2
                bw_end_freq = center_freq + bandwidth/2
            else:
                bandwidth = bw_start_freq = bw_end_freq = center_freq = 0
                
            psd_db = 10 * np.log10(psd_linear + 1e-12)
        
        return {
            'bandwidth': abs(bandwidth),  # Ensure positive bandwidth
            'start_freq': bw_start_freq,
            'end_freq': bw_end_freq,
            'center_freq': center_freq,
            'freqs': freqs,
            'psd_db': psd_db,
            'method': method
        }
    
    def calculate_evm(self, tx_symbols, rx_symbols):
        """Calculate Error Vector Magnitude"""
        if len(tx_symbols) != len(rx_symbols):
            min_len = min(len(tx_symbols), len(rx_symbols))
            tx_symbols = tx_symbols[:min_len]
            rx_symbols = rx_symbols[:min_len]
        
        # Error vector
        error_vector = rx_symbols - tx_symbols
        
        # EVM calculation
        error_power = np.mean(np.abs(error_vector)**2)
        signal_power = np.mean(np.abs(tx_symbols)**2)
        
        evm_rms = np.sqrt(error_power / signal_power) if signal_power > 0 else np.inf
        evm_rms_percent = evm_rms * 100
        evm_rms_db = 20 * np.log10(evm_rms) if evm_rms > 0 else -np.inf
        
        return {
            'evm_rms': evm_rms,
            'evm_rms_percent': evm_rms_percent,
            'evm_rms_db': evm_rms_db,
            'error_power': error_power,
            'signal_power': signal_power
        }
    
    def calculate_snr(self, signal_data, noise_floor=None):
        """Estimate SNR from signal"""
        if noise_floor is None:
            # Estimate noise floor from signal
            # Simple method: assume lowest 10% of power samples are noise
            power_samples = np.abs(signal_data)**2
            sorted_power = np.sort(power_samples)
            noise_floor = np.mean(sorted_power[:len(sorted_power)//10])
        
        signal_power = np.mean(np.abs(signal_data)**2)
        noise_power = noise_floor
        
        snr_linear = signal_power / noise_power if noise_power > 0 else np.inf
        snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf
        
        return {
            'snr_linear': snr_linear,
            'snr_db': snr_db,
            'signal_power': signal_power,
            'noise_power': noise_power
        }
    
    def detect_peaks(self, signal_data, min_distance=100):
        """Detect peaks in signal spectrum"""
        freqs, psd = signal.periodogram(signal_data, fs=self.sample_rate)
        psd_db = 10 * np.log10(psd + 1e-12)
        
        # Find peaks
        peaks, properties = signal.find_peaks(psd_db, distance=min_distance)
        
        peak_freqs = freqs[peaks]
        peak_powers = psd_db[peaks]
        
        return {
            'peak_frequencies': peak_freqs,
            'peak_powers': peak_powers,
            'num_peaks': len(peaks),
            'freqs': freqs,
            'psd_db': psd_db
        }


class StandardsValidator:
    """Validate signals against cellular standards"""
    
    def __init__(self):
        """Initialize standards validator"""
        pass
    
    def validate_gsm_signal(self, signal_data, sample_rate):
        """Validate GSM signal characteristics"""
        analyzer = SignalAnalyzer(sample_rate)
        
        # Expected GSM parameters
        expected_bw = 200e3  # 200 kHz
        expected_papr_max = 2  # dB (GMSK has low PAPR)
        
        # Calculate metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        
        # Validation results
        results = {
            'bandwidth_valid': abs(bw_metrics['bandwidth'] - expected_bw) / expected_bw < 0.2,
            'papr_valid': power_metrics['papr_db'] <= expected_papr_max,
            'measured_bandwidth': bw_metrics['bandwidth'],
            'expected_bandwidth': expected_bw,
            'measured_papr': power_metrics['papr_db'],
            'expected_papr_max': expected_papr_max
        }
        
        results['overall_valid'] = results['bandwidth_valid'] and results['papr_valid']
        
        return results
    
    def validate_lte_signal(self, signal_data, sample_rate, expected_bw_mhz=20):
        """Validate LTE signal characteristics"""
        analyzer = SignalAnalyzer(sample_rate)
        
        # Expected LTE parameters
        expected_bw = expected_bw_mhz * 1e6
        expected_papr_min = 6  # dB (OFDM has higher PAPR)
        expected_papr_max = 15  # dB
        
        # Calculate metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        
        # Validation results
        results = {
            'bandwidth_valid': abs(bw_metrics['bandwidth'] - expected_bw) / expected_bw < 0.3,
            'papr_valid': expected_papr_min <= power_metrics['papr_db'] <= expected_papr_max,
            'measured_bandwidth': bw_metrics['bandwidth'],
            'expected_bandwidth': expected_bw,
            'measured_papr': power_metrics['papr_db'],
            'expected_papr_range': [expected_papr_min, expected_papr_max]
        }
        
        results['overall_valid'] = results['bandwidth_valid'] and results['papr_valid']
        
        return results
    
    def validate_umts_signal(self, signal_data, sample_rate):
        """Validate UMTS signal characteristics"""
        analyzer = SignalAnalyzer(sample_rate)
        
        # Expected UMTS parameters
        expected_bw = 5e6  # 5 MHz
        expected_papr_min = 3  # dB (CDMA has moderate PAPR)
        expected_papr_max = 8  # dB
        
        # Calculate metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        
        # Validation results
        results = {
            'bandwidth_valid': abs(bw_metrics['bandwidth'] - expected_bw) / expected_bw < 0.3,
            'papr_valid': expected_papr_min <= power_metrics['papr_db'] <= expected_papr_max,
            'measured_bandwidth': bw_metrics['bandwidth'],
            'expected_bandwidth': expected_bw,
            'measured_papr': power_metrics['papr_db'],
            'expected_papr_range': [expected_papr_min, expected_papr_max]
        }
        
        results['overall_valid'] = results['bandwidth_valid'] and results['papr_valid']
        
        return results
    
    def validate_nr_signal(self, signal_data, sample_rate, expected_bw_mhz=100):
        """Validate 5G NR signal characteristics"""
        analyzer = SignalAnalyzer(sample_rate)
        
        # Expected NR parameters
        expected_bw = expected_bw_mhz * 1e6
        expected_papr_min = 8  # dB (OFDM with high-order QAM)
        expected_papr_max = 18  # dB
        
        # Calculate metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        
        # Validation results
        results = {
            'bandwidth_valid': abs(bw_metrics['bandwidth'] - expected_bw) / expected_bw < 0.3,
            'papr_valid': expected_papr_min <= power_metrics['papr_db'] <= expected_papr_max,
            'measured_bandwidth': bw_metrics['bandwidth'],
            'expected_bandwidth': expected_bw,
            'measured_papr': power_metrics['papr_db'],
            'expected_papr_range': [expected_papr_min, expected_papr_max]
        }
        
        results['overall_valid'] = results['bandwidth_valid'] and results['papr_valid']
        
        return results


class ValidationReport:
    """Generate validation reports"""
    
    def __init__(self):
        """Initialize validation report generator"""
        pass
    
    def generate_signal_report(self, signal_data, sample_rate, signal_type='Unknown'):
        """Generate comprehensive signal analysis report"""
        analyzer = SignalAnalyzer(sample_rate)
        validator = StandardsValidator()
        
        # Calculate all metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        snr_metrics = analyzer.calculate_snr(signal_data)
        peak_metrics = analyzer.detect_peaks(signal_data)
        
        # Validate against standards if type is known
        validation_results = None
        if signal_type.lower() in ['gsm', '2g']:
            validation_results = validator.validate_gsm_signal(signal_data, sample_rate)
        elif signal_type.lower() in ['lte', '4g']:
            validation_results = validator.validate_lte_signal(signal_data, sample_rate)
        elif signal_type.lower() in ['umts', '3g']:
            validation_results = validator.validate_umts_signal(signal_data, sample_rate)
        elif signal_type.lower() in ['nr', '5g']:
            validation_results = validator.validate_nr_signal(signal_data, sample_rate)
        
        # Compile report
        report = {
            'signal_type': signal_type,
            'sample_rate': sample_rate,
            'signal_length': len(signal_data),
            'duration': len(signal_data) / sample_rate,
            'power_metrics': power_metrics,
            'bandwidth_metrics': bw_metrics,
            'snr_metrics': snr_metrics,
            'peak_metrics': peak_metrics,
            'validation_results': validation_results
        }
        
        return report
    
    def print_report(self, report):
        """Print validation report to console"""
        print(f"\n=== Signal Validation Report ===")
        print(f"Signal Type: {report['signal_type']}")
        print(f"Sample Rate: {report['sample_rate']/1e6:.2f} MHz")
        print(f"Duration: {report['duration']*1000:.2f} ms")
        print(f"Length: {report['signal_length']} samples")
        
        print(f"\nPower Metrics:")
        pm = report['power_metrics']
        print(f"  Average Power: {pm['avg_power_db']:.2f} dB")
        print(f"  Peak Power: {pm['peak_power_db']:.2f} dB")
        print(f"  PAPR: {pm['papr_db']:.2f} dB")
        
        print(f"\nBandwidth Metrics:")
        bm = report['bandwidth_metrics']
        print(f"  Bandwidth: {bm['bandwidth']/1e6:.3f} MHz")
        print(f"  Center Frequency: {bm['center_freq']/1e6:.3f} MHz")
        
        print(f"\nSNR Metrics:")
        sm = report['snr_metrics']
        print(f"  Estimated SNR: {sm['snr_db']:.2f} dB")
        
        print(f"\nSpectral Peaks:")
        pem = report['peak_metrics']
        print(f"  Number of peaks: {pem['num_peaks']}")
        if pem['num_peaks'] > 0:
            for i, (freq, power) in enumerate(zip(pem['peak_frequencies'][:5], 
                                                 pem['peak_powers'][:5])):
                print(f"    Peak {i+1}: {freq/1e6:.3f} MHz, {power:.2f} dB")
        
        if report['validation_results']:
            print(f"\nStandards Validation:")
            vr = report['validation_results']
            print(f"  Overall Valid: {vr['overall_valid']}")
            print(f"  Bandwidth Valid: {vr['bandwidth_valid']}")
            print(f"  PAPR Valid: {vr['papr_valid']}")
            print(f"  Measured BW: {vr['measured_bandwidth']/1e6:.3f} MHz")
            print(f"  Expected BW: {vr['expected_bandwidth']/1e6:.3f} MHz")
            print(f"  Measured PAPR: {vr['measured_papr']:.2f} dB")


if __name__ == "__main__":
    # Test validation framework
    print("Testing validation framework...")
    
    # Generate test signals
    sample_rate = 10e6
    duration = 0.01
    t = np.arange(int(sample_rate * duration)) / sample_rate
    
    # Test GSM-like signal (narrow bandwidth, low PAPR)
    gsm_signal = np.exp(1j * 2 * np.pi * 1e3 * t)  # Simple tone
    
    # Test LTE-like signal (wide bandwidth, higher PAPR) 
    lte_signal = np.random.randn(len(t)) + 1j * np.random.randn(len(t))
    
    # Create validation report
    validator = ValidationReport()
    
    # Test GSM validation
    gsm_report = validator.generate_signal_report(gsm_signal, sample_rate, 'GSM')
    validator.print_report(gsm_report)
    
    # Test LTE validation  
    lte_report = validator.generate_signal_report(lte_signal, sample_rate, 'LTE')
    validator.print_report(lte_report)
    
    print("Validation framework test completed")