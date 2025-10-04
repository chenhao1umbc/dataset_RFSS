#!/usr/bin/env python3
"""
Comprehensive Data Quality Validation for RFSS Dataset

This script validates the quality of generated RF signals to ensure:
1. Signal power and spectral characteristics are correct
2. 3GPP standards compliance
3. Modulation constellation quality
4. Channel modeling accuracy
5. Statistical distributions are reasonable

Run this BEFORE using data for machine learning to avoid garbage-in-garbage-out.
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats, signal
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from validation.signal_metrics import SignalAnalyzer, StandardsValidator
from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator


class DataQualityValidator:
    """Comprehensive data quality validation for RF signal dataset"""

    def __init__(self, output_dir='data/quality_reports'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def validate_signal_standard(self, signal_data, sample_rate, standard,
                                 bandwidth_mhz=None):
        """
        Validate signal against 3GPP standards

        Args:
            signal_data: Complex baseband signal
            sample_rate: Sampling rate in Hz
            standard: 'gsm', 'umts', 'lte', or 'nr'
            bandwidth_mhz: Required for LTE/NR
        """
        print(f"\nValidating {standard.upper()} signal...")

        analyzer = SignalAnalyzer(sample_rate)
        validator = StandardsValidator()

        # Calculate comprehensive metrics
        power_metrics = analyzer.calculate_power_metrics(signal_data)
        bw_metrics = analyzer.calculate_bandwidth(signal_data)
        snr_metrics = analyzer.calculate_snr(signal_data)

        # Standards validation
        validation = validator.validate_signal(
            signal_data, sample_rate, standard, bandwidth_mhz
        )

        results = {
            'standard': standard.upper(),
            'power_metrics': power_metrics,
            'bandwidth_metrics': bw_metrics,
            'snr_metrics': snr_metrics,
            'standards_compliance': validation,
            'signal_length': len(signal_data),
            'sample_rate': sample_rate
        }

        # Check for issues
        issues = []
        if not validation['bandwidth_valid']:
            issues.append(f"Bandwidth mismatch: {bw_metrics['bandwidth']/1e6:.2f} MHz "
                         f"vs expected {validation['expected_bandwidth']/1e6:.2f} MHz")

        if not validation['papr_valid']:
            issues.append(f"PAPR out of range: {power_metrics['papr_db']:.2f} dB "
                         f"(expected {validation['expected_papr_range']})")

        if power_metrics['avg_power'] < 0.9 or power_metrics['avg_power'] > 1.1:
            issues.append(f"Power normalization issue: {power_metrics['avg_power']:.3f} "
                         "(expected ~1.0)")

        results['issues'] = issues
        results['quality_score'] = self._compute_quality_score(results)

        return results

    def _compute_quality_score(self, results):
        """Compute overall quality score (0-100)"""
        score = 100.0

        # Deduct points for issues
        if not results['standards_compliance']['bandwidth_valid']:
            score -= 20

        if not results['standards_compliance']['papr_valid']:
            score -= 15

        # Check power normalization (should be ~1.0)
        power = results['power_metrics']['avg_power']
        power_error = abs(power - 1.0)
        if power_error > 0.1:
            score -= min(20, power_error * 100)

        return max(0, score)

    def validate_constellation(self, symbols, modulation_type):
        """
        Validate modulation constellation quality

        Args:
            symbols: Modulated symbols (complex)
            modulation_type: 'QPSK', '16QAM', '64QAM', '256QAM'
        """
        print(f"\nValidating {modulation_type} constellation...")

        # Expected constellation points
        if modulation_type == 'QPSK':
            M = 4
        elif modulation_type == '16QAM':
            M = 16
        elif modulation_type == '64QAM':
            M = 64
        elif modulation_type == '256QAM':
            M = 256
        else:
            return {'error': f'Unknown modulation: {modulation_type}'}

        # Normalize symbols
        symbols_norm = symbols / np.sqrt(np.mean(np.abs(symbols)**2))

        # Calculate EVM (Error Vector Magnitude)
        # For ideal constellation, symbols should cluster around constellation points
        I = np.real(symbols_norm)
        Q = np.imag(symbols_norm)

        # Compute variance (lower is better)
        I_std = np.std(I)
        Q_std = np.std(Q)

        # Compute dynamic range
        I_range = np.max(I) - np.min(I)
        Q_range = np.max(Q) - np.min(Q)

        results = {
            'modulation': modulation_type,
            'num_symbols': len(symbols),
            'I_std': float(I_std),
            'Q_std': float(Q_std),
            'I_range': float(I_range),
            'Q_range': float(Q_range),
            'avg_magnitude': float(np.mean(np.abs(symbols_norm))),
            'power': float(np.mean(np.abs(symbols)**2))
        }

        # Quality check
        issues = []
        if I_std > 0.5:
            issues.append(f"High I-channel variance: {I_std:.3f}")
        if Q_std > 0.5:
            issues.append(f"High Q-channel variance: {Q_std:.3f}")

        results['issues'] = issues
        results['quality_ok'] = len(issues) == 0

        return results

    def validate_spectral_purity(self, signal_data, sample_rate):
        """
        Check for spectral artifacts and spurious emissions
        """
        print("\nValidating spectral purity...")

        # Compute power spectral density
        freqs, psd = signal.welch(signal_data, fs=sample_rate, nperseg=2048)
        psd_db = 10 * np.log10(psd + 1e-12)

        # Find main lobe (90% of power)
        cumulative_power = np.cumsum(psd) / np.sum(psd)
        main_lobe_start = np.where(cumulative_power >= 0.05)[0][0]
        main_lobe_end = np.where(cumulative_power >= 0.95)[0][0]

        main_lobe_bw = freqs[main_lobe_end] - freqs[main_lobe_start]

        # Check for spurious emissions outside main lobe
        main_lobe_power = np.mean(psd_db[main_lobe_start:main_lobe_end])
        outside_lobe_power = np.max(np.concatenate([
            psd_db[:main_lobe_start],
            psd_db[main_lobe_end:]
        ]))

        spurious_suppression = main_lobe_power - outside_lobe_power

        results = {
            'main_lobe_bandwidth': float(main_lobe_bw),
            'spurious_suppression_db': float(spurious_suppression),
            'peak_to_average_spectral': float(np.max(psd_db) - np.mean(psd_db))
        }

        # Quality check
        issues = []
        if spurious_suppression < 20:
            issues.append(f"Poor spurious suppression: {spurious_suppression:.1f} dB "
                         "(should be >20 dB)")

        results['issues'] = issues
        results['quality_ok'] = len(issues) == 0

        return results

    def validate_statistical_properties(self, signal_data):
        """
        Check if signal has expected statistical properties
        """
        print("\nValidating statistical properties...")

        # Real and imaginary parts should be approximately Gaussian for complex signals
        I = np.real(signal_data)
        Q = np.imag(signal_data)

        # Normality test (Kolmogorov-Smirnov test)
        _, p_value_I = stats.kstest(I, 'norm', args=(np.mean(I), np.std(I)))
        _, p_value_Q = stats.kstest(Q, 'norm', args=(np.mean(Q), np.std(Q)))

        # Compute kurtosis (should be ~3 for Gaussian)
        kurtosis_I = float(stats.kurtosis(I, fisher=False))
        kurtosis_Q = float(stats.kurtosis(Q, fisher=False))

        # Compute skewness (should be ~0 for Gaussian)
        skewness_I = float(stats.skew(I))
        skewness_Q = float(stats.skew(Q))

        results = {
            'I_mean': float(np.mean(I)),
            'Q_mean': float(np.mean(Q)),
            'I_std': float(np.std(I)),
            'Q_std': float(np.std(Q)),
            'I_kurtosis': kurtosis_I,
            'Q_kurtosis': kurtosis_Q,
            'I_skewness': skewness_I,
            'Q_skewness': skewness_Q,
            'I_normality_pvalue': float(p_value_I),
            'Q_normality_pvalue': float(p_value_Q)
        }

        # Quality check
        issues = []
        if abs(kurtosis_I - 3) > 2:
            issues.append(f"Abnormal I kurtosis: {kurtosis_I:.2f} (expected ~3)")
        if abs(kurtosis_Q - 3) > 2:
            issues.append(f"Abnormal Q kurtosis: {kurtosis_Q:.2f} (expected ~3)")

        results['issues'] = issues
        results['quality_ok'] = len(issues) == 0

        return results

    def generate_quality_report(self, validation_results, report_name='quality_report'):
        """Generate comprehensive quality report with visualizations"""
        print(f"\nGenerating quality report: {report_name}")

        # Save JSON report
        report_path = self.output_dir / f'{report_name}.json'
        with open(report_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = self._convert_to_json_serializable(validation_results)
            json.dump(json_results, f, indent=2)

        print(f"Report saved to: {report_path}")

        # Print summary
        print("\n" + "="*60)
        print("DATA QUALITY VALIDATION SUMMARY")
        print("="*60)

        for key, result in validation_results.items():
            if isinstance(result, dict) and 'quality_score' in result:
                score = result['quality_score']
                issues = result.get('issues', [])
                print(f"\n{key}:")
                print(f"  Quality Score: {score:.1f}/100")
                if issues:
                    print(f"  Issues Found:")
                    for issue in issues:
                        print(f"    - {issue}")
                else:
                    print(f"  Status: PASSED")

        print("\n" + "="*60)

        return report_path

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


def validate_reference_signals():
    """
    Generate and validate reference signals for each standard
    This ensures the signal generators are working correctly
    """
    print("="*60)
    print("REFERENCE SIGNAL VALIDATION")
    print("="*60)

    validator = DataQualityValidator()

    # Common parameters
    sample_rate = 30.72e6
    duration = 0.01  # 10ms

    all_results = {}

    # Validate GSM
    print("\n### GSM (2G) ###")
    gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
    gsm_signal = gsm_gen.generate_baseband()

    gsm_results = validator.validate_signal_standard(
        gsm_signal, sample_rate, 'gsm'
    )
    gsm_spectral = validator.validate_spectral_purity(gsm_signal, sample_rate)
    gsm_stats = validator.validate_statistical_properties(gsm_signal)

    all_results['GSM'] = {
        'signal': gsm_results,
        'spectral': gsm_spectral,
        'statistics': gsm_stats
    }

    # Validate UMTS
    print("\n### UMTS (3G) ###")
    umts_gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
    umts_signal = umts_gen.generate_baseband()

    umts_results = validator.validate_signal_standard(
        umts_signal, sample_rate, 'umts'
    )
    umts_spectral = validator.validate_spectral_purity(umts_signal, sample_rate)
    umts_stats = validator.validate_statistical_properties(umts_signal)

    all_results['UMTS'] = {
        'signal': umts_results,
        'spectral': umts_spectral,
        'statistics': umts_stats
    }

    # Validate LTE
    print("\n### LTE (4G) ###")
    lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration,
                          bandwidth=20, modulation='64QAM')
    lte_signal = lte_gen.generate_baseband()

    lte_results = validator.validate_signal_standard(
        lte_signal, sample_rate, 'lte', bandwidth_mhz=20
    )
    lte_spectral = validator.validate_spectral_purity(lte_signal, sample_rate)
    lte_stats = validator.validate_statistical_properties(lte_signal)

    all_results['LTE'] = {
        'signal': lte_results,
        'spectral': lte_spectral,
        'statistics': lte_stats
    }

    # Validate 5G NR
    print("\n### 5G NR ###")
    nr_gen = NRGenerator(sample_rate=61.44e6, duration=duration,
                        bandwidth=100, modulation='256QAM', numerology=1)
    nr_signal = nr_gen.generate_baseband()

    nr_results = validator.validate_signal_standard(
        nr_signal, 61.44e6, 'nr', bandwidth_mhz=100
    )
    nr_spectral = validator.validate_spectral_purity(nr_signal, 61.44e6)
    nr_stats = validator.validate_statistical_properties(nr_signal)

    all_results['5G_NR'] = {
        'signal': nr_results,
        'spectral': nr_spectral,
        'statistics': nr_stats
    }

    # Generate comprehensive report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = validator.generate_quality_report(
        all_results,
        report_name=f'reference_validation_{timestamp}'
    )

    print(f"\nDetailed report saved to: {report_path}")
    print("\nIf all quality scores are >80, the signal generators are working correctly.")
    print("You can proceed to generate the full dataset.")


if __name__ == "__main__":
    validate_reference_signals()
