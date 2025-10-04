#!/usr/bin/env python3
"""
Data Inspection and Visualization Tool

Visualize RF signals to understand data quality:
- Time domain plots (I/Q)
- Frequency domain (PSD)
- Spectrograms
- Constellation diagrams
- Statistical distributions
"""
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator


def plot_signal_analysis(signal_data, sample_rate, title='Signal Analysis',
                         save_path=None):
    """
    Create comprehensive visualization of RF signal
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. Time domain - I/Q components
    ax1 = plt.subplot(3, 3, 1)
    time_ms = np.arange(len(signal_data)) / sample_rate * 1000
    ax1.plot(time_ms, np.real(signal_data), label='I', alpha=0.7, linewidth=0.5)
    ax1.plot(time_ms, np.imag(signal_data), label='Q', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Time Domain - I/Q Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Time domain - Magnitude
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time_ms, np.abs(signal_data), linewidth=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Time Domain - Magnitude')
    ax2.grid(True, alpha=0.3)

    # 3. Time domain - Phase
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time_ms, np.angle(signal_data), linewidth=0.5)
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Phase (radians)')
    ax3.set_title('Time Domain - Phase')
    ax3.grid(True, alpha=0.3)

    # 4. Power Spectral Density
    ax4 = plt.subplot(3, 3, 4)
    freqs, psd = signal.welch(signal_data, fs=sample_rate, nperseg=2048)
    ax4.plot(freqs/1e6, 10*np.log10(psd + 1e-12))
    ax4.set_xlabel('Frequency (MHz)')
    ax4.set_ylabel('PSD (dB/Hz)')
    ax4.set_title('Power Spectral Density')
    ax4.grid(True, alpha=0.3)

    # 5. Spectrogram
    ax5 = plt.subplot(3, 3, 5)
    f, t, Sxx = signal.spectrogram(signal_data, fs=sample_rate, nperseg=256)
    ax5.pcolormesh(t*1000, f/1e6, 10*np.log10(Sxx + 1e-12), shading='gouraud',
                   cmap='viridis')
    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Frequency (MHz)')
    ax5.set_title('Spectrogram')
    plt.colorbar(ax5.pcolormesh(t*1000, f/1e6, 10*np.log10(Sxx + 1e-12),
                                shading='gouraud', cmap='viridis'),
                ax=ax5, label='dB')

    # 6. Constellation diagram
    ax6 = plt.subplot(3, 3, 6)
    # Downsample for clarity
    step = max(1, len(signal_data) // 5000)
    signal_norm = signal_data / np.sqrt(np.mean(np.abs(signal_data)**2))
    ax6.scatter(np.real(signal_norm[::step]), np.imag(signal_norm[::step]),
               s=1, alpha=0.3)
    ax6.set_xlabel('In-phase (I)')
    ax6.set_ylabel('Quadrature (Q)')
    ax6.set_title('Constellation Diagram')
    ax6.grid(True, alpha=0.3)
    ax6.axis('equal')

    # 7. I/Q histogram
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(np.real(signal_data), bins=100, alpha=0.7, label='I', density=True)
    ax7.hist(np.imag(signal_data), bins=100, alpha=0.7, label='Q', density=True)
    ax7.set_xlabel('Amplitude')
    ax7.set_ylabel('Probability Density')
    ax7.set_title('I/Q Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Power distribution
    ax8 = plt.subplot(3, 3, 8)
    power_samples = np.abs(signal_data)**2
    ax8.hist(10*np.log10(power_samples + 1e-12), bins=100, density=True)
    ax8.set_xlabel('Power (dB)')
    ax8.set_ylabel('Probability Density')
    ax8.set_title('Power Distribution')
    ax8.grid(True, alpha=0.3)

    # 9. Signal statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    stats_text = [
        'Signal Statistics',
        '=' * 40,
        f'Length: {len(signal_data)} samples',
        f'Duration: {len(signal_data)/sample_rate*1000:.2f} ms',
        f'Sample Rate: {sample_rate/1e6:.2f} MHz',
        '',
        f'Mean Power: {np.mean(np.abs(signal_data)**2):.6f}',
        f'Peak Power: {np.max(np.abs(signal_data)**2):.6f}',
        f'PAPR: {10*np.log10(np.max(np.abs(signal_data)**2)/np.mean(np.abs(signal_data)**2)):.2f} dB',
        '',
        f'I Mean: {np.mean(np.real(signal_data)):.6f}',
        f'Q Mean: {np.mean(np.imag(signal_data)):.6f}',
        f'I Std: {np.std(np.real(signal_data)):.6f}',
        f'Q Std: {np.std(np.imag(signal_data)):.6f}',
    ]

    ax9.text(0.1, 0.9, '\n'.join(stats_text), transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def inspect_standard(standard, output_dir='data/quality_reports'):
    """
    Generate and inspect a signal from a specific standard
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 30.72e6
    duration = 0.01  # 10ms

    print(f"Generating {standard.upper()} signal for inspection...")

    if standard.lower() == 'gsm':
        gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        sig = gen.generate_baseband()
    elif standard.lower() == 'umts':
        gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
        sig = gen.generate_baseband()
    elif standard.lower() == 'lte':
        gen = LTEGenerator(sample_rate=sample_rate, duration=duration,
                          bandwidth=20, modulation='64QAM')
        sig = gen.generate_baseband()
    elif standard.lower() == 'nr' or standard.lower() == '5g':
        sample_rate = 61.44e6
        gen = NRGenerator(sample_rate=sample_rate, duration=duration,
                         bandwidth=100, modulation='256QAM', numerology=1)
        sig = gen.generate_baseband()
    else:
        print(f"Unknown standard: {standard}")
        return

    # Plot analysis
    save_path = output_dir / f'{standard.lower()}_inspection.png'
    plot_signal_analysis(sig, sample_rate, title=f'{standard.upper()} Signal Analysis',
                        save_path=save_path)


def compare_standards(output_dir='data/quality_reports'):
    """
    Generate comparison visualization of all standards
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating comparison of all standards...")

    sample_rate = 30.72e6
    duration = 0.01

    # Generate all signals
    signals = {}

    gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
    signals['GSM'] = gsm_gen.generate_baseband()

    umts_gen = UMTSGenerator(sample_rate=sample_rate, duration=duration)
    signals['UMTS'] = umts_gen.generate_baseband()

    lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration,
                          bandwidth=20, modulation='64QAM')
    signals['LTE'] = lte_gen.generate_baseband()

    nr_gen = NRGenerator(sample_rate=61.44e6, duration=duration,
                        bandwidth=100, modulation='256QAM', numerology=1)
    signals['5G NR'] = nr_gen.generate_baseband()

    # Create comparison plot
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))

    for idx, (name, sig) in enumerate(signals.items()):
        sr = 61.44e6 if name == '5G NR' else sample_rate

        # Time domain
        ax = axes[idx, 0]
        time_ms = np.arange(len(sig)) / sr * 1000
        ax.plot(time_ms[:1000], np.real(sig[:1000]), label='I', alpha=0.7)
        ax.plot(time_ms[:1000], np.imag(sig[:1000]), label='Q', alpha=0.7)
        ax.set_title(f'{name} - Time Domain')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PSD
        ax = axes[idx, 1]
        freqs, psd = signal.welch(sig, fs=sr, nperseg=2048)
        ax.plot(freqs/1e6, 10*np.log10(psd + 1e-12))
        ax.set_title(f'{name} - PSD')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('PSD (dB/Hz)')
        ax.grid(True, alpha=0.3)

        # Constellation
        ax = axes[idx, 2]
        step = max(1, len(sig) // 2000)
        sig_norm = sig / np.sqrt(np.mean(np.abs(sig)**2))
        ax.scatter(np.real(sig_norm[::step]), np.imag(sig_norm[::step]),
                  s=0.5, alpha=0.3)
        ax.set_title(f'{name} - Constellation')
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

    plt.suptitle('Multi-Standard Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = output_dir / 'standards_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect RF signal data quality')
    parser.add_argument('--standard', type=str, choices=['gsm', 'umts', 'lte', 'nr', 'all'],
                       default='all', help='Standard to inspect')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plot of all standards')
    parser.add_argument('--output', type=str, default='data/quality_reports',
                       help='Output directory for plots')

    args = parser.parse_args()

    if args.compare:
        compare_standards(args.output)
    elif args.standard == 'all':
        for std in ['gsm', 'umts', 'lte', 'nr']:
            inspect_standard(std, args.output)
        compare_standards(args.output)
    else:
        inspect_standard(args.standard, args.output)

    print("\nInspection complete! Check the output directory for visualizations.")
