#!/usr/bin/env python3
"""
Create comprehensive STFT analysis visualizations for RFSS dataset paper.
This script generates publication-quality spectrograms and time-frequency analysis
showing the realistic characteristics of multi-standard RF signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
import os

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator  
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator
from mixing.signal_mixer import SignalMixer

# Configure matplotlib for publication quality with larger fonts
plt.rcParams.update({
    'font.size': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.titlesize': 22,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3
})

def generate_signal_samples(duration_ms=10, sample_rate=30.72e6):
    """Generate realistic RF signal samples for each standard."""
    samples_per_signal = int(duration_ms * 1e-3 * sample_rate)
    
    # Initialize generators
    generators = {
        'GSM': GSMGenerator(sample_rate=sample_rate),
        'UMTS': UMTSGenerator(sample_rate=sample_rate),
        'LTE': LTEGenerator(sample_rate=sample_rate),
        '5G NR': NRGenerator(sample_rate=sample_rate)
    }
    
    # Generate signals
    signals = {}
    for name, gen in generators.items():
        print(f"Generating {name} signal...")
        if name == 'GSM':
            signal_data = gen.generate_baseband(num_bits=1000)
        else:
            signal_data = gen.generate_baseband()
        # Ensure consistent length
        if len(signal_data) > samples_per_signal:
            signals[name] = signal_data[:samples_per_signal]
        else:
            # Pad with zeros if too short
            padded = np.zeros(samples_per_signal, dtype=complex)
            padded[:len(signal_data)] = signal_data
            signals[name] = padded
    
    return signals, sample_rate

def create_stft_analysis(signals, sample_rate, save_path):
    """Create comprehensive STFT analysis showing time-frequency characteristics."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # STFT parameters for good time-frequency resolution
    nperseg = 1024  # Good frequency resolution
    noverlap = 512  # 50% overlap for smooth time resolution
    
    signal_names = list(signals.keys())
    n_signals = len(signal_names)
    
    # Create subplots: 2 rows, 2 columns for individual signals
    for i, (name, sig) in enumerate(signals.items()):
        ax = plt.subplot(2, 2, i+1)
        
        # Compute STFT
        f, t, Zxx = signal.stft(sig, fs=sample_rate, nperseg=nperseg, 
                               noverlap=noverlap, return_onesided=False)
        
        # Convert to MHz for better readability
        f_mhz = np.fft.fftshift(f) / 1e6
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        
        # Convert to dB scale
        magnitude_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)
        
        # Create spectrogram
        im = ax.pcolormesh(t * 1000, f_mhz, magnitude_db, 
                          shading='gouraud', cmap='viridis')
        
        ax.set_title(f'{name} Signal Spectrogram', fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (MHz)')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)
        
        # Set frequency limits for better visualization
        ax.set_ylim([-15, 15])  # ±15 MHz view
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"STFT analysis saved to: {save_path}")

def create_mixed_signal_analysis(signals, sample_rate, save_path):
    """Create analysis of mixed multi-standard signals."""
    
    # Create realistic mixed signal scenario
    mixer = SignalMixer(sample_rate)
    
    # Add signals with different carrier frequencies and power levels
    mixer.add_signal(signals['GSM'], carrier_freq=900e6, power_db=-20)
    mixer.add_signal(signals['LTE'], carrier_freq=1800e6, power_db=-15) 
    mixer.add_signal(signals['5G NR'], carrier_freq=3500e6, power_db=-18)
    
    # Generate mixed signal
    mixed_signal, _ = mixer.mix_signals()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # STFT parameters
    nperseg = 1024
    noverlap = 512
    
    # Individual signals analysis
    individual_signals = {
        'GSM (900 MHz)': signals['GSM'],
        'LTE (1800 MHz)': signals['LTE'], 
        '5G NR (3500 MHz)': signals['5G NR']
    }
    
    for i, (name, sig) in enumerate(individual_signals.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        f, t, Zxx = signal.stft(sig, fs=sample_rate, nperseg=nperseg,
                               noverlap=noverlap, return_onesided=False)
        
        f_mhz = np.fft.fftshift(f) / 1e6
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        magnitude_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)
        
        im = ax.pcolormesh(t * 1000, f_mhz, magnitude_db,
                          shading='gouraud', cmap='viridis')
        
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (MHz)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-15, 15])
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)
    
    # Mixed signal analysis
    ax = axes[1, 1]
    f, t, Zxx = signal.stft(mixed_signal, fs=sample_rate, nperseg=nperseg,
                           noverlap=noverlap, return_onesided=False)
    
    f_mhz = np.fft.fftshift(f) / 1e6
    Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
    magnitude_db = 20 * np.log10(np.abs(Zxx_shifted) + 1e-12)
    
    im = ax.pcolormesh(t * 1000, f_mhz, magnitude_db,
                      shading='gouraud', cmap='plasma')
    
    ax.set_title('Multi-Standard Mixed Signal', fontweight='bold', color='red')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-15, 15])
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)
    
    plt.suptitle('Multi-Standard RF Signal Source Separation Challenge', 
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Mixed signal analysis saved to: {save_path}")

def create_constellation_diagrams(signals, save_path):
    """Create constellation diagrams showing modulation characteristics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for i, (name, sig) in enumerate(signals.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Sample points for constellation (every 10th sample to avoid crowding)
        constellation_points = sig[::10]
        
        # Limit to reasonable number of points
        if len(constellation_points) > 2000:
            constellation_points = constellation_points[:2000]
        
        # Plot constellation
        ax.scatter(constellation_points.real, constellation_points.imag, 
                  alpha=0.6, s=1, c='blue')
        
        ax.set_title(f'{name} Constellation', fontweight='bold')
        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set reasonable axis limits based on signal amplitude
        max_val = np.max([np.abs(constellation_points.real), 
                         np.abs(constellation_points.imag)]) * 1.1
        ax.set_xlim(-max_val, max_val)
        ax.set_ylim(-max_val, max_val)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Constellation diagrams saved to: {save_path}")

def main():
    """Main function to generate all STFT visualizations."""
    
    print("Generating comprehensive STFT analysis for RFSS dataset...")
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate signal samples
    print("Generating signal samples...")
    signals, sample_rate = generate_signal_samples()
    
    # Create STFT analysis
    print("Creating STFT spectrograms...")
    create_stft_analysis(signals, sample_rate, 
                        os.path.join(output_dir, 'stft_spectrograms.pdf'))
    
    # Create mixed signal analysis
    print("Creating mixed signal analysis...")
    create_mixed_signal_analysis(signals, sample_rate,
                                os.path.join(output_dir, 'mixed_signal_analysis.pdf'))
    
    # Create constellation diagrams
    print("Creating constellation diagrams...")
    create_constellation_diagrams(signals, 
                                 os.path.join(output_dir, 'constellation_diagrams.pdf'))
    
    print("\nAll STFT visualizations completed successfully!")
    print("Generated files:")
    print("- stft_spectrograms.pdf: Individual signal STFT analysis")
    print("- mixed_signal_analysis.pdf: Multi-standard coexistence scenarios")
    print("- constellation_diagrams.pdf: I/Q constellation characteristics")

if __name__ == "__main__":
    main()