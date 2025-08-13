#!/usr/bin/env python3
"""
Create figure showing actual 2G-5G signal samples and their mixture
"""
import sys
sys.path.append('/Users/hc/Documents/research/Projects/dataset_RFSS')

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
from pathlib import Path

# Import actual generators
from src.signal_generation.gsm_generator import GSMGenerator
from src.signal_generation.lte_generator import LTEGenerator
from src.signal_generation.nr_generator import NRGenerator
from src.signal_generation.umts_generator import UMTSGenerator

# Set up matplotlib for high-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def generate_actual_signals():
    """Generate actual signals from our generators"""
    print("Generating actual signals from codebase...")
    
    # Common parameters
    duration = 0.005  # 5ms for good visualization
    
    # Generate GSM signal
    print("  Generating GSM...")
    gsm_gen = GSMGenerator(sample_rate=1e6, duration=duration)
    gsm_signal = gsm_gen.generate_baseband()
    gsm_time = np.arange(len(gsm_signal)) / 1e6
    
    # Generate UMTS signal  
    print("  Generating UMTS...")
    umts_gen = UMTSGenerator(sample_rate=15.36e6, duration=duration)
    umts_signal = umts_gen.generate_baseband()
    umts_time = np.arange(len(umts_signal)) / 15.36e6
    
    # Generate LTE signal
    print("  Generating LTE...")
    lte_gen = LTEGenerator(sample_rate=30.72e6, duration=duration)
    lte_signal = lte_gen.generate_baseband()
    lte_time = np.arange(len(lte_signal)) / 30.72e6
    
    # Generate 5G NR signal
    print("  Generating 5G NR...")
    nr_gen = NRGenerator(sample_rate=61.44e6, duration=duration)
    nr_signal = nr_gen.generate_baseband()
    nr_time = np.arange(len(nr_signal)) / 61.44e6
    
    return {
        'GSM': {'signal': gsm_signal, 'time': gsm_time, 'fs': 1e6},
        'UMTS': {'signal': umts_signal, 'time': umts_time, 'fs': 15.36e6},
        'LTE': {'signal': lte_signal, 'time': lte_time, 'fs': 30.72e6},
        '5G NR': {'signal': nr_signal, 'time': nr_time, 'fs': 61.44e6}
    }

def create_signal_samples_figure():
    """Create comprehensive signal samples figure"""
    signals = generate_actual_signals()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3x2 subplot layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.3)
    
    # Individual signal time domain plots (top 4 subplots)
    signal_names = ['GSM', 'UMTS', 'LTE', '5G NR']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (name, color) in enumerate(zip(signal_names, colors)):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        sig_data = signals[name]
        time = sig_data['time'] * 1000  # Convert to ms
        signal_iq = sig_data['signal']
        
        # Plot I and Q components
        ax.plot(time, np.real(signal_iq), color=color, linewidth=1.5, alpha=0.8, label='I')
        ax.plot(time, np.imag(signal_iq), color=color, linewidth=1.5, alpha=0.6, linestyle='--', label='Q')
        
        ax.set_title(f'{name} Signal (Actual Generated)', fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xlim(0, 5)
        
        # Add technical specs as text
        if name == 'GSM':
            specs = f'BW: 200 kHz, GMSK\nSymbol Rate: 271 ksps'
        elif name == 'UMTS':
            specs = f'BW: 5 MHz, W-CDMA\nChip Rate: 3.84 Mcps'
        elif name == 'LTE':
            specs = f'BW: 20 MHz, OFDM\nSubcarrier: 15 kHz'
        else:  # 5G NR
            specs = f'BW: 100 MHz, OFDM\nFlexible Numerology'
            
        ax.text(0.02, 0.98, specs, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='white', alpha=0.8))
    
    # Mixed signal plot (bottom spanning both columns)
    ax_mix = fig.add_subplot(gs[2, :])
    
    # Create mixed signal with proper resampling
    print("Creating mixed signal...")
    target_fs = 61.44e6  # Highest sample rate for proper mixing
    target_samples = int(target_fs * 0.005)  # 5ms duration
    
    mixed_signal = np.zeros(target_samples, dtype=complex)
    
    # Add each signal with frequency offset and power scaling
    freq_offsets = [-15e6, -5e6, 5e6, 20e6]  # MHz offsets for visualization
    powers = [0.8, 0.9, 1.0, 0.7]  # Relative power levels
    
    for i, (name, offset, power) in enumerate(zip(signal_names, freq_offsets, powers)):
        sig_data = signals[name]
        original_signal = sig_data['signal']
        
        # Resample to target rate if needed
        if len(original_signal) != target_samples:
            # Simple resampling
            indices = np.linspace(0, len(original_signal)-1, target_samples)
            resampled = np.interp(indices, np.arange(len(original_signal)), 
                                np.real(original_signal)) + 1j * np.interp(indices, 
                                np.arange(len(original_signal)), np.imag(original_signal))
        else:
            resampled = original_signal
        
        # Apply frequency offset and power scaling
        t = np.arange(target_samples) / target_fs
        carrier = np.exp(1j * 2 * np.pi * offset * t)
        mixed_signal += power * resampled * carrier
    
    # Add noise
    noise_power = 0.1 * np.var(mixed_signal)
    mixed_signal += np.sqrt(noise_power/2) * (np.random.randn(target_samples) + 
                                             1j * np.random.randn(target_samples))
    
    # Plot mixed signal
    time_mix = np.arange(target_samples) / target_fs * 1000  # ms
    ax_mix.plot(time_mix, np.real(mixed_signal), 'k-', linewidth=1, alpha=0.8, label='Mixed I')
    ax_mix.plot(time_mix, np.imag(mixed_signal), 'k--', linewidth=1, alpha=0.6, label='Mixed Q')
    
    ax_mix.set_title('Mixed Multi-Standard Signal (2G+3G+4G+5G)', fontweight='bold', fontsize=14)
    ax_mix.set_xlabel('Time (ms)')
    ax_mix.set_ylabel('Amplitude')
    ax_mix.grid(True, alpha=0.3)
    ax_mix.legend(loc='upper right')
    ax_mix.set_xlim(0, 5)
    
    # Add mixture equation
    equation_text = r'$y(t) = \sum_{i=1}^{4} \sqrt{P_i} s_i(t) e^{j2\pi f_i t} + n(t)$'
    ax_mix.text(0.5, 0.95, equation_text, transform=ax_mix.transAxes, 
                fontsize=12, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    # Add power and frequency info
    power_text = "Power: GSM(0.8), UMTS(0.9), LTE(1.0), 5G(0.7)\nFreq Offset: -15, -5, +5, +20 MHz"
    ax_mix.text(0.02, 0.05, power_text, transform=ax_mix.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', 
                facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('RFSS Dataset: Actual Multi-Standard RF Signal Samples', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_dir = Path(".")
    plt.savefig(output_dir / 'signal_samples_actual.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'signal_samples_actual.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Signal samples figure saved to: {output_dir.absolute()}")
    plt.close()

if __name__ == "__main__":
    create_signal_samples_figure()
    print("✓ Created actual signal samples figure")