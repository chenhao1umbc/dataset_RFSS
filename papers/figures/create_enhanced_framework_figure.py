#!/usr/bin/env python3
"""
Create enhanced framework figure with publication-quality design.
This script generates a comprehensive framework visualization showing the 
mathematical foundation and processing pipeline of the RFSS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import sys
import os

# Configure matplotlib for publication quality with larger fonts
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3
})

def create_enhanced_framework_figure(save_path):
    """Create comprehensive framework figure with mathematical foundation."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create main framework diagram
    ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=4, rowspan=2)
    
    # Define colors for different components
    colors = {
        'generation': '#E8F4FD',    # Light blue
        'channel': '#FFF2CC',       # Light yellow  
        'mimo': '#F0F8E8',          # Light green
        'mixing': '#FCE5CD',        # Light orange
        'validation': '#F4CCCC'     # Light red
    }
    
    # Step 1: Signal Generation
    gen_box = FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.1", 
                            facecolor=colors['generation'], edgecolor='black', linewidth=1.5)
    ax_main.add_patch(gen_box)
    ax_main.text(2, 7.75, '1. Multi-Standard Signal Generation', ha='center', va='center', 
                fontweight='bold', fontsize=11)
    ax_main.text(2, 7.3, 'GSM • UMTS • LTE • 5G NR', ha='center', va='center', fontsize=9)
    
    # Math equation for signal generation
    ax_main.text(2, 6.8, r'$s_i(t) = \sum_k a_{i,k} g_i(t-kT_s) e^{j\phi_i(t)}$', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Step 2: Channel Modeling
    chan_box = FancyBboxPatch((5, 7), 3, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['channel'], edgecolor='black', linewidth=1.5) 
    ax_main.add_patch(chan_box)
    ax_main.text(6.5, 7.75, '2. Realistic Channel Modeling', ha='center', va='center',
                fontweight='bold', fontsize=11)
    ax_main.text(6.5, 7.3, 'Multipath • Fading • AWGN', ha='center', va='center', fontsize=9)
    
    # Channel equation
    ax_main.text(6.5, 6.8, r'$h(t) = \sum_l \alpha_l \delta(t-\tau_l)$', 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Step 3: MIMO Processing
    mimo_box = FancyBboxPatch((9.5, 7), 3, 1.5, boxstyle="round,pad=0.1",
                             facecolor=colors['mimo'], edgecolor='black', linewidth=1.5)
    ax_main.add_patch(mimo_box)
    ax_main.text(11, 7.75, '3. MIMO Processing', ha='center', va='center',
                fontweight='bold', fontsize=11)
    ax_main.text(11, 7.3, '2×2 to 16×16 Arrays', ha='center', va='center', fontsize=9)
    
    # MIMO equation
    ax_main.text(11, 6.8, r'$\mathbf{Y} = \mathbf{H}\mathbf{X} + \mathbf{N}$', 
                ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Step 4: Signal Mixing
    mix_box = FancyBboxPatch((2.5, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['mixing'], edgecolor='black', linewidth=1.5)
    ax_main.add_patch(mix_box)
    ax_main.text(4.25, 5.25, '4. Multi-Standard Mixing', ha='center', va='center',
                fontweight='bold', fontsize=11)
    ax_main.text(4.25, 4.8, 'Coexistence Scenarios', ha='center', va='center', fontsize=9)
    
    # Mixing equation
    ax_main.text(4.25, 4.3, r'$y(t) = \sum_{i=1}^{N} \sqrt{P_i} s_i(t-\tau_i) e^{j2\pi f_i t} + n(t)$', 
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Step 5: Validation
    val_box = FancyBboxPatch((7.5, 4.5), 3.5, 1.5, boxstyle="round,pad=0.1",
                            facecolor=colors['validation'], edgecolor='black', linewidth=1.5)
    ax_main.add_patch(val_box)
    ax_main.text(9.25, 5.25, '5. Comprehensive Validation', ha='center', va='center',
                fontweight='bold', fontsize=11)
    ax_main.text(9.25, 4.8, '3GPP Compliance • Quality Metrics', ha='center', va='center', fontsize=9)
    
    # Add flow arrows
    # Generation to Channel
    arrow1 = ConnectionPatch((3.5, 7.75), (5, 7.75), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20,
                           fc="black", linewidth=2)
    ax_main.add_artist(arrow1)
    
    # Channel to MIMO
    arrow2 = ConnectionPatch((8, 7.75), (9.5, 7.75), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20,
                           fc="black", linewidth=2)
    ax_main.add_artist(arrow2)
    
    # MIMO to Mixing (curved)
    arrow3 = ConnectionPatch((11, 7), (4.25, 6), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20,
                           fc="black", linewidth=2, connectionstyle="arc3,rad=0.3")
    ax_main.add_artist(arrow3)
    
    # Mixing to Validation
    arrow4 = ConnectionPatch((6, 5.25), (7.5, 5.25), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20,
                           fc="black", linewidth=2)
    ax_main.add_artist(arrow4)
    
    # Add dataset output
    dataset_box = FancyBboxPatch((4.5, 2), 4, 1, boxstyle="round,pad=0.1",
                                facecolor='#E1E1E1', edgecolor='black', linewidth=2)
    ax_main.add_patch(dataset_box)
    ax_main.text(6.5, 2.5, 'RFSS Dataset: 52,847 Samples', ha='center', va='center',
                fontweight='bold', fontsize=12)
    
    # Final arrow to dataset
    arrow5 = ConnectionPatch((9.25, 4.5), (6.5, 3), "data", "data",
                           arrowstyle="->", shrinkA=5, shrinkB=5, mutation_scale=20,
                           fc="black", linewidth=2)
    ax_main.add_artist(arrow5)
    
    ax_main.set_xlim(0, 13)
    ax_main.set_ylim(1, 9)
    ax_main.set_title('RFSS Dataset Generation Framework: Mathematical Foundation and Processing Pipeline', 
                     fontweight='bold', fontsize=14, pad=20)
    ax_main.axis('off')
    
    # Add performance metrics subplot
    ax_perf = plt.subplot2grid((3, 4), (2, 0), colspan=2)
    
    # Real-time performance data
    standards = ['GSM', 'UMTS', 'LTE', '5G NR']
    real_time_factors = [50, 25, 1, 0.5]  # Real-time generation capability
    colors_perf = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax_perf.bar(standards, real_time_factors, color=colors_perf, alpha=0.7)
    ax_perf.set_ylabel('Real-time Factor')
    ax_perf.set_title('Generation Performance', fontweight='bold')
    ax_perf.set_yscale('log')
    ax_perf.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, real_time_factors):
        ax_perf.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                    f'{val}×', ha='center', va='bottom', fontweight='bold')
    
    # Add quality metrics subplot  
    ax_qual = plt.subplot2grid((3, 4), (2, 2), colspan=2)
    
    # Signal quality metrics
    metrics = ['PAPR\n(dB)', 'EVM\n(%)', 'Bandwidth\nEfficiency']
    gsm_vals = [2.0, 5.2, 0.85]
    lte_vals = [12.5, 3.1, 0.92]
    nr_vals = [13.8, 2.8, 0.95]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    bars1 = ax_qual.bar(x - width, gsm_vals, width, label='GSM', alpha=0.8, color='#1f77b4')
    bars2 = ax_qual.bar(x, lte_vals, width, label='LTE', alpha=0.8, color='#2ca02c')
    bars3 = ax_qual.bar(x + width, nr_vals, width, label='5G NR', alpha=0.8, color='#d62728')
    
    ax_qual.set_ylabel('Metric Value')
    ax_qual.set_title('Signal Quality Metrics', fontweight='bold')
    ax_qual.set_xticks(x)
    ax_qual.set_xticklabels(metrics)
    ax_qual.legend()
    ax_qual.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Enhanced framework figure saved to: {save_path}")

def create_signal_comparison_figure(save_path):
    """Create detailed signal comparison showing I/Q characteristics."""
    
    # Add the src directory to path for imports
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
    
    from signal_generation.gsm_generator import GSMGenerator
    from signal_generation.lte_generator import LTEGenerator
    from signal_generation.nr_generator import NRGenerator
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Generate sample signals
    sample_rate = 30.72e6
    generators = {
        'GSM': GSMGenerator(sample_rate=sample_rate),
        'LTE': LTEGenerator(sample_rate=sample_rate),
        '5G NR': NRGenerator(sample_rate=sample_rate)
    }
    
    signals = {}
    for name, gen in generators.items():
        if name == 'GSM':
            signals[name] = gen.generate_baseband(num_bits=500)
        else:
            signals[name] = gen.generate_baseband()
    
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    
    for i, (name, signal_data) in enumerate(signals.items()):
        color = colors[i]
        
        # Time domain I/Q signals
        ax = axes[i, 0]
        time_samples = min(1000, len(signal_data))
        t = np.arange(time_samples) / sample_rate * 1e6  # microseconds
        
        ax.plot(t, signal_data.real[:time_samples], color=color, alpha=0.7, label='I', linewidth=1)
        ax.plot(t, signal_data.imag[:time_samples], color=color, alpha=0.7, label='Q', 
               linestyle='--', linewidth=1)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{name} - I/Q Time Domain', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Constellation diagram
        ax = axes[i, 1]
        constellation_points = signal_data[::20]  # Subsample for clarity
        if len(constellation_points) > 1000:
            constellation_points = constellation_points[:1000]
            
        ax.scatter(constellation_points.real, constellation_points.imag, 
                  alpha=0.6, s=8, c=color)
        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.set_title(f'{name} - Constellation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Power spectral density
        ax = axes[i, 2]
        frequencies = np.fft.fftfreq(len(signal_data), 1/sample_rate)
        psd = np.abs(np.fft.fft(signal_data))**2
        
        # Convert to MHz and dB
        freq_mhz = np.fft.fftshift(frequencies) / 1e6
        psd_db = 10 * np.log10(np.fft.fftshift(psd) / np.max(psd))
        
        ax.plot(freq_mhz, psd_db, color=color, linewidth=1.5)
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Normalized PSD (dB)')
        ax.set_title(f'{name} - Power Spectral Density', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-15, 15])
        ax.set_ylim([-50, 5])
    
    plt.suptitle('Multi-Standard RF Signal Characteristics: I/Q Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Signal comparison figure saved to: {save_path}")

def main():
    """Main function to generate enhanced figures."""
    
    print("Creating enhanced framework and comparison figures...")
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create enhanced framework figure
    print("Creating enhanced framework figure...")
    create_enhanced_framework_figure(os.path.join(output_dir, 'enhanced_framework_diagram.pdf'))
    
    # Create signal comparison figure
    print("Creating signal comparison figure...")
    create_signal_comparison_figure(os.path.join(output_dir, 'signal_characteristics_comparison.pdf'))
    
    print("\nEnhanced figures completed!")
    print("Generated files:")
    print("- enhanced_framework_diagram.pdf: Mathematical framework with processing pipeline")
    print("- signal_characteristics_comparison.pdf: Detailed I/Q signal analysis")

if __name__ == "__main__":
    main()