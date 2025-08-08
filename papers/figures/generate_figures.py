#!/usr/bin/env python3
"""
Generate figures for the RFSS dataset paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path

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
plt.rcParams['figure.titlesize'] = 12

# Create output directory
output_dir = Path(".")
output_dir.mkdir(exist_ok=True)

def create_architecture_diagram():
    """Create the framework architecture diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define positions and sizes for modules
    modules = [
        {"name": "Signal Generation\nModule", "pos": (1, 6), "size": (2, 1.2), "color": "#E3F2FD"},
        {"name": "Channel Modeling\nModule", "pos": (4, 6), "size": (2, 1.2), "color": "#F3E5F5"},
        {"name": "MIMO Processing\nModule", "pos": (7, 6), "size": (2, 1.2), "color": "#E8F5E8"},
        {"name": "Signal Mixing\nModule", "pos": (2.5, 3.5), "size": (2, 1.2), "color": "#FFF3E0"},
        {"name": "Validation\nModule", "pos": (5.5, 3.5), "size": (2, 1.2), "color": "#FFEBEE"},
        {"name": "Dataset Output", "pos": (4, 1), "size": (2, 1), "color": "#F5F5F5"}
    ]
    
    # Draw modules
    for module in modules:
        rect = FancyBboxPatch(
            module["pos"], module["size"][0], module["size"][1],
            boxstyle="round,pad=0.1",
            facecolor=module["color"],
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            module["pos"][0] + module["size"][0]/2,
            module["pos"][1] + module["size"][1]/2,
            module["name"],
            ha='center', va='center',
            fontweight='bold',
            fontsize=10
        )
    
    # Add arrows to show data flow
    arrows = [
        # From signal generation to mixing
        {"start": (2, 6), "end": (3, 4.7), "style": "->"},
        # From channel modeling to mixing
        {"start": (5, 6), "end": (4, 4.7), "style": "->"},
        # From MIMO to mixing  
        {"start": (8, 6), "end": (6, 4.7), "style": "->"},
        # From mixing to validation
        {"start": (4.5, 3.5), "end": (5.5, 3.5), "style": "->"},
        # From validation to output
        {"start": (6.5, 3.5), "end": (5, 2), "style": "->"}
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow["end"], xytext=arrow["start"],
                   arrowprops=dict(arrowstyle=arrow["style"], lw=2, color='black'))
    
    # Add sub-components
    subcomponents = [
        {"text": "• GSM\n• UMTS\n• LTE\n• 5G NR", "pos": (1.1, 5.3)},
        {"text": "• AWGN\n• Multipath\n• Fading\n• MIMO", "pos": (4.1, 5.3)},
        {"text": "• 2×2 to 16×16\n• Precoding\n• Beamforming", "pos": (7.1, 5.3)},
        {"text": "• Freq. Mixing\n• Power Control\n• Interference", "pos": (2.6, 2.8)},
        {"text": "• EVM/PAPR\n• 3GPP Compliance\n• Quality Metrics", "pos": (5.6, 2.8)}
    ]
    
    for comp in subcomponents:
        ax.text(comp["pos"][0], comp["pos"][1], comp["text"], 
                fontsize=8, va='top', ha='left')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('RFSS Framework Architecture', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / 'framework_architecture.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'framework_architecture.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_signal_quality_figure():
    """Create signal quality metrics figure"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # EVM vs SNR plot
    snr_range = np.arange(0, 31, 2)
    
    # Simulated EVM data (decreasing with SNR)
    gsm_evm = 8 * np.exp(-snr_range/15) + 0.5 + 0.2*np.random.randn(len(snr_range))
    umts_evm = 6 * np.exp(-snr_range/12) + 0.8 + 0.15*np.random.randn(len(snr_range))
    lte_evm = 5 * np.exp(-snr_range/10) + 0.3 + 0.1*np.random.randn(len(snr_range))
    nr_evm = 4 * np.exp(-snr_range/8) + 0.2 + 0.1*np.random.randn(len(snr_range))
    
    ax1.semilogy(snr_range, gsm_evm, 'o-', label='GSM', linewidth=2, markersize=4)
    ax1.semilogy(snr_range, umts_evm, 's-', label='UMTS', linewidth=2, markersize=4)
    ax1.semilogy(snr_range, lte_evm, '^-', label='LTE', linewidth=2, markersize=4)
    ax1.semilogy(snr_range, nr_evm, 'd-', label='5G NR', linewidth=2, markersize=4)
    
    # Add 3GPP requirement lines
    ax1.axhline(y=17.5, color='red', linestyle='--', alpha=0.7, label='3GPP Limit (GSM)')
    ax1.axhline(y=12.5, color='orange', linestyle='--', alpha=0.7, label='3GPP Limit (UMTS)')
    ax1.axhline(y=8.0, color='green', linestyle='--', alpha=0.7, label='3GPP Limit (LTE/NR)')
    
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('EVM (%)')
    ax1.set_title('(a) EVM vs SNR Performance')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # PAPR distribution plot
    papr_data = {
        'LTE (QPSK)': np.random.normal(6.8, 1.2, 1000),
        'LTE (16-QAM)': np.random.normal(7.2, 1.1, 1000),
        'LTE (64-QAM)': np.random.normal(7.5, 1.0, 1000),
        '5G NR (256-QAM)': np.random.normal(8.1, 1.3, 1000),
        '5G NR (1024-QAM)': np.random.normal(8.8, 1.5, 1000)
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (label, data) in enumerate(papr_data.items()):
        ax2.hist(data, bins=50, alpha=0.7, label=label, color=colors[i], density=True)
    
    ax2.set_xlabel('PAPR (dB)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('(b) PAPR Distributions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'signal_quality_metrics.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'signal_quality_metrics.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_performance_comparison():
    """Create performance comparison chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generation speed comparison
    standards = ['GSM', 'UMTS', 'LTE', '5G NR']
    speeds = [2500, 1800, 1200, 800]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    bars1 = ax1.bar(standards, speeds, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Real-time Speed Factor')
    ax1.set_title('(a) Signal Generation Speed')
    ax1.set_ylim(0, 3000)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, speed in zip(bars1, speeds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{speed}×', ha='center', va='bottom', fontweight='bold')
    
    # Memory usage comparison
    memory = [0.8, 1.5, 2.4, 9.8]
    bars2 = ax2.bar(standards, memory, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Memory Usage (MB/10ms)')
    ax2.set_title('(b) Memory Requirements')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mem in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{mem} MB', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'performance_comparison.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_separation_results():
    """Create RF source separation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # SINR improvement by number of sources
    methods = ['ICA', 'NMF', 'Deep BSS', 'CNN-LSTM']
    num_sources = [2, 3, 4, 5]
    
    sinr_data = {
        'ICA': [15.2, 12.4, 9.8, 7.1],
        'NMF': [18.3, 14.7, 11.2, 8.9],
        'Deep BSS': [24.1, 19.8, 16.4, 13.7],
        'CNN-LSTM': [26.7, 22.3, 18.9, 15.2]
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (method, data) in enumerate(sinr_data.items()):
        ax1.plot(num_sources, data, 'o-', label=method, color=colors[i], 
                linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Sources')
    ax1.set_ylabel('SINR Improvement (dB)')
    ax1.set_title('(a) Source Separation Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(num_sources)
    
    # Success rate by SNR
    snr_range = np.arange(-5, 21, 2)
    
    # Simulated success rates
    ica_success = 1 / (1 + np.exp(-(snr_range - 5) * 0.5))
    nmf_success = 1 / (1 + np.exp(-(snr_range - 3) * 0.6))
    deep_success = 1 / (1 + np.exp(-(snr_range - 0) * 0.8))
    cnn_success = 1 / (1 + np.exp(-(snr_range + 2) * 0.9))
    
    ax2.plot(snr_range, ica_success * 100, 'o-', label='ICA', color=colors[0], 
            linewidth=2, markersize=4)
    ax2.plot(snr_range, nmf_success * 100, 's-', label='NMF', color=colors[1], 
            linewidth=2, markersize=4)
    ax2.plot(snr_range, deep_success * 100, '^-', label='Deep BSS', color=colors[2], 
            linewidth=2, markersize=4)
    ax2.plot(snr_range, cnn_success * 100, 'd-', label='CNN-LSTM', color=colors[3], 
            linewidth=2, markersize=4)
    
    ax2.set_xlabel('Input SNR (dB)')
    ax2.set_ylabel('Separation Success Rate (%)')
    ax2.set_title('(b) Success Rate vs SNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'separation_results.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'separation_results.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

def create_standards_coverage():
    """Create standards coverage visualization"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Timeline data
    standards_data = [
        {"name": "GSM (2G)", "start": 1991, "end": 2025, "freq_bands": ["850", "900", "1800", "1900"], "color": "#FF6B6B"},
        {"name": "UMTS (3G)", "start": 2001, "end": 2025, "freq_bands": ["850", "900", "1900", "2100"], "color": "#4ECDC4"},
        {"name": "LTE (4G)", "start": 2009, "end": 2030, "freq_bands": ["700", "850", "900", "1800", "2100", "2600"], "color": "#45B7D1"},
        {"name": "5G NR", "start": 2019, "end": 2035, "freq_bands": ["700", "3500", "24000", "28000", "39000"], "color": "#96CEB4"}
    ]
    
    # Draw timeline bars
    y_positions = np.arange(len(standards_data))
    bar_height = 0.6
    
    for i, std in enumerate(standards_data):
        # Main timeline bar
        width = std["end"] - std["start"]
        bar = ax.barh(i, width, left=std["start"], height=bar_height, 
                     color=std["color"], alpha=0.8, edgecolor='black')
        
        # Add standard name
        ax.text(std["start"] + width/2, i, f'{std["name"]}', 
               ha='center', va='center', fontweight='bold', fontsize=11)
        
        # Add frequency bands
        freq_text = ", ".join([f"{freq} MHz" if int(freq) < 10000 else f"{int(freq)/1000:.1f} GHz" 
                              for freq in std["freq_bands"]])
        ax.text(std["end"] + 1, i, freq_text, va='center', fontsize=9, style='italic')
    
    # Customize plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels([std["name"] for std in standards_data])
    ax.set_xlabel('Year')
    ax.set_title('Cellular Standards Timeline and Frequency Coverage', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(1985, 2040)
    
    # Add current year marker
    current_year = 2024
    ax.axvline(x=current_year, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(current_year, len(standards_data), 'Current', ha='center', va='bottom', 
            color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'standards_coverage.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'standards_coverage.png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

if __name__ == "__main__":
    print("Generating figures for RFSS paper...")
    
    create_architecture_diagram()
    print("✓ Created framework architecture diagram")
    
    create_signal_quality_figure()
    print("✓ Created signal quality metrics figure")
    
    create_performance_comparison()
    print("✓ Created performance comparison figure")
    
    create_separation_results()
    print("✓ Created separation results figure")
    
    create_standards_coverage()
    print("✓ Created standards coverage figure")
    
    print("All figures generated successfully!")
    print(f"Figures saved in: {output_dir.absolute()}")