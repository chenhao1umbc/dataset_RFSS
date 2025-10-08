"""
Demonstration of RF signal dataset generation
Shows complete workflow from signal generation to mixed scenarios
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator
from channel_models.basic_channels import ChannelSimulator, AWGNChannel, RayleighChannel
from mixing.signal_mixer import SignalMixer, InterferenceGenerator


def generate_single_standard_signals():
    """Generate signals from individual standards"""
    print("=== Generating Individual Standard Signals ===")
    
    # Common parameters
    sample_rate = 10e6  # 10 MHz
    duration = 0.01     # 10 ms
    
    results = {}
    
    # Generate GSM signal
    print("Generating GSM signal...")
    gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
    gsm_signal = gsm_gen.generate_baseband()
    gsm_freqs = gsm_gen.get_carrier_frequencies()
    
    results['GSM'] = {
        'signal': gsm_signal,
        'carrier_freqs': gsm_freqs,
        'bandwidth': gsm_gen.bandwidth,
        'power': np.mean(np.abs(gsm_signal)**2)
    }
    
    print(f"  GSM signal length: {len(gsm_signal)}")
    print(f"  GSM signal power: {results['GSM']['power']:.6f}")
    print(f"  GSM carrier frequencies: {gsm_freqs}")
    
    # Generate LTE signal
    print("\nGenerating LTE signal...")
    lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, 
                          bandwidth=20, modulation='64QAM')
    lte_signal = lte_gen.generate_baseband()
    lte_freqs = lte_gen.get_carrier_frequencies()
    
    results['LTE'] = {
        'signal': lte_signal,
        'carrier_freqs': lte_freqs,
        'bandwidth': lte_gen.bandwidth_mhz * 1e6,
        'power': np.mean(np.abs(lte_signal)**2)
    }
    
    print(f"  LTE signal length: {len(lte_signal)}")
    print(f"  LTE signal power: {results['LTE']['power']:.6f}")  
    print(f"  LTE carrier frequencies: {lte_freqs}")
    print(f"  LTE modulation: {lte_gen.modulation}")
    print(f"  LTE bandwidth: {lte_gen.bandwidth_mhz} MHz")
    
    return results


def apply_channel_effects(signals, sample_rate):
    """Apply various channel effects to signals"""
    print("\n=== Applying Channel Effects ===")
    
    results = {}
    
    for std_name, sig_data in signals.items():
        print(f"\nApplying channels to {std_name} signal...")
        
        signal = sig_data['signal']
        results[std_name] = {}
        
        # AWGN only
        awgn_sim = ChannelSimulator(sample_rate)
        awgn_sim.add_awgn(snr_db=15)
        awgn_signal = awgn_sim.apply(signal)
        results[std_name]['AWGN'] = awgn_signal
        
        # Rayleigh fading + AWGN
        rayleigh_sim = ChannelSimulator(sample_rate)
        rayleigh_sim.add_rayleigh_fading(doppler_freq=50).add_awgn(snr_db=10)
        rayleigh_signal = rayleigh_sim.apply(signal)
        results[std_name]['Rayleigh+AWGN'] = rayleigh_signal
        
        # Multipath + Rayleigh + AWGN
        multipath_sim = ChannelSimulator(sample_rate)
        multipath_sim.add_multipath().add_rayleigh_fading(doppler_freq=100).add_awgn(snr_db=8)
        multipath_signal = multipath_sim.apply(signal)
        results[std_name]['Multipath+Rayleigh+AWGN'] = multipath_signal
        
        print(f"  Original power: {np.mean(np.abs(signal)**2):.6f}")
        print(f"  AWGN power: {np.mean(np.abs(awgn_signal)**2):.6f}")
        print(f"  Rayleigh+AWGN power: {np.mean(np.abs(rayleigh_signal)**2):.6f}")
        print(f"  Multipath+Rayleigh+AWGN power: {np.mean(np.abs(multipath_signal)**2):.6f}")
    
    return results


def create_mixed_scenarios(signals, sample_rate):
    """Create mixed signal scenarios"""
    print("\n=== Creating Mixed Signal Scenarios ===")
    
    duration = 0.01  # 10 ms
    results = {}
    
    # Scenario 1: GSM + LTE (different frequencies)
    print("\nScenario 1: GSM + LTE Co-existence")
    mixer1 = SignalMixer(sample_rate)
    
    # Add GSM signal at 900 MHz
    gsm_signal = signals['GSM']['signal']
    mixer1.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
    
    # Add LTE signal at 2.1 GHz  
    lte_signal = signals['LTE']['signal']
    mixer1.add_signal(lte_signal, carrier_freq=2.1e9, power_db=-3, label='LTE')
    
    mixed_signal1, mixing_info1 = mixer1.mix_signals(duration=duration)
    results['GSM_LTE_Coexist'] = {
        'signal': mixed_signal1,
        'info': mixing_info1
    }
    
    print(f"  Mixed signal power: {np.mean(np.abs(mixed_signal1)**2):.6f}")
    print(f"  Number of components: {mixing_info1['num_signals']}")
    
    # Scenario 2: Adjacent channel interference
    print("\nScenario 2: Adjacent Channel Interference")
    mixer2 = SignalMixer(sample_rate)
    
    # Primary LTE signal
    mixer2.add_signal(lte_signal, carrier_freq=2.1e9, power_db=0, label='LTE_Primary')
    
    # Adjacent channel interferer
    interference = InterferenceGenerator.generate_narrowband_noise(
        sample_rate, duration, center_freq=0, bandwidth=5e6, power_db=-15
    )
    mixer2.add_signal(interference, carrier_freq=2.105e9, power_db=-15, 
                     label='Adjacent_Interference')
    
    mixed_signal2, mixing_info2 = mixer2.mix_signals(duration=duration)
    results['Adjacent_Interference'] = {
        'signal': mixed_signal2,
        'info': mixing_info2
    }
    
    print(f"  Mixed signal power: {np.mean(np.abs(mixed_signal2)**2):.6f}")
    print(f"  Primary vs interferer power ratio: {0 - (-15)} dB")
    
    # Scenario 3: Multiple standard interference
    print("\nScenario 3: Multi-Standard Interference")
    mixer3 = SignalMixer(sample_rate)
    
    # Add multiple signals at different power levels
    mixer3.add_signal(lte_signal, carrier_freq=2.1e9, power_db=0, label='LTE_Strong')
    mixer3.add_signal(gsm_signal, carrier_freq=1.8e9, power_db=-10, label='GSM_Medium')
    
    # Add CW interferer
    cw_interference = InterferenceGenerator.generate_cw_tone(
        sample_rate, duration, freq=0, power_db=-20
    )
    mixer3.add_signal(cw_interference, carrier_freq=2.05e9, power_db=-20, 
                     label='CW_Interferer')
    
    mixed_signal3, mixing_info3 = mixer3.mix_signals(duration=duration)
    results['Multi_Standard'] = {
        'signal': mixed_signal3,
        'info': mixing_info3
    }
    
    print(f"  Mixed signal power: {np.mean(np.abs(mixed_signal3)**2):.6f}")
    print(f"  Power levels: LTE=0dB, GSM=-10dB, CW=-20dB")
    
    return results


def analyze_results(single_signals, channel_effects, mixed_scenarios):
    """Analyze and summarize results"""
    print("\n=== Results Analysis ===")
    
    # Signal statistics
    print("\nSingle Signal Statistics:")
    for std_name, sig_data in single_signals.items():
        signal = sig_data['signal']
        power = np.mean(np.abs(signal)**2)
        peak_power = np.max(np.abs(signal)**2)
        papr = 10 * np.log10(peak_power / power)
        
        print(f"  {std_name}:")
        print(f"    Length: {len(signal)} samples")
        print(f"    Average power: {power:.6f}")
        print(f"    PAPR: {papr:.2f} dB")
        print(f"    Bandwidth: {sig_data['bandwidth']/1e6:.1f} MHz")
    
    # Channel effects impact
    print("\nChannel Effects Impact:")
    for std_name in single_signals.keys():
        original_power = np.mean(np.abs(single_signals[std_name]['signal'])**2)
        
        print(f"  {std_name}:")
        for channel_type, affected_signal in channel_effects[std_name].items():
            affected_power = np.mean(np.abs(affected_signal)**2)
            power_change = 10 * np.log10(affected_power / original_power)
            print(f"    {channel_type}: {power_change:+.2f} dB power change")
    
    # Mixed scenarios summary
    print("\nMixed Scenarios Summary:")
    for scenario_name, scenario_data in mixed_scenarios.items():
        info = scenario_data['info']
        signal = scenario_data['signal']
        
        print(f"  {scenario_name}:")
        print(f"    Duration: {info['duration']:.3f} s")
        print(f"    Components: {info['num_signals']}")
        print(f"    Total power: {np.mean(np.abs(signal)**2):.6f}")
        
        for sig_info in info['signals']:
            print(f"      - {sig_info['label']}: {sig_info['carrier_freq']/1e9:.2f} GHz, "
                  f"{sig_info['power_db']:+.0f} dB")


def save_sample_data(mixed_scenarios, output_dir='data/processed'):
    """Save sample dataset files"""
    print(f"\n=== Saving Sample Data to {output_dir} ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for scenario_name, scenario_data in mixed_scenarios.items():
        signal = scenario_data['signal']
        info = scenario_data['info']
        
        # Save signal data
        signal_file = os.path.join(output_dir, f'{scenario_name}_signal.npy')
        np.save(signal_file, signal)
        
        # Save metadata
        metadata_file = os.path.join(output_dir, f'{scenario_name}_metadata.npy')
        np.save(metadata_file, info, allow_pickle=True)
        
        print(f"  Saved {scenario_name}: {len(signal)} samples")
    
    print(f"Sample data saved to {output_dir}/")


def main():
    """Main demonstration function"""
    print("RF Signal Source Separation Dataset - Demonstration")
    print("=" * 60)
    
    try:
        # Generate individual standard signals
        single_signals = generate_single_standard_signals()
        
        # Apply channel effects
        sample_rate = 10e6
        channel_effects = apply_channel_effects(single_signals, sample_rate)
        
        # Create mixed scenarios
        mixed_scenarios = create_mixed_scenarios(single_signals, sample_rate)
        
        # Analyze results
        analyze_results(single_signals, channel_effects, mixed_scenarios)
        
        # Save sample data
        save_sample_data(mixed_scenarios)
        
        print("\n" + "=" * 60)
        print("Demonstration completed successfully!")
        print("Generated signals for: GSM (2G), LTE (4G)")
        print("Applied channel models: AWGN, Rayleigh fading, Multipath")
        print("Created mixed scenarios: Co-existence, Interference, Multi-standard")
        print("Saved sample dataset files to data/processed/")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()