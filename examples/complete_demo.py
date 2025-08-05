"""
Complete RF dataset demonstration with all cellular standards, MIMO, and validation
Shows full 2G/3G/4G/5G generation pipeline with channel effects and validation
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.nr_generator import NRGenerator
from channel_models.basic_channels import ChannelSimulator
from mixing.signal_mixer import SignalMixer, InterferenceGenerator
from mimo.mimo_channel import MIMOSystemSimulator
from validation.signal_metrics import ValidationReport


def generate_all_standards():
    """Generate signals from all cellular standards"""
    print("=== Generating All Cellular Standards ===")
    
    # Common parameters
    sample_rate = 30.72e6  # 30.72 MHz (suitable for most standards)
    duration = 0.005       # 5 ms
    
    results = {}
    
    # Generate 2G GSM signal  
    print("Generating GSM (2G) signal...")
    gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
    gsm_signal = gsm_gen.generate_baseband()
    results['GSM'] = {
        'signal': gsm_signal,
        'generator': gsm_gen,
        'standard': '2G',
        'power': np.mean(np.abs(gsm_signal)**2)
    }
    print(f"  GSM power: {results['GSM']['power']:.6f}")
    
    # Generate 3G UMTS signal
    print("Generating UMTS (3G) signal...")
    umts_gen = UMTSGenerator(sample_rate=sample_rate, duration=duration, 
                            spreading_factor=128, num_users=4)
    umts_signal = umts_gen.generate_baseband()
    results['UMTS'] = {
        'signal': umts_signal,
        'generator': umts_gen,
        'standard': '3G', 
        'power': np.mean(np.abs(umts_signal)**2)
    }
    print(f"  UMTS power: {results['UMTS']['power']:.6f}")
    
    # Generate 4G LTE signal
    print("Generating LTE (4G) signal...")
    lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration,
                          bandwidth=20, modulation='64QAM')
    lte_signal = lte_gen.generate_baseband()
    results['LTE'] = {
        'signal': lte_signal,
        'generator': lte_gen,
        'standard': '4G',
        'power': np.mean(np.abs(lte_signal)**2)
    }
    print(f"  LTE power: {results['LTE']['power']:.6f}")
    
    # Generate 5G NR signal
    print("Generating NR (5G) signal...")
    nr_gen = NRGenerator(sample_rate=sample_rate, duration=duration,
                        bandwidth=100, numerology=1, modulation='256QAM')
    nr_signal = nr_gen.generate_baseband()
    results['NR'] = {
        'signal': nr_signal,
        'generator': nr_gen,
        'standard': '5G',
        'power': np.mean(np.abs(nr_signal)**2)
    }
    print(f"  NR power: {results['NR']['power']:.6f}")
    
    return results, sample_rate


def apply_mimo_processing(signals, sample_rate):
    """Apply MIMO processing to generated signals"""
    print("\n=== Applying MIMO Processing ===")
    
    mimo_results = {}
    
    # Test different MIMO configurations
    mimo_configs = [(2, 2), (4, 4)]
    
    for num_tx, num_rx in mimo_configs:
        print(f"\nTesting {num_tx}×{num_rx} MIMO...")
        
        # Create MIMO simulator
        mimo_sim = MIMOSystemSimulator(num_tx, num_rx, correlation='medium')
        
        config_results = {}
        
        for std_name, sig_data in signals.items():
            signal = sig_data['signal']
            
            # Prepare MIMO input (duplicate signal for multiple streams)
            if len(signal) > 1000:  # Use subset for faster processing
                signal_subset = signal[:1000]
            else:
                signal_subset = signal
                
            # Create multiple streams
            mimo_input = np.tile(signal_subset, (num_tx, 1))
            
            # Apply different processing methods
            processing_results = {}
            
            for method in ['none', 'zf', 'mmse']:
                try:
                    rx_signals, H = mimo_sim.simulate_transmission(
                        mimo_input, precoding=method, snr_db=20
                    )
                    
                    # Calculate performance metrics
                    metrics = mimo_sim.calculate_performance_metrics(mimo_input, rx_signals)
                    
                    processing_results[method] = {
                        'rx_signals': rx_signals,
                        'channel_matrix': H,
                        'metrics': metrics
                    }
                    
                    print(f"    {std_name} {method.upper()}: SNR={metrics['snr_measured']:.2f} dB")
                    
                except Exception as e:
                    print(f"    {std_name} {method.upper()}: Failed ({str(e)[:50]})")
                    processing_results[method] = None
            
            config_results[std_name] = processing_results
        
        mimo_results[f'{num_tx}x{num_rx}'] = config_results
    
    return mimo_results


def create_complex_scenarios(signals, sample_rate):
    """Create complex multi-standard scenarios"""
    print("\n=== Creating Complex Multi-Standard Scenarios ===")
    
    duration = 0.005  # 5 ms
    scenarios = {}
    
    # Scenario 1: All standards coexistence
    print("\nScenario 1: All Standards Co-existence")
    mixer1 = SignalMixer(sample_rate)
    
    # Add all standards at different frequencies and power levels
    mixer1.add_signal(signals['GSM']['signal'], carrier_freq=900e6, power_db=0, 
                     label='GSM_2G')
    mixer1.add_signal(signals['UMTS']['signal'], carrier_freq=2.1e9, power_db=-5,
                     label='UMTS_3G') 
    mixer1.add_signal(signals['LTE']['signal'], carrier_freq=1.8e9, power_db=-3,
                     label='LTE_4G')
    mixer1.add_signal(signals['NR']['signal'], carrier_freq=3.5e9, power_db=-2,
                     label='NR_5G')
    
    mixed_signal1, info1 = mixer1.mix_signals(duration=duration)
    scenarios['All_Standards'] = {
        'signal': mixed_signal1,
        'info': info1,
        'description': 'All 2G/3G/4G/5G standards coexisting'
    }
    
    print(f"  Mixed signal power: {np.mean(np.abs(mixed_signal1)**2):.6f}")
    print(f"  Components: {info1['num_signals']}")
    
    # Scenario 2: Dense interference environment
    print("\nScenario 2: Dense Interference Environment")
    mixer2 = SignalMixer(sample_rate)
    
    # Primary 5G signal
    mixer2.add_signal(signals['NR']['signal'], carrier_freq=3.5e9, power_db=0,
                     label='NR_Primary')
    
    # Multiple interferers
    interference1 = InterferenceGenerator.generate_cw_tone(
        sample_rate, duration, 0, power_db=-15
    )
    mixer2.add_signal(interference1, carrier_freq=3.48e9, power_db=-15,
                     label='CW_Interferer')
    
    interference2 = InterferenceGenerator.generate_narrowband_noise(
        sample_rate, duration, 0, 10e6, power_db=-12
    )  
    mixer2.add_signal(interference2, carrier_freq=3.52e9, power_db=-12,
                     label='Wideband_Interferer')
    
    # Adjacent LTE signal
    mixer2.add_signal(signals['LTE']['signal'], carrier_freq=3.45e9, power_db=-8,
                     label='LTE_Adjacent')
    
    mixed_signal2, info2 = mixer2.mix_signals(duration=duration)
    scenarios['Dense_Interference'] = {
        'signal': mixed_signal2,
        'info': info2,
        'description': 'Dense interference with multiple signal types'
    }
    
    print(f"  Mixed signal power: {np.mean(np.abs(mixed_signal2)**2):.6f}")
    print(f"  Components: {info2['num_signals']}")
    
    return scenarios


def apply_advanced_channels(scenarios, sample_rate):
    """Apply advanced channel effects to complex scenarios"""
    print("\n=== Applying Advanced Channel Effects ===")
    
    channel_results = {}
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\nProcessing {scenario_name}...")
        
        signal = scenario_data['signal']
        scenario_results = {}
        
        # Channel 1: Urban multipath + high Doppler
        urban_sim = ChannelSimulator(sample_rate)
        urban_sim.add_multipath().add_rayleigh_fading(200).add_awgn(5)
        urban_signal = urban_sim.apply(signal)
        scenario_results['Urban_Mobile'] = urban_signal
        
        # Channel 2: Rural with Rician fading
        rural_sim = ChannelSimulator(sample_rate)  
        rural_sim.add_rician_fading(50, 10).add_awgn(15)
        rural_signal = rural_sim.apply(signal)
        scenario_results['Rural_LOS'] = rural_signal
        
        # Channel 3: Indoor multipath
        indoor_sim = ChannelSimulator(sample_rate)
        indoor_sim.add_multipath().add_rayleigh_fading(10).add_awgn(8)
        indoor_signal = indoor_sim.apply(signal)
        scenario_results['Indoor_NLOS'] = indoor_signal
        
        channel_results[scenario_name] = scenario_results
        
        # Print power comparison
        original_power = np.mean(np.abs(signal)**2)
        print(f"  Original power: {original_power:.6f}")
        for channel_type, processed_signal in scenario_results.items():
            processed_power = np.mean(np.abs(processed_signal)**2)
            power_change = 10 * np.log10(processed_power / original_power)
            print(f"    {channel_type}: {power_change:+.2f} dB")
    
    return channel_results


def validate_all_signals(signals, sample_rate):
    """Validate all generated signals"""
    print("\n=== Signal Validation ===")
    
    validator = ValidationReport()
    validation_results = {}
    
    for std_name, sig_data in signals.items():
        print(f"\nValidating {std_name} signal...")
        
        signal = sig_data['signal']
        standard = sig_data['standard']
        
        # Generate validation report
        report = validator.generate_signal_report(signal, sample_rate, std_name)
        validation_results[std_name] = report
        
        # Print summary
        print(f"  Duration: {report['duration']*1000:.2f} ms")
        print(f"  PAPR: {report['power_metrics']['papr_db']:.2f} dB")
        print(f"  Bandwidth: {report['bandwidth_metrics']['bandwidth']/1e6:.3f} MHz")
        print(f"  SNR: {report['snr_metrics']['snr_db']:.2f} dB")
        
        if report['validation_results']:
            vr = report['validation_results']
            status = "✓ PASS" if vr['overall_valid'] else "✗ FAIL"
            print(f"  Standards Compliance: {status}")
    
    return validation_results


def save_complete_dataset(signals, scenarios, channel_results, output_dir='data/processed'):
    """Save complete dataset with all components"""
    print(f"\n=== Saving Complete Dataset to {output_dir} ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual standards
    for std_name, sig_data in signals.items():
        signal_file = os.path.join(output_dir, f'{std_name}_baseband.npy')
        np.save(signal_file, sig_data['signal'])
        print(f"  Saved {std_name} baseband: {len(sig_data['signal'])} samples")
    
    # Save mixed scenarios
    for scenario_name, scenario_data in scenarios.items():
        signal_file = os.path.join(output_dir, f'{scenario_name}_mixed.npy')
        metadata_file = os.path.join(output_dir, f'{scenario_name}_metadata.npy')
        
        np.save(signal_file, scenario_data['signal'])
        np.save(metadata_file, scenario_data['info'], allow_pickle=True)
        
        print(f"  Saved {scenario_name}: {len(scenario_data['signal'])} samples")
    
    # Save channel processed signals
    for scenario_name, channel_data in channel_results.items():
        for channel_type, processed_signal in channel_data.items():
            filename = f'{scenario_name}_{channel_type}.npy'
            signal_file = os.path.join(output_dir, filename)
            np.save(signal_file, processed_signal)
    
    print(f"Complete dataset saved to {output_dir}/")


def main():
    """Main demonstration function"""
    print("RF Signal Source Separation Dataset - Complete Demonstration")
    print("=" * 70)
    
    try:
        # Generate all cellular standards
        signals, sample_rate = generate_all_standards()
        
        # Apply MIMO processing
        mimo_results = apply_mimo_processing(signals, sample_rate)
        
        # Create complex scenarios
        scenarios = create_complex_scenarios(signals, sample_rate)
        
        # Apply advanced channel effects
        channel_results = apply_advanced_channels(scenarios, sample_rate)
        
        # Validate all signals
        validation_results = validate_all_signals(signals, sample_rate)
        
        # Save complete dataset
        save_complete_dataset(signals, scenarios, channel_results)
        
        # Final summary
        print("\n" + "=" * 70)
        print("Complete demonstration finished successfully!")
        print("\nGenerated Components:")
        print("  ✓ 2G GSM with GMSK modulation")
        print("  ✓ 3G UMTS with CDMA spreading") 
        print("  ✓ 4G LTE with OFDM and 64-QAM")
        print("  ✓ 5G NR with flexible numerology and 256-QAM")
        print("  ✓ MIMO processing (2×2, 4×4 configurations)")
        print("  ✓ Advanced channel models (urban, rural, indoor)")
        print("  ✓ Complex interference scenarios")
        print("  ✓ Standards compliance validation")
        
        print(f"\nDataset Statistics:")
        print(f"  Sample Rate: {sample_rate/1e6:.2f} MHz")
        print(f"  Signal Duration: 5 ms")
        print(f"  Total Standards: {len(signals)}")
        print(f"  Mixed Scenarios: {len(scenarios)}")
        print(f"  Channel Variants: {sum(len(v) for v in channel_results.values())}")
        
        print(f"\nFiles saved to data/processed/")
        print("Ready for RF source separation research!")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()