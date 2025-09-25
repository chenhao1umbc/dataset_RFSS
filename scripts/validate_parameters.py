#!/usr/bin/env python3
"""
Validate all technical parameters against paper claims
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.config_loader import get_standard_specs, get_mimo_config
from src.signal_generation.gsm_generator import GSMGenerator
from src.signal_generation.umts_generator import UMTSGenerator
from src.signal_generation.lte_generator import LTEGenerator
from src.signal_generation.nr_generator import NRGenerator


def validate_gsm_parameters():
    """Validate GSM parameters against paper claims"""
    print("=== GSM Parameter Validation ===")
    specs = get_standard_specs('2g')
    
    # Paper claims: 270.833 ksps symbol rate, BT=0.3, 200 kHz bandwidth
    symbol_rate = specs['symbol_rate']  # ksps
    channel_bw = specs['channel_bandwidth']  # MHz
    
    gsm_gen = GSMGenerator(sample_rate=10e6, duration=0.01)
    bt = gsm_gen.bt
    
    print(f"Symbol rate: {symbol_rate} ksps (paper: 270.833 ksps)")
    print(f"Channel BW: {channel_bw} MHz (paper: 0.2 MHz)")
    print(f"BT product: {bt} (paper: 0.3)")
    
    # Check matches
    matches = []
    matches.append(("Symbol rate", abs(symbol_rate - 270.833) < 0.001))
    matches.append(("Channel BW", abs(channel_bw - 0.2) < 0.001))
    matches.append(("BT product", abs(bt - 0.3) < 0.001))
    
    for param, match in matches:
        status = "✓" if match else "✗"
        print(f"  {status} {param}")
    
    return all(match for _, match in matches)


def validate_umts_parameters():
    """Validate UMTS parameters against paper claims"""
    print("\n=== UMTS Parameter Validation ===")
    specs = get_standard_specs('3g')
    
    # Paper claims: 3.84 Mcps chip rate, 5 MHz bandwidth
    chip_rate = specs['chip_rate']  # Mcps
    frame_duration = specs['frame_duration']  # ms
    
    print(f"Chip rate: {chip_rate} Mcps (paper: 3.84 Mcps)")
    print(f"Frame duration: {frame_duration} ms (paper: 10 ms)")
    
    # Check matches
    matches = []
    matches.append(("Chip rate", abs(chip_rate - 3.84) < 0.001))
    matches.append(("Frame duration", abs(frame_duration - 10) < 0.001))
    
    for param, match in matches:
        status = "✓" if match else "✗"
        print(f"  {status} {param}")
    
    return all(match for _, match in matches)


def validate_lte_parameters():
    """Validate LTE parameters against paper claims"""
    print("\n=== LTE Parameter Validation ===")
    specs = get_standard_specs('4g')
    
    # Paper claims: 15 kHz subcarrier spacing, OFDMA
    subcarrier_spacing = specs['subcarrier_spacing']  # kHz
    frame_duration = specs['frame_duration']  # ms
    modulation_schemes = specs['modulation_schemes']
    
    print(f"Subcarrier spacing: {subcarrier_spacing} kHz (paper: 15 kHz)")
    print(f"Frame duration: {frame_duration} ms (paper: 10 ms)")
    print(f"Modulation schemes: {modulation_schemes} (paper: QPSK, 16QAM, 64QAM)")
    
    # Check matches
    matches = []
    matches.append(("Subcarrier spacing", abs(subcarrier_spacing - 15) < 0.001))
    matches.append(("Frame duration", abs(frame_duration - 10) < 0.001))
    matches.append(("Modulation schemes", set(['QPSK', '16QAM', '64QAM']).issubset(set(modulation_schemes))))
    
    for param, match in matches:
        status = "✓" if match else "✗"
        print(f"  {status} {param}")
    
    return all(match for _, match in matches)


def validate_nr_parameters():
    """Validate 5G NR parameters against paper claims"""
    print("\n=== 5G NR Parameter Validation ===")
    specs = get_standard_specs('5g')
    
    # Paper claims: flexible numerology, 256QAM support
    subcarrier_spacings = specs['subcarrier_spacing']  # kHz list
    modulation_schemes = specs['modulation_schemes']
    numerology = specs['numerology']
    
    print(f"Subcarrier spacings: {subcarrier_spacings} kHz (paper: [15, 30, 60, 120, 240])")
    print(f"Modulation schemes: {modulation_schemes}")
    print(f"Numerology: {numerology} (paper: [0, 1, 2, 3, 4])")
    
    # Check matches
    matches = []
    matches.append(("Subcarrier spacings", set([15, 30, 60, 120, 240]).issubset(set(subcarrier_spacings))))
    matches.append(("256QAM support", '256QAM' in modulation_schemes))
    matches.append(("Numerology", set([0, 1, 2, 3, 4]).issubset(set(numerology))))
    
    for param, match in matches:
        status = "✓" if match else "✗"
        print(f"  {status} {param}")
    
    return all(match for _, match in matches)


def validate_mimo_parameters():
    """Validate MIMO parameters against paper claims"""
    print("\n=== MIMO Parameter Validation ===")
    mimo_config = get_mimo_config()
    
    antenna_configs = mimo_config['antenna_configurations']
    print(f"Supported MIMO configurations: {list(antenna_configs.keys())}")
    
    # Paper claims: up to 8×8 MIMO
    has_8x8 = 'MIMO_8x8' in antenna_configs and antenna_configs['MIMO_8x8'] == [8, 8]
    no_16x16 = 'MIMO_16x16' not in antenna_configs
    
    print(f"  ✓ 8×8 MIMO support: {has_8x8}")
    print(f"  ✓ No false 16×16 claim: {no_16x16}")
    
    return has_8x8 and no_16x16


def main():
    """Main validation function"""
    print("Technical Parameter Validation Against Paper Claims")
    print("=" * 60)
    
    results = []
    
    results.append(("GSM", validate_gsm_parameters()))
    results.append(("UMTS", validate_umts_parameters()))  
    results.append(("LTE", validate_lte_parameters()))
    results.append(("5G NR", validate_nr_parameters()))
    results.append(("MIMO", validate_mimo_parameters()))
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_valid = True
    for standard, valid in results:
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"{standard:<15} {status}")
        if not valid:
            all_valid = False
    
    print(f"\nOverall Status: {'✓ ALL PARAMETERS MATCH' if all_valid else '✗ SOME MISMATCHES FOUND'}")
    
    return all_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)