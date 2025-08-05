"""
Simple configuration loader for RF signal specifications
"""
import yaml
import os
from pathlib import Path


def load_signal_specs():
    """Load signal specifications from YAML config file"""
    config_path = Path(__file__).parent.parent.parent / "config" / "signal_specs.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        specs = yaml.safe_load(f)
    
    return specs


def get_standard_specs(standard_name):
    """Get specifications for a specific cellular standard"""
    specs = load_signal_specs()
    
    standard_mapping = {
        '2g': 'GSM_2G',
        'gsm': 'GSM_2G',
        '3g': 'UMTS_3G', 
        'umts': 'UMTS_3G',
        '4g': 'LTE_4G',
        'lte': 'LTE_4G',
        '5g': 'NR_5G',
        'nr': 'NR_5G'
    }
    
    key = standard_mapping.get(standard_name.lower())
    if not key:
        raise ValueError(f"Unknown standard: {standard_name}")
    
    return specs[key]


def get_channel_models():
    """Get channel model specifications"""
    specs = load_signal_specs()
    return specs['channel_models']


def get_mimo_config():
    """Get MIMO configuration specifications"""
    specs = load_signal_specs()
    return specs['mimo_config']


if __name__ == "__main__":
    # Test the config loader
    try:
        specs = load_signal_specs()
        print("Successfully loaded signal specifications")
        print(f"Available standards: {list(specs.keys())}")
        
        # Test getting specific standard
        lte_specs = get_standard_specs('4g')
        print(f"LTE modulation schemes: {lte_specs['modulation_schemes']}")
        
    except Exception as e:
        print(f"Error: {e}")