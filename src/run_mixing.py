"""
Demonstration script for signal mixing.

Shows realistic multi-standard coexistence scenarios:
- Co-channel mixing (hardest case)
- Adjacent-channel mixing
- Power ratio effects (near-far)
- Timing offsets
- MIMO spatial mixing
"""

import torch
import argparse
from src.utils_mixing import SignalMixer, MIMOSignalMixer
from src.run_gsm import generate_gsm_signal
from src.run_lte import generate_lte_signal
from src.run_5g import generate_5g_signal
from src.run_umts import generate_umts_signal


def demo_co_channel_mixing():
    """Demonstrate co-channel mixing (hardest separation case)."""
    print("\n" + "=" * 70)
    print("Demo: Co-Channel Mixing (LTE + 5G NR)")
    print("=" * 70)

    sample_rate = 15.36e6
    duration = 0.001

    # Create mixer
    mixer = SignalMixer(sample_rate=sample_rate)

    # Generate LTE signal
    lte_result = generate_lte_signal(
        duration_ms=1.0,
        bandwidth_mhz=10.0,
        modulation_scheme='QPSK'
    )
    lte_signal = lte_result['signal']

    # Generate 5G NR signal
    nr_result = generate_5g_signal(
        duration_ms=1.0,
        numerology=1,
        bandwidth_mhz=100.0,
        modulation_scheme='QPSK'
    )
    nr_signal = nr_result['signal']

    # Add sources with different channel effects
    mixer.add_source(
        signal=lte_signal,
        label='LTE',
        power_db=0.0,  # Reference power
        freq_offset_hz=0.0,  # Co-channel
        timing_offset_samples=0,
        channel_params={
            'tdl_model': 'TDL-A',
            'delay_spread_ns': 30.0,
            'doppler_hz': 50.0,
            'cfo_hz': 200.0,  # CFO
            'iq_amplitude_db': 0.5,
            'iq_phase_deg': 2.0,
        }
    )

    mixer.add_source(
        signal=nr_signal,
        label='5G NR',
        power_db=-10.0,  # 10 dB weaker (near-far)
        freq_offset_hz=0.0,  # Co-channel
        timing_offset_samples=500,  # Timing offset
        channel_params={
            'tdl_model': 'TDL-C',
            'delay_spread_ns': 100.0,
            'doppler_hz': 100.0,
            'cfo_hz': -150.0,  # Different CFO
            'iq_amplitude_db': 1.0,
            'iq_phase_deg': 5.0,
        }
    )

    # Mix signals
    result = mixer.mix(mode='co-channel')

    print(f"Number of sources: {result['mixing_params']['num_sources']}")
    print(f"Mode: {result['mixing_params']['mode']}")
    print(f"Mixed signal length: {len(result['mixed_signal'])}")
    print(f"Power ratios: {result['mixing_params']['power_ratios_db']} dB")
    print(f"Timing offsets: {result['mixing_params']['timing_offsets_samples']} samples")
    print(f"Mixed signal power: {torch.mean(torch.abs(result['mixed_signal'])**2).item():.4f}")

    # Check ground truth preservation
    print(f"\nGround truth preservation:")
    print(f"  Clean sources: {len(result['source_signals_clean'])}")
    print(f"  Channelized sources: {len(result['source_signals_channelized'])}")
    print(f"  Aligned sources: {len(result['source_signals_aligned'])}")


def demo_adjacent_channel_mixing():
    """Demonstrate adjacent-channel mixing with frequency offsets."""
    print("\n" + "=" * 70)
    print("Demo: Adjacent-Channel Mixing (GSM + UMTS + LTE)")
    print("=" * 70)

    sample_rate = 15.36e6

    # Create mixer
    mixer = SignalMixer(sample_rate=sample_rate)

    # Generate GSM signal
    gsm_result = generate_gsm_signal(
        duration_ms=1.0,
        sample_rate=sample_rate
    )
    gsm_signal = gsm_result['signal']

    # Generate UMTS signal
    umts_result = generate_umts_signal(
        duration_ms=1.0,
        spreading_factor=16,
        sample_rate=sample_rate
    )
    umts_signal = umts_result['signal']

    # Generate LTE signal
    lte_result = generate_lte_signal(
        duration_ms=1.0,
        bandwidth_mhz=5.0,
        modulation_scheme='QPSK'
    )
    lte_signal = lte_result['signal']

    # Add sources with frequency offsets
    mixer.add_source(
        signal=gsm_signal,
        label='GSM',
        power_db=0.0,
        freq_offset_hz=-1e6,  # -1 MHz offset
        timing_offset_samples=0,
        channel_params={
            'tdl_model': 'TDL-A',
            'doppler_hz': 50.0,
            'cfo_hz': 100.0,
        }
    )

    mixer.add_source(
        signal=umts_signal,
        label='UMTS',
        power_db=-3.0,  # 3 dB weaker
        freq_offset_hz=0.0,  # Center
        timing_offset_samples=200,
        channel_params={
            'tdl_model': 'TDL-B',
            'doppler_hz': 75.0,
            'cfo_hz': -50.0,
        }
    )

    mixer.add_source(
        signal=lte_signal,
        label='LTE',
        power_db=-6.0,  # 6 dB weaker
        freq_offset_hz=1.5e6,  # +1.5 MHz offset
        timing_offset_samples=400,
        channel_params={
            'tdl_model': 'TDL-C',
            'doppler_hz': 100.0,
            'cfo_hz': 200.0,
        }
    )

    # Mix signals
    result = mixer.mix(mode='adjacent-channel')

    print(f"Number of sources: {result['mixing_params']['num_sources']}")
    print(f"Mode: {result['mixing_params']['mode']}")
    print(f"Standards: {result['metadata']['standards']}")
    print(f"Power ratios: {result['mixing_params']['power_ratios_db']} dB")
    print(f"Frequency offsets: {[f/1e6 for f in result['mixing_params']['frequency_offsets_hz']]} MHz")
    print(f"Timing offsets: {result['mixing_params']['timing_offsets_samples']} samples")
    print(f"Mixed signal power: {torch.mean(torch.abs(result['mixed_signal'])**2).item():.4f}")


def demo_near_far_scenario():
    """Demonstrate near-far scenario with extreme power imbalance."""
    print("\n" + "=" * 70)
    print("Demo: Near-Far Scenario (LTE strong + 5G weak)")
    print("=" * 70)

    sample_rate = 15.36e6

    # Create mixer
    mixer = SignalMixer(sample_rate=sample_rate)

    # Generate LTE signal (strong)
    lte_result = generate_lte_signal(
        duration_ms=1.0,
        bandwidth_mhz=10.0,
        modulation_scheme='QPSK'
    )
    lte_signal = lte_result['signal']

    # Generate 5G NR signal (weak)
    nr_result = generate_5g_signal(
        duration_ms=1.0,
        numerology=1,
        bandwidth_mhz=100.0,
        modulation_scheme='QPSK'
    )
    nr_signal = nr_result['signal']

    # Near-far: 20 dB power difference
    mixer.add_source(
        signal=lte_signal,
        label='LTE',
        power_db=10.0,  # Strong
        freq_offset_hz=0.0,
        timing_offset_samples=0,
        channel_params={'tdl_model': 'TDL-A', 'doppler_hz': 50.0}
    )

    mixer.add_source(
        signal=nr_signal,
        label='5G NR',
        power_db=-10.0,  # Weak (20 dB difference)
        freq_offset_hz=0.0,
        timing_offset_samples=100,
        channel_params={'tdl_model': 'TDL-C', 'doppler_hz': 100.0}
    )

    # Mix signals
    result = mixer.mix(mode='co-channel')

    print(f"Standards: {result['metadata']['standards']}")
    print(f"Power ratios: {result['mixing_params']['power_ratios_db']} dB")
    print(f"Power difference: {result['mixing_params']['power_ratios_db'][0] - result['mixing_params']['power_ratios_db'][1]:.1f} dB")
    print(f"Mixed signal power: {torch.mean(torch.abs(result['mixed_signal'])**2).item():.4f}")

    # Check individual source powers
    for i, src in enumerate(result['source_signals_aligned']):
        power = torch.mean(torch.abs(src)**2).item()
        power_db = 10 * torch.log10(torch.tensor(power)).item()
        print(f"Source {i} ({result['metadata']['standards'][i]}): {power_db:.2f} dB")


def demo_mimo_mixing():
    """Demonstrate MIMO mixing with spatial channels."""
    print("\n" + "=" * 70)
    print("Demo: MIMO Mixing (4x4 LTE + 5G)")
    print("=" * 70)

    sample_rate = 15.36e6
    num_tx = 4
    num_rx = 4

    # Create MIMO mixer
    mixer = MIMOSignalMixer(
        num_tx=num_tx,
        num_rx=num_rx,
        sample_rate=sample_rate,
        spatial_correlation=0.5
    )

    # Generate LTE signals for 4 antennas
    lte_signals = []
    for i in range(num_tx):
        result = generate_lte_signal(
            duration_ms=1.0,
            bandwidth_mhz=5.0,
            modulation_scheme='QPSK'
        )
        lte_signals.append(result['signal'])
    lte_signals_mimo = torch.stack(lte_signals)

    # Generate 5G signals for 4 antennas
    nr_signals = []
    for i in range(num_tx):
        result = generate_5g_signal(
            duration_ms=1.0,
            numerology=1,
            bandwidth_mhz=50.0,
            modulation_scheme='QPSK'
        )
        nr_signals.append(result['signal'])
    nr_signals_mimo = torch.stack(nr_signals)

    # Add sources
    mixer.add_source(
        signals=lte_signals_mimo,
        label='LTE',
        power_db=0.0,
        freq_offset_hz=0.0,
        timing_offset_samples=0,
        doppler_hz=50.0
    )

    mixer.add_source(
        signals=nr_signals_mimo,
        label='5G NR',
        power_db=-6.0,
        freq_offset_hz=0.0,
        timing_offset_samples=200,
        doppler_hz=100.0
    )

    # Mix signals
    result = mixer.mix(mode='co-channel')

    print(f"MIMO configuration: {num_tx}x{num_rx}")
    print(f"Number of sources: {result['mixing_params']['num_sources']}")
    print(f"Mixed signals shape: {result['mixed_signals_mimo'].shape}")
    print(f"Power ratios: {result['mixing_params']['power_ratios_db']} dB")
    print(f"Spatial correlation: {result['mixing_params']['spatial_correlation']}")

    # Check per-antenna power
    for i in range(num_rx):
        power = torch.mean(torch.abs(result['mixed_signals_mimo'][i])**2).item()
        print(f"Antenna {i+1} power: {power:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Signal Mixing Demonstration')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['all', 'co-channel', 'adjacent', 'near-far', 'mimo'],
                        help='Which demo to run')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("RFSS Dataset - Signal Mixing Demonstrations")
    print("Based on 3GPP coexistence scenarios and realistic spectrum sharing")
    print("=" * 70)

    if args.demo in ['all', 'co-channel']:
        demo_co_channel_mixing()

    if args.demo in ['all', 'adjacent']:
        demo_adjacent_channel_mixing()

    if args.demo in ['all', 'near-far']:
        demo_near_far_scenario()

    if args.demo in ['all', 'mimo']:
        demo_mimo_mixing()

    print("\n" + "=" * 70)
    print("All demonstrations completed successfully")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
