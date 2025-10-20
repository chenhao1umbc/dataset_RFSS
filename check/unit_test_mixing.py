"""
Comprehensive unit tests for signal mixing per Phase 1.3.

Tests:
- SignalMixer with per-source channels
- Co-channel and adjacent-channel modes
- Power ratio accuracy (SIR validation)
- Timing offset handling
- Ground truth preservation
- MIMO spatial mixing
- Realistic 2/3/4-source scenarios
"""

import torch
import math
from src.utils_mixing import SignalMixer, MIMOSignalMixer
from src.run_lte import generate_lte_signal
from src.run_5g import generate_5g_signal
from src.run_gsm import generate_gsm_signal
from src.run_umts import generate_umts_signal


def test_cochannel_basic_mixing():
    """Test basic co-channel mixing with 2 sources."""
    print("\nTest: Co-channel basic mixing")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=0.0)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', power_db=-5.0)

    result = mixer.mix(mode='co-channel')

    assert result['mixed_signal'].shape[0] > 0
    assert len(result['source_signals_clean']) == 2
    assert len(result['source_signals_channelized']) == 2
    assert len(result['source_signals_aligned']) == 2
    assert result['mixing_params']['num_sources'] == 2
    assert result['mixing_params']['mode'] == 'co-channel'
    print("  PASS")


def test_power_ratio_accuracy():
    """Test that power ratios are maintained correctly."""
    print("\nTest: Power ratio accuracy")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    power_db_1 = 5.0
    power_db_2 = -5.0

    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=power_db_1)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', power_db=power_db_2)

    result = mixer.mix(mode='co-channel')

    # Check source powers
    src1_power_db = 10 * math.log10(torch.mean(torch.abs(result['source_signals_aligned'][0])**2).item())
    src2_power_db = 10 * math.log10(torch.mean(torch.abs(result['source_signals_aligned'][1])**2).item())

    # Power ratio should match target (within 0.5 dB tolerance)
    measured_ratio = src1_power_db - src2_power_db
    expected_ratio = power_db_1 - power_db_2
    error = abs(measured_ratio - expected_ratio)
    assert error < 0.5, f"Power ratio error: {error} dB"
    print(f"  Power ratio error: {error:.3f} dB - PASS")


def test_timing_offset_handling():
    """Test timing offset application."""
    print("\nTest: Timing offset handling")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    offset_samples = 500

    mixer.add_source(signal=lte_result['signal'], label='LTE', timing_offset_samples=0)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', timing_offset_samples=offset_samples)

    result = mixer.mix(mode='co-channel')

    # Check that timing offsets are preserved in metadata
    assert result['mixing_params']['timing_offsets_samples'][0] == 0
    assert result['mixing_params']['timing_offsets_samples'][1] == offset_samples

    # Check that second source has zeros at beginning
    src2_aligned = result['source_signals_aligned'][1]
    assert torch.allclose(src2_aligned[:offset_samples], torch.zeros(offset_samples, dtype=torch.complex64))
    print("  PASS")


def test_adjacent_channel_frequency_offsets():
    """Test frequency shifting in adjacent-channel mode."""
    print("\nTest: Adjacent-channel frequency offsets")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=5.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=50.0)

    freq_offset_1 = -1e6  # -1 MHz
    freq_offset_2 = 1e6   # +1 MHz

    mixer.add_source(signal=lte_result['signal'], label='LTE', freq_offset_hz=freq_offset_1)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', freq_offset_hz=freq_offset_2)

    result = mixer.mix(mode='adjacent-channel')

    assert result['mixing_params']['frequency_offsets_hz'][0] == freq_offset_1
    assert result['mixing_params']['frequency_offsets_hz'][1] == freq_offset_2
    print("  PASS")


def test_ground_truth_preservation():
    """Test that all ground truth signals are preserved."""
    print("\nTest: Ground truth preservation")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    lte_signal = lte_result['signal']
    nr_signal = nr_result['signal']

    mixer.add_source(signal=lte_signal, label='LTE', power_db=0.0,
                     channel_params={'tdl_model': 'TDL-A', 'doppler_hz': 50.0})
    mixer.add_source(signal=nr_signal, label='5G NR', power_db=-5.0,
                     channel_params={'tdl_model': 'TDL-C', 'doppler_hz': 100.0})

    result = mixer.mix(mode='co-channel')

    # Verify clean signals match input
    assert torch.allclose(result['source_signals_clean'][0], lte_signal)
    assert torch.allclose(result['source_signals_clean'][1], nr_signal)

    # Verify channelized signals are different from clean (due to channel)
    assert not torch.allclose(result['source_signals_channelized'][0], lte_signal)
    assert not torch.allclose(result['source_signals_channelized'][1], nr_signal)

    # Verify aligned signals have correct lengths
    assert len(result['source_signals_aligned'][0]) == len(result['mixed_signal'])
    assert len(result['source_signals_aligned'][1]) == len(result['mixed_signal'])

    # Verify channel params are preserved
    assert result['channel_params'][0]['tdl_model'] == 'TDL-A'
    assert result['channel_params'][1]['tdl_model'] == 'TDL-C'
    print("  PASS")


def test_metadata_completeness():
    """Test that metadata is complete and correct."""
    print("\nTest: Metadata completeness")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=0.0)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', power_db=-5.0)

    result = mixer.mix(mode='co-channel')

    # Check metadata
    assert 'scenario_name' in result['metadata']
    assert 'standards' in result['metadata']
    assert 'generation_timestamp' in result['metadata']
    assert 'device' in result['metadata']

    assert result['metadata']['standards'] == ['LTE', '5G NR']
    assert 'LTE' in result['metadata']['scenario_name']
    assert '5G NR' in result['metadata']['scenario_name']

    # Check mixing parameters
    assert 'num_sources' in result['mixing_params']
    assert 'mode' in result['mixing_params']
    assert 'power_ratios_db' in result['mixing_params']
    assert 'sample_rate' in result['mixing_params']
    print("  PASS")


def test_three_source_mixing():
    """Test realistic 3-source scenario (GSM + UMTS + LTE)."""
    print("\nTest: Three-source mixing scenario")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    gsm_result = generate_gsm_signal(duration_ms=1.0, sample_rate=sample_rate)
    umts_result = generate_umts_signal(duration_ms=1.0, spreading_factor=16, sample_rate=sample_rate)
    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=5.0)

    mixer.add_source(signal=gsm_result['signal'], label='GSM', power_db=0.0,
                     freq_offset_hz=-1e6, channel_params={'tdl_model': 'TDL-A'})
    mixer.add_source(signal=umts_result['signal'], label='UMTS', power_db=-3.0,
                     freq_offset_hz=0.0, channel_params={'tdl_model': 'TDL-B'})
    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=-6.0,
                     freq_offset_hz=1.5e6, channel_params={'tdl_model': 'TDL-C'})

    result = mixer.mix(mode='adjacent-channel')

    assert result['mixing_params']['num_sources'] == 3
    assert len(result['source_signals_clean']) == 3
    assert result['metadata']['standards'] == ['GSM', 'UMTS', 'LTE']
    assert result['mixing_params']['power_ratios_db'] == [0.0, -3.0, -6.0]
    print("  PASS")


def test_near_far_scenario():
    """Test near-far scenario with extreme power imbalance."""
    print("\nTest: Near-far scenario")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=10.0)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', power_db=-10.0)

    result = mixer.mix(mode='co-channel')

    # Verify 20 dB power difference
    power_diff = result['mixing_params']['power_ratios_db'][0] - result['mixing_params']['power_ratios_db'][1]
    assert abs(power_diff - 20.0) < 0.01
    print(f"  Power difference: {power_diff:.1f} dB - PASS")


def test_mimo_basic_mixing():
    """Test basic MIMO mixing with 2x2 configuration."""
    print("\nTest: MIMO basic 2x2 mixing")
    sample_rate = 15.36e6
    num_tx = 2
    num_rx = 2

    mixer = MIMOSignalMixer(num_tx=num_tx, num_rx=num_rx, sample_rate=sample_rate)

    lte_signals = []
    for i in range(num_tx):
        result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=5.0)
        lte_signals.append(result['signal'])
    lte_signals_mimo = torch.stack(lte_signals)

    nr_signals = []
    for i in range(num_tx):
        result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=50.0)
        nr_signals.append(result['signal'])
    nr_signals_mimo = torch.stack(nr_signals)

    mixer.add_source(signals=lte_signals_mimo, label='LTE', power_db=0.0, doppler_hz=50.0)
    mixer.add_source(signals=nr_signals_mimo, label='5G NR', power_db=-6.0, doppler_hz=100.0)

    result = mixer.mix(mode='co-channel')

    assert result['mixed_signals_mimo'].shape[0] == num_rx
    assert result['mixed_signals_mimo'].shape[1] > 0
    assert result['mixing_params']['num_sources'] == 2
    assert result['mixing_params']['num_tx'] == num_tx
    assert result['mixing_params']['num_rx'] == num_rx
    print("  PASS")


def test_mimo_4x4_configuration():
    """Test MIMO 4x4 mixing configuration."""
    print("\nTest: MIMO 4x4 configuration")
    sample_rate = 15.36e6
    num_tx = 4
    num_rx = 4

    mixer = MIMOSignalMixer(num_tx=num_tx, num_rx=num_rx, sample_rate=sample_rate,
                           spatial_correlation=0.5)

    lte_signals = []
    for i in range(num_tx):
        result = generate_lte_signal(duration_ms=0.5, bandwidth_mhz=5.0)
        lte_signals.append(result['signal'])
    lte_signals_mimo = torch.stack(lte_signals)

    nr_signals = []
    for i in range(num_tx):
        result = generate_5g_signal(duration_ms=0.5, numerology=1, bandwidth_mhz=50.0)
        nr_signals.append(result['signal'])
    nr_signals_mimo = torch.stack(nr_signals)

    mixer.add_source(signals=lte_signals_mimo, label='LTE', power_db=0.0)
    mixer.add_source(signals=nr_signals_mimo, label='5G NR', power_db=-5.0)

    result = mixer.mix(mode='co-channel')

    assert result['mixed_signals_mimo'].shape[0] == num_rx
    assert result['mixed_signals_mimo'].shape[1] > 0
    assert result['mixing_params']['spatial_correlation'] == 0.5

    # Check per-antenna power variations
    antenna_powers = []
    for i in range(num_rx):
        power = torch.mean(torch.abs(result['mixed_signals_mimo'][i])**2).item()
        antenna_powers.append(power)

    # Powers should be similar but not identical (due to spatial mixing)
    mean_power = sum(antenna_powers) / len(antenna_powers)
    power_variation = max([abs(p - mean_power) / mean_power for p in antenna_powers])
    assert power_variation < 0.5, f"Antenna power variation too high: {power_variation}"
    print(f"  Antenna power variation: {power_variation*100:.1f}% - PASS")


def test_mixer_clear():
    """Test mixer clearing functionality."""
    print("\nTest: Mixer clear")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=0.0)

    sources_before = len(mixer.sources)
    mixer.clear()
    sources_after = len(mixer.sources)

    assert sources_before == 1
    assert sources_after == 0
    print("  PASS")


def test_mixer_source_info():
    """Test get_source_info method."""
    print("\nTest: Mixer source info")
    sample_rate = 15.36e6

    mixer = SignalMixer(sample_rate=sample_rate)

    lte_result = generate_lte_signal(duration_ms=1.0, bandwidth_mhz=10.0)
    nr_result = generate_5g_signal(duration_ms=1.0, numerology=1, bandwidth_mhz=100.0)

    mixer.add_source(signal=lte_result['signal'], label='LTE', power_db=0.0,
                     freq_offset_hz=1e6)
    mixer.add_source(signal=nr_result['signal'], label='5G NR', power_db=-5.0,
                     timing_offset_samples=100)

    info = mixer.get_source_info()

    assert len(info) == 2
    assert info[0]['label'] == 'LTE'
    assert info[0]['freq_offset_hz'] == 1e6
    assert info[1]['label'] == '5G NR'
    assert info[1]['timing_offset_samples'] == 100
    print("  PASS")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("Signal Mixing Comprehensive Unit Tests")
    print("=" * 70)

    test_cochannel_basic_mixing()
    test_power_ratio_accuracy()
    test_timing_offset_handling()
    test_adjacent_channel_frequency_offsets()
    test_ground_truth_preservation()
    test_metadata_completeness()
    test_three_source_mixing()
    test_near_far_scenario()
    test_mimo_basic_mixing()
    test_mimo_4x4_configuration()
    test_mixer_clear()
    test_mixer_source_info()

    print("\n" + "=" * 70)
    print("All tests PASSED")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()
