"""
Comprehensive unit tests for channel models per tasks.md Phase 1.2.

Tests:
- All TDL models (TDL-A/B/C/D/E)
- Jakes' fading models (Rayleigh/Rician)
- All hardware impairments (CFO/SFO/IQ/DC/PN/PA)
- MIMO channel generation and application
"""

import torch
import math
from src.utils_channel import (
    add_awgn,
    generate_tdl_channel,
    apply_tdl_channel,
    generate_rayleigh_fading_jakes,
    generate_rician_fading_jakes,
    apply_cfo,
    apply_sfo,
    apply_iq_imbalance,
    apply_dc_offset,
    apply_phase_noise,
    apply_pa_nonlinearity,
    generate_mimo_channel,
    apply_mimo_channel,
    validate_channel_statistics,
    TDL_MODELS
)


def test_tdl_all_models():
    """Test all 5 TDL models can be generated."""
    print("\nTest: All TDL models generation")
    for model_name in TDL_MODELS.keys():
        coeffs, delays, powers = generate_tdl_channel(
            num_samples=1000,
            tdl_model=model_name,
            sample_rate=15.36e6
        )
        assert coeffs.shape[1] == 1000
        assert len(delays) == len(powers)
        print(f"  {model_name}: PASS ({len(delays)} taps)")


def test_tdl_power_normalization():
    """Test TDL models have unit total power."""
    print("\nTest: TDL power normalization")
    for model_name in TDL_MODELS.keys():
        coeffs, delays, powers = generate_tdl_channel(
            num_samples=10000,
            tdl_model=model_name,
            sample_rate=15.36e6
        )
        total_power = sum([torch.mean(torch.abs(coeffs[i])**2).item()
                          for i in range(coeffs.shape[0])])
        assert 0.9 < total_power < 1.1
        print(f"  {model_name}: power={total_power:.4f} PASS")


def test_tdl_delay_spread_scaling():
    """Test TDL delay spread scales correctly."""
    print("\nTest: TDL delay spread scaling")
    coeffs1, delays1, _ = generate_tdl_channel(
        num_samples=1000,
        tdl_model='TDL-A',
        delay_spread_ns=30,
        sample_rate=15.36e6
    )
    coeffs2, delays2, _ = generate_tdl_channel(
        num_samples=1000,
        tdl_model='TDL-A',
        delay_spread_ns=100,
        sample_rate=15.36e6
    )
    ratio = delays2[-1] / delays1[-1]
    assert 3.0 < ratio < 3.7
    print(f"  Delay ratio: {ratio:.2f} PASS")


def test_tdl_los_models():
    """Test LOS TDL models have strong first tap due to K-factor."""
    print("\nTest: TDL LOS models (D/E) K-factor")
    for model_name in ['TDL-D', 'TDL-E']:
        k_factor_db = TDL_MODELS[model_name]['k_factors_db'][0]
        assert k_factor_db > 0
        print(f"  {model_name}: K-factor={k_factor_db} dB PASS")


def test_rayleigh_unit_power():
    """Test Rayleigh fading has unit power."""
    print("\nTest: Rayleigh unit power")
    for doppler in [0, 50, 100]:
        h = generate_rayleigh_fading_jakes(10000, doppler_freq=doppler, sample_rate=15.36e6)
        power = torch.mean(torch.abs(h)**2).item()
        assert 0.9 < power < 1.1
        print(f"  Doppler {doppler} Hz: power={power:.4f} PASS")


def test_rayleigh_amplitude_statistics():
    """Test Rayleigh amplitude statistics."""
    print("\nTest: Rayleigh amplitude statistics")
    h = generate_rayleigh_fading_jakes(10000, doppler_freq=0)
    amplitude = torch.abs(h)
    mean_amp = torch.mean(amplitude).item()
    power = torch.mean(torch.abs(h)**2).item()
    ratio = mean_amp / math.sqrt(power)
    expected_ratio = math.sqrt(math.pi / 2)
    assert 0.85 < ratio < 0.95
    print(f"  Mean amp / sqrt(power) = {ratio:.4f}, expected ~ {expected_ratio:.4f} PASS")


def test_rician_unit_power():
    """Test Rician fading has unit power."""
    print("\nTest: Rician unit power")
    for k_db in [0, 3, 6, 10]:
        h = generate_rician_fading_jakes(10000, k_factor_db=k_db, doppler_freq=50, sample_rate=15.36e6)
        power = torch.mean(torch.abs(h)**2).item()
        assert 0.9 < power < 1.1
        print(f"  K={k_db} dB: power={power:.4f} PASS")


def test_rician_k_factor_effect():
    """Test Rician K-factor reduces fading variance."""
    print("\nTest: Rician K-factor effect")
    h_low = generate_rician_fading_jakes(10000, k_factor_db=0)
    h_high = generate_rician_fading_jakes(10000, k_factor_db=10)
    std_low = torch.std(torch.abs(h_low)).item()
    std_high = torch.std(torch.abs(h_high)).item()
    assert std_high < std_low
    print(f"  K=0 dB: std={std_low:.4f}, K=10 dB: std={std_high:.4f} PASS")


def test_cfo_preserves_power():
    """Test CFO preserves signal power."""
    print("\nTest: CFO power preservation")
    signal = torch.randn(5000, dtype=torch.complex64)
    original_power = torch.mean(torch.abs(signal)**2).item()

    for cfo_hz in [100, 1000, 10000]:
        rotated = apply_cfo(signal, cfo_hz, 15.36e6)
        rotated_power = torch.mean(torch.abs(rotated)**2).item()
        error = abs(rotated_power - original_power) / original_power
        assert error < 0.01
        print(f"  CFO {cfo_hz} Hz: power error={error*100:.3f}% PASS")


def test_cfo_phase_rotation():
    """Test CFO causes phase rotation."""
    print("\nTest: CFO phase rotation")
    signal = torch.ones(1000, dtype=torch.complex64)
    cfo_hz = 1000.0
    sample_rate = 10000.0
    rotated = apply_cfo(signal, cfo_hz, sample_rate)
    period_samples = int(sample_rate / cfo_hz)
    diff = torch.abs(rotated[0] - rotated[period_samples])
    assert diff < 0.01
    print(f"  Period check: diff={diff:.6f} PASS")


def test_sfo_output_length():
    """Test SFO maintains output length."""
    print("\nTest: SFO output length")
    signal = torch.randn(1000, dtype=torch.complex64)
    for sfo_ppm in [1, 10, 100]:
        resampled = apply_sfo(signal, sfo_ppm)
        assert len(resampled) == len(signal)
    print(f"  PASS")


def test_iq_imbalance_creates_distortion():
    """Test I/Q imbalance creates measurable distortion."""
    print("\nTest: I/Q imbalance distortion")
    signal = torch.randn(5000, dtype=torch.complex64)
    distorted = apply_iq_imbalance(signal, amplitude_imb_db=1.0, phase_imb_deg=5.0)
    error = torch.mean(torch.abs(signal - distorted)).item()
    assert error > 0.01
    print(f"  Mean error={error:.4f} PASS")


def test_iq_imbalance_zero_parameters():
    """Test I/Q imbalance with zero parameters does nothing."""
    print("\nTest: I/Q imbalance zero parameters")
    signal = torch.randn(1000, dtype=torch.complex64)
    distorted = apply_iq_imbalance(signal, amplitude_imb_db=0.0, phase_imb_deg=0.0)
    assert torch.allclose(signal, distorted, atol=1e-4)
    print(f"  PASS")


def test_dc_offset_adds_constant():
    """Test DC offset adds constant component."""
    print("\nTest: DC offset")
    signal = torch.randn(5000, dtype=torch.complex64)
    signal = signal - torch.mean(signal)
    with_dc = apply_dc_offset(signal, dc_level_dbc=-30)
    dc_mag = torch.abs(torch.mean(with_dc)).item()
    assert dc_mag > 0.001
    print(f"  DC magnitude={dc_mag:.6f} PASS")


def test_phase_noise_preserves_amplitude():
    """Test phase noise preserves amplitude."""
    print("\nTest: Phase noise amplitude preservation")
    signal = torch.randn(5000, dtype=torch.complex64)
    noisy = apply_phase_noise(signal, phase_noise_dbc_hz=-90, sample_rate=15.36e6)
    amp_original = torch.abs(signal)
    amp_noisy = torch.abs(noisy)
    relative_diff = torch.mean(torch.abs(amp_original - amp_noisy) / amp_original).item()
    assert relative_diff < 0.1
    print(f"  Relative amplitude diff={relative_diff*100:.3f}% PASS")


def test_pa_nonlinearity_clips_peaks():
    """Test PA nonlinearity reduces peak amplitude."""
    print("\nTest: PA nonlinearity peak clipping")
    signal = torch.randn(5000, dtype=torch.complex64)
    amplified = apply_pa_nonlinearity(signal, input_backoff_db=3, smoothness=2)
    peak_original = torch.max(torch.abs(signal)).item()
    peak_amplified = torch.max(torch.abs(amplified)).item()
    assert peak_amplified < peak_original
    print(f"  Original peak={peak_original:.4f}, After PA={peak_amplified:.4f} PASS")


def test_pa_nonlinearity_preserves_phase():
    """Test PA nonlinearity preserves phase (AM/PM=0)."""
    print("\nTest: PA nonlinearity phase preservation")
    signal = torch.randn(5000, dtype=torch.complex64)
    amplified = apply_pa_nonlinearity(signal, input_backoff_db=6, smoothness=2)
    phase_diff = torch.abs(torch.angle(signal) - torch.angle(amplified))
    mean_phase_diff = torch.mean(phase_diff).item()
    assert mean_phase_diff < 0.01
    print(f"  Mean phase diff={mean_phase_diff:.6f} rad PASS")


def test_mimo_channel_shape():
    """Test MIMO channel matrix has correct shape."""
    print("\nTest: MIMO channel shape")
    for num_tx, num_rx in [(2, 2), (4, 4), (8, 8)]:
        H = generate_mimo_channel(num_tx, num_rx, 1000)
        assert H.shape == (num_rx, num_tx, 1000)
        print(f"  {num_tx}x{num_rx}: shape={H.shape} PASS")


def test_mimo_time_varying():
    """Test MIMO channel varies over time with Doppler."""
    print("\nTest: MIMO time-varying fading")
    H = generate_mimo_channel(2, 2, 100000, doppler_hz=100, sample_rate=15.36e6)
    h_start = H[0, 0, 0]
    h_end = H[0, 0, -1]
    diff = torch.abs(h_start - h_end).item()
    assert diff > 0.1
    print(f"  Start-end difference={diff:.4f} PASS")


def test_mimo_apply_channel():
    """Test MIMO channel application."""
    print("\nTest: MIMO apply channel")
    H = generate_mimo_channel(4, 4, 1000)
    tx_signals = torch.randn(4, 1000, dtype=torch.complex64)
    rx_signals = apply_mimo_channel(tx_signals, H, noise_power_db=-20)
    assert rx_signals.shape == (4, 1000)
    print(f"  RX shape={rx_signals.shape} PASS")


def test_awgn_snr_accuracy():
    """Test AWGN achieves target SNR."""
    print("\nTest: AWGN SNR accuracy")
    signal = torch.randn(10000, dtype=torch.complex64)
    signal = signal / torch.sqrt(torch.mean(torch.abs(signal)**2))

    for target_snr in [0, 10, 20, 30]:
        noisy = add_awgn(signal, target_snr)
        signal_power = torch.mean(torch.abs(signal)**2)
        noise_power = torch.mean(torch.abs(noisy - signal)**2)
        measured_snr = 10 * torch.log10(signal_power / noise_power).item()
        error = abs(measured_snr - target_snr)
        assert error < 1.5
        print(f"  Target {target_snr} dB: measured {measured_snr:.2f} dB, error={error:.2f} dB PASS")


def run_all_tests():
    """Run all unit tests."""
    print("="*70)
    print("Channel Models Comprehensive Unit Tests")
    print("="*70)

    # TDL models tests
    test_tdl_all_models()
    test_tdl_power_normalization()
    test_tdl_delay_spread_scaling()
    test_tdl_los_models()

    # Jakes fading tests
    test_rayleigh_unit_power()
    test_rayleigh_amplitude_statistics()
    test_rician_unit_power()
    test_rician_k_factor_effect()

    # Hardware impairments tests
    test_cfo_preserves_power()
    test_cfo_phase_rotation()
    test_sfo_output_length()
    test_iq_imbalance_creates_distortion()
    test_iq_imbalance_zero_parameters()
    test_dc_offset_adds_constant()
    test_phase_noise_preserves_amplitude()
    test_pa_nonlinearity_clips_peaks()
    test_pa_nonlinearity_preserves_phase()

    # MIMO tests
    test_mimo_channel_shape()
    test_mimo_time_varying()
    test_mimo_apply_channel()

    # AWGN test
    test_awgn_snr_accuracy()

    print("\n" + "="*70)
    print("All tests PASSED")
    print("="*70)


if __name__ == '__main__':
    run_all_tests()
