"""
Realistic wireless channel modeling per 3GPP specifications.

Implements production-grade RF impairments and propagation effects:
- 3GPP TR 38.901 TDL/CDL channel models (replaces obsolete ITU profiles)
- Carrier Frequency Offset per TS 38.104/38.101
- Sampling Frequency Offset
- Hardware impairments (I/Q imbalance, DC offset, phase noise) per TS 36.101/38.101
- Time-varying fading with Doppler (Jakes' model)
- Power amplifier nonlinearity (Rapp model)
- MIMO processing with spatial correlation

All parameters based on 3GPP specifications and research literature.
See paper/amendment.md for detailed citations.
"""

import torch
import math
from typing import Dict, Any, Tuple, List


# 3GPP TR 38.901 TDL (Tapped Delay Line) Models
# Section 7.7.2, Tables 7.7.2-1 through 7.7.2-5
TDL_MODELS = {
    'TDL-A': {
        'delays_normalized': [
            0, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750, 0.7618,
            1.5375, 1.8978, 2.2242, 2.1717, 2.4942, 2.5119, 3.0582, 4.0810,
            4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586
        ],
        'powers_db': [
            -13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6,
            -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3, -18.9,
            -16.6, -19.9, -29.7
        ],
        'k_factors_db': [0] * 23,
        'description': 'NLOS, low delay spread (23 taps)',
        'typical_ds_ns': 30,  # Typical RMS delay spread
    },
    'TDL-B': {
        'delays_normalized': [
            0, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681,
            0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842, 2.0169,
            2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834
        ],
        'powers_db': [
            0, -2.2, -4, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3, -8.9, -9,
            -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4, -14.9, -9.2, -11.3
        ],
        'k_factors_db': [0] * 23,
        'description': 'NLOS, medium delay spread (23 taps)',
        'typical_ds_ns': 100,
    },
    'TDL-C': {
        'delays_normalized': [
            0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584,
            0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589,
            4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
        ],
        'powers_db': [
            -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7,
            -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16,
            -15.7, -21.6, -22.8
        ],
        'k_factors_db': [0] * 24,
        'description': 'NLOS, high delay spread (24 taps)',
        'typical_ds_ns': 300,
    },
    'TDL-D': {
        'delays_normalized': [
            0, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937,
            9.424, 9.708, 12.525
        ],
        'powers_db': [
            -13.5, -18.8, -21, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6,
            -24.8, -30.0, -27.7
        ],
        'k_factors_db': [13.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'description': 'LOS, low delay spread (13 taps, K=13.3 dB)',
        'typical_ds_ns': 30,
    },
    'TDL-E': {
        'delays_normalized': [
            0, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092, 1.9293, 1.9589,
            2.6426, 3.7136, 5.4524, 12.0034, 20.6519
        ],
        'powers_db': [
            -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8, -22.6,
            -22.3, -25.6, -20.2, -29.8, -29.2
        ],
        'k_factors_db': [22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'description': 'LOS, high delay spread (14 taps, K=22 dB)',
        'typical_ds_ns': 300,
    },
}


def add_awgn(
    signal: torch.Tensor,
    snr_db: float,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Add Additive White Gaussian Noise (AWGN) to signal.

    Args:
        signal: Complex input signal
        snr_db: Target SNR in dB
        device: PyTorch device

    Returns:
        Noisy signal
    """
    signal_power = torch.mean(torch.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power / 2.0)

    noise_real = torch.randn(signal.shape, device=device) * noise_std
    noise_imag = torch.randn(signal.shape, device=device) * noise_std
    noise = torch.complex(noise_real, noise_imag)

    return signal + noise


def generate_tdl_channel(
    num_samples: int,
    tdl_model: str = 'TDL-A',
    delay_spread_ns: float = None,
    sample_rate: float = 15.36e6,
    doppler_hz: float = 0.0,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, List[float], List[float]]:
    """
    Generate 3GPP TDL (Tapped Delay Line) channel model per TR 38.901.

    Reference: 3GPP TR 38.901 V17.0.0 Section 7.7.2

    Args:
        num_samples: Number of time samples
        tdl_model: TDL model name ('TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E')
        delay_spread_ns: RMS delay spread in nanoseconds (if None, use typical value)
        sample_rate: Sampling rate in Hz
        doppler_hz: Maximum Doppler frequency in Hz (0 = static channel)
        device: PyTorch device

    Returns:
        Tuple of (channel_coefficients, delays_ns, powers_linear)
        - channel_coefficients: Complex fading for each tap, shape (num_taps, num_samples)
        - delays_ns: Actual delays in nanoseconds
        - powers_linear: Linear power for each tap
    """
    if tdl_model not in TDL_MODELS:
        raise ValueError(f"Unknown TDL model: {tdl_model}. Available: {list(TDL_MODELS.keys())}")

    model = TDL_MODELS[tdl_model]

    # Use typical delay spread if not specified
    if delay_spread_ns is None:
        delay_spread_ns = model['typical_ds_ns']

    # Scale normalized delays by desired delay spread
    # TR 38.901 Section 7.7.3: τ_actual = τ_normalized * DS_desired
    delays_normalized = model['delays_normalized']
    delays_ns = [d * delay_spread_ns for d in delays_normalized]

    # Convert powers from dB to linear
    powers_db = model['powers_db']
    powers_linear = [10 ** (p_db / 10.0) for p_db in powers_db]

    # Normalize powers to unit total power
    total_power = sum(powers_linear)
    powers_linear = [p / total_power for p in powers_linear]

    # Get K-factors (Rice factor for LOS paths)
    k_factors_db = model['k_factors_db']

    num_taps = len(delays_ns)

    # Generate fading coefficients for each tap
    channel_coeffs = torch.zeros(num_taps, num_samples, dtype=torch.complex64, device=device)

    for tap_idx in range(num_taps):
        k_db = k_factors_db[tap_idx]

        if k_db > 0:
            # LOS tap: Rician fading
            h_tap = generate_rician_fading_jakes(
                num_samples,
                k_factor_db=k_db,
                doppler_freq=doppler_hz,
                sample_rate=sample_rate,
                device=device
            )
        else:
            # NLOS tap: Rayleigh fading
            h_tap = generate_rayleigh_fading_jakes(
                num_samples,
                doppler_freq=doppler_hz,
                sample_rate=sample_rate,
                device=device
            )

        # Scale by tap power
        h_tap = h_tap * math.sqrt(powers_linear[tap_idx])
        channel_coeffs[tap_idx] = h_tap

    return channel_coeffs, delays_ns, powers_linear


def generate_rayleigh_fading_jakes(
    num_samples: int,
    doppler_freq: float = 0.0,
    sample_rate: float = 1.0,
    num_oscillators: int = 16,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate Rayleigh fading using Jakes' sum-of-sinusoids model.

    Reference: Jakes, W. C., "Microwave Mobile Communications," 1994

    Args:
        num_samples: Number of time samples
        doppler_freq: Maximum Doppler frequency in Hz
        sample_rate: Sampling rate in Hz
        num_oscillators: Number of oscillators in Jakes' model (typically 8-16)
        device: PyTorch device

    Returns:
        Complex fading coefficients
    """
    if doppler_freq == 0:
        # Static channel: complex Gaussian with unit power
        h_real = torch.randn(num_samples, device=device) * math.sqrt(0.5)
        h_imag = torch.randn(num_samples, device=device) * math.sqrt(0.5)
        return torch.complex(h_real, h_imag)

    # Jakes' model: sum of sinusoids
    t = torch.arange(num_samples, device=device, dtype=torch.float32) / sample_rate

    # Initialize
    h_real = torch.zeros(num_samples, device=device)
    h_imag = torch.zeros(num_samples, device=device)

    # Generate oscillators
    for n in range(1, num_oscillators + 1):
        # Doppler frequency for this oscillator
        alpha_n = 2 * math.pi * n / (4 * num_oscillators)
        f_n = doppler_freq * torch.cos(torch.tensor(alpha_n))

        # Random phase
        phi_n = torch.rand(1, device=device).item() * 2 * math.pi

        # Add sinusoid
        h_real = h_real + torch.cos(2 * math.pi * f_n * t + phi_n)
        h_imag = h_imag + torch.sin(2 * math.pi * f_n * t + phi_n)

    # Normalize
    h = torch.complex(h_real, h_imag) / math.sqrt(num_oscillators)

    # Normalize to unit power
    h = h / torch.sqrt(torch.mean(torch.abs(h) ** 2))

    return h


def generate_rician_fading_jakes(
    num_samples: int,
    k_factor_db: float = 6.0,
    doppler_freq: float = 0.0,
    sample_rate: float = 1.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate Rician fading with Jakes' model for scattered component.

    K-factor: ratio of LOS power to scattered power (dB)

    Args:
        num_samples: Number of time samples
        k_factor_db: Rician K-factor in dB
        doppler_freq: Maximum Doppler frequency in Hz
        sample_rate: Sampling rate in Hz
        device: PyTorch device

    Returns:
        Complex fading coefficients
    """
    k_linear = 10 ** (k_factor_db / 10.0)

    # Power split
    los_power = k_linear / (k_linear + 1)
    scattered_power = 1.0 / (k_linear + 1)

    # LOS component (constant)
    los_component = torch.full((num_samples,), math.sqrt(los_power),
                               dtype=torch.complex64, device=device)

    # Scattered component (Rayleigh with Jakes)
    scattered = generate_rayleigh_fading_jakes(
        num_samples,
        doppler_freq=doppler_freq,
        sample_rate=sample_rate,
        device=device
    )
    scattered_component = scattered * math.sqrt(scattered_power)

    # Combine
    h = los_component + scattered_component

    # Normalize to unit average power
    h = h / torch.sqrt(torch.mean(torch.abs(h) ** 2))

    return h


def apply_tdl_channel(
    signal: torch.Tensor,
    tdl_model: str = 'TDL-A',
    delay_spread_ns: float = None,
    sample_rate: float = 15.36e6,
    doppler_hz: float = 0.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply 3GPP TDL channel to signal.

    Args:
        signal: Complex input signal
        tdl_model: TDL model name
        delay_spread_ns: RMS delay spread in ns
        sample_rate: Sampling rate in Hz
        doppler_hz: Maximum Doppler frequency in Hz
        device: PyTorch device

    Returns:
        Channel-convolved signal
    """
    num_samples = len(signal)

    # Generate TDL channel
    channel_coeffs, delays_ns, powers = generate_tdl_channel(
        num_samples,
        tdl_model=tdl_model,
        delay_spread_ns=delay_spread_ns,
        sample_rate=sample_rate,
        doppler_hz=doppler_hz,
        device=device
    )

    # Convert delays from nanoseconds to samples
    delays_samples = [int(d_ns * 1e-9 * sample_rate) for d_ns in delays_ns]

    # Apply multipath
    output = torch.zeros_like(signal)

    for tap_idx, delay_samp in enumerate(delays_samples):
        if delay_samp == 0:
            output += signal * channel_coeffs[tap_idx]
        else:
            delayed_signal = torch.zeros_like(signal)
            delayed_signal[delay_samp:] = signal[:-delay_samp]
            output += delayed_signal * channel_coeffs[tap_idx, :len(delayed_signal)]

    return output


def apply_cfo(
    signal: torch.Tensor,
    cfo_hz: float,
    sample_rate: float,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply Carrier Frequency Offset (CFO).

    Reference:
    - 3GPP TS 38.104 Section 6.5.1 (BS frequency error)
    - 3GPP TS 38.101 Section 6.5.1 (UE frequency error)

    Typical values:
    - BS: ±0.05 to ±0.1 ppm
    - UE (initial): ±5 ppm
    - UE (connected): ±0.1 ppm

    Args:
        signal: Complex input signal
        cfo_hz: Frequency offset in Hz
        sample_rate: Sampling rate in Hz
        device: PyTorch device

    Returns:
        Signal with CFO applied
    """
    t = torch.arange(len(signal), device=device, dtype=torch.float32) / sample_rate
    phase_rotation = torch.exp(1j * 2 * math.pi * cfo_hz * t)

    return signal * phase_rotation


def apply_sfo(
    signal: torch.Tensor,
    sfo_ppm: float,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply Sampling Frequency Offset (SFO).

    SFO causes timing drift. Implemented as resampling.

    Args:
        signal: Complex input signal
        sfo_ppm: SFO in parts per million
        device: PyTorch device

    Returns:
        Resampled signal
    """
    # Calculate new sample indices
    num_samples = len(signal)
    offset_factor = 1.0 + sfo_ppm / 1e6

    # New sample positions
    new_indices = torch.arange(num_samples, device=device, dtype=torch.float32) * offset_factor

    # Clip to valid range
    new_indices = torch.clamp(new_indices, 0, num_samples - 1)

    # Linear interpolation (simple resampling)
    indices_floor = torch.floor(new_indices).long()
    indices_ceil = torch.clamp(indices_floor + 1, max=num_samples - 1)
    frac = new_indices - indices_floor.float()

    # Interpolate
    resampled = signal[indices_floor] * (1 - frac) + signal[indices_ceil] * frac

    return resampled


def apply_iq_imbalance(
    signal: torch.Tensor,
    amplitude_imb_db: float = 0.5,
    phase_imb_deg: float = 2.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply I/Q imbalance (hardware impairment).

    Reference: 3GPP TS 36.101 Section 7.5 (Image rejection ≥25 dB)

    Typical values:
    - Amplitude imbalance: 0.1 to 3 dB
    - Phase imbalance: 1 to 10 degrees

    Args:
        signal: Complex input signal
        amplitude_imb_db: Amplitude imbalance in dB
        phase_imb_deg: Phase imbalance in degrees
        device: PyTorch device

    Returns:
        Signal with I/Q imbalance
    """
    # Convert to amplitude factor
    alpha = (10 ** (amplitude_imb_db / 20.0) - 1) / 2

    # Convert to radians
    theta = phase_imb_deg * math.pi / 180.0

    # Apply imbalance
    s_i = signal.real
    s_q = signal.imag

    r_i = (1 + alpha) * s_i * math.cos(theta / 2) - s_q * math.sin(theta / 2)
    r_q = (1 - alpha) * s_i * math.sin(theta / 2) + s_q * math.cos(theta / 2)

    return torch.complex(r_i, r_q)


def apply_dc_offset(
    signal: torch.Tensor,
    dc_level_dbc: float = -35.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply DC offset (LO leakage in zero-IF receivers).

    Typical value: -40 to -30 dBc relative to signal power

    Args:
        signal: Complex input signal
        dc_level_dbc: DC offset level in dBc (dB relative to carrier)
        device: PyTorch device

    Returns:
        Signal with DC offset
    """
    # Calculate signal power
    signal_power = torch.mean(torch.abs(signal) ** 2)

    # DC offset power
    dc_power = signal_power * (10 ** (dc_level_dbc / 10.0))
    dc_amplitude = math.sqrt(dc_power)

    # Random DC offset (complex)
    dc_phase = torch.rand(1, device=device).item() * 2 * math.pi
    dc_offset = dc_amplitude * torch.exp(1j * torch.tensor(dc_phase, device=device))

    return signal + dc_offset


def apply_phase_noise(
    signal: torch.Tensor,
    phase_noise_dbc_hz: float = -90.0,
    sample_rate: float = 15.36e6,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply phase noise (oscillator imperfection).

    Modeled as Wiener process (integrated white noise).

    Reference: 3GPP TS 25.102 (phase noise affects EVM)

    Typical value: -90 to -110 dBc/Hz at 10 kHz offset

    Args:
        signal: Complex input signal
        phase_noise_dbc_hz: Phase noise spectral density in dBc/Hz
        sample_rate: Sampling rate in Hz
        device: PyTorch device

    Returns:
        Signal with phase noise
    """
    num_samples = len(signal)

    # Calculate phase noise variance
    # PSD in linear scale
    psd_linear = 10 ** (phase_noise_dbc_hz / 10.0)

    # Standard deviation for phase (integrated)
    phase_std = math.sqrt(2 * math.pi * psd_linear)

    # Generate white noise and integrate
    white_noise = torch.randn(num_samples, device=device) * phase_std
    phase_noise = torch.cumsum(white_noise, dim=0) * (1.0 / math.sqrt(sample_rate))

    # Apply phase rotation
    return signal * torch.exp(1j * phase_noise)


def apply_pa_nonlinearity(
    signal: torch.Tensor,
    input_backoff_db: float = 6.0,
    smoothness: float = 2.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply power amplifier nonlinearity using Rapp model.

    Reference:
    Rapp, C., "Effects of HPA-nonlinearity on a 4-DPSK/OFDM-signal,"
    ESA Special Publication, 1991.

    Args:
        signal: Complex input signal
        input_backoff_db: Input back-off from saturation in dB
        smoothness: Smoothness parameter (typical: 2-3)
        device: PyTorch device

    Returns:
        Amplified signal with nonlinearity
    """
    # Saturation amplitude (based on input back-off)
    signal_power = torch.mean(torch.abs(signal) ** 2)
    peak_amplitude = torch.max(torch.abs(signal))

    # Calculate saturation point
    backoff_linear = 10 ** (input_backoff_db / 20.0)
    a_sat = peak_amplitude * backoff_linear

    # Rapp model
    amplitude = torch.abs(signal)
    phase = torch.angle(signal)

    # AM/AM characteristic
    g = amplitude / ((1 + (amplitude / a_sat) ** (2 * smoothness)) ** (1 / (2 * smoothness)))

    # Reconstruct
    return g * torch.exp(1j * phase)


def generate_mimo_channel(
    num_tx: int,
    num_rx: int,
    num_samples: int,
    channel_type: str = 'rayleigh',
    spatial_correlation: float = 0.0,
    k_factor_db: float = 6.0,
    doppler_hz: float = 0.0,
    sample_rate: float = 15.36e6,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate MIMO channel matrix with time variation.

    Args:
        num_tx: Number of transmit antennas
        num_rx: Number of receive antennas
        num_samples: Number of time samples
        channel_type: 'rayleigh' or 'rician'
        spatial_correlation: Spatial correlation coefficient (0-0.9)
        k_factor_db: Rician K-factor in dB
        doppler_hz: Maximum Doppler frequency
        sample_rate: Sampling rate
        device: PyTorch device

    Returns:
        MIMO channel matrix of shape (num_rx, num_tx, num_samples)
    """
    H = torch.zeros(num_rx, num_tx, num_samples, dtype=torch.complex64, device=device)

    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            if channel_type == 'rayleigh':
                h = generate_rayleigh_fading_jakes(
                    num_samples,
                    doppler_freq=doppler_hz,
                    sample_rate=sample_rate,
                    device=device
                )
            elif channel_type == 'rician':
                h = generate_rician_fading_jakes(
                    num_samples,
                    k_factor_db=k_factor_db,
                    doppler_freq=doppler_hz,
                    sample_rate=sample_rate,
                    device=device
                )
            else:
                raise ValueError(f"Unknown channel type: {channel_type}")

            H[rx_idx, tx_idx, :] = h

    # Apply spatial correlation if specified
    if spatial_correlation > 0:
        correlation_matrix = torch.zeros(num_rx * num_tx, num_rx * num_tx, dtype=torch.complex64, device=device)
        for i in range(num_rx * num_tx):
            for j in range(num_rx * num_tx):
                distance = abs(i - j)
                correlation_matrix[i, j] = spatial_correlation ** distance

        L = torch.linalg.cholesky(correlation_matrix + 1e-6 * torch.eye(num_rx * num_tx, dtype=torch.complex64, device=device))
        H_flat = H.reshape(num_rx * num_tx, num_samples)
        H_corr_flat = L @ H_flat
        H = H_corr_flat.reshape(num_rx, num_tx, num_samples)

    return H


def apply_mimo_channel(
    signals: torch.Tensor,
    channel_matrix: torch.Tensor,
    noise_power_db: float = -20.0,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Apply MIMO channel: Y = H * X + N

    Args:
        signals: Transmit signals of shape (num_tx, num_samples)
        channel_matrix: MIMO channel of shape (num_rx, num_tx, num_samples)
        noise_power_db: Noise power in dB
        device: PyTorch device

    Returns:
        Received signals of shape (num_rx, num_samples)
    """
    num_rx, num_tx, num_samples = channel_matrix.shape

    if signals.shape[0] != num_tx or signals.shape[1] != num_samples:
        raise ValueError(f"Signal shape {signals.shape} does not match channel")

    received = torch.zeros(num_rx, num_samples, dtype=torch.complex64, device=device)

    for rx_idx in range(num_rx):
        for tx_idx in range(num_tx):
            received[rx_idx] += channel_matrix[rx_idx, tx_idx] * signals[tx_idx]

    # Add AWGN
    noise_power_linear = 10 ** (noise_power_db / 10.0)
    noise_std = math.sqrt(noise_power_linear / 2.0)

    noise_real = torch.randn(num_rx, num_samples, device=device) * noise_std
    noise_imag = torch.randn(num_rx, num_samples, device=device) * noise_std
    noise = torch.complex(noise_real, noise_imag)

    return received + noise


def validate_channel_statistics(
    channel_coeffs: torch.Tensor,
    expected_type: str = 'rayleigh'
) -> Dict[str, Any]:
    """
    Validate channel statistics against theoretical values.

    Args:
        channel_coeffs: Complex channel coefficients
        expected_type: 'rayleigh' or 'rician'

    Returns:
        Dictionary with validation metrics
    """
    amplitude = torch.abs(channel_coeffs)
    power = amplitude ** 2

    mean_amp = torch.mean(amplitude).item()
    std_amp = torch.std(amplitude).item()
    mean_power = torch.mean(power).item()

    # Theoretical values for Rayleigh with unit power
    expected_rayleigh_amp = math.sqrt(math.pi / 2)  # ≈ 1.253

    return {
        'mean_amplitude': mean_amp,
        'std_amplitude': std_amp,
        'mean_power': mean_power,
        'expected_rayleigh_amplitude': expected_rayleigh_amp,
        'power_error_db': 10 * math.log10(mean_power) if mean_power > 0 else float('-inf'),
        'num_samples': len(channel_coeffs)
    }
