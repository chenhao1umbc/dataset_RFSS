"""
Shared utility functions for RF signal processing in PyTorch.

This module provides fundamental signal processing operations:
- Power normalization
- AWGN noise generation
- Carrier frequency modulation
- Time vector generation
- SINR calculation
"""

import torch
import math


def generate_time_vector(num_samples: int, sample_rate: float, device: str = 'cpu') -> torch.Tensor:
    """
    Generate time vector for signal generation.

    Args:
        num_samples: Number of time samples
        sample_rate: Sampling rate in Hz
        device: PyTorch device ('cpu', 'cuda', 'mps')

    Returns:
        Time vector of shape (num_samples,)
    """
    return torch.arange(num_samples, dtype=torch.float32, device=device) / sample_rate


def normalize_power(signal: torch.Tensor, target_power_db: float = 0.0) -> torch.Tensor:
    """
    Normalize signal to target power level.

    Args:
        signal: Input signal (complex or real)
        target_power_db: Target power in dB (default: 0 dB = unit power)

    Returns:
        Power-normalized signal
    """
    # Calculate current power
    current_power = torch.mean(torch.abs(signal) ** 2)

    # Avoid division by zero
    if current_power < 1e-12:
        return signal

    # Calculate scaling factor
    target_power_linear = 10 ** (target_power_db / 10.0)
    scale = torch.sqrt(target_power_linear / current_power)

    return signal * scale


def add_awgn_noise(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add additive white Gaussian noise to signal.

    Args:
        signal: Input signal (complex or real)
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy signal with same shape as input
    """
    # Calculate signal power
    signal_power = torch.mean(torch.abs(signal) ** 2)

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)

    # Generate noise based on signal type
    if torch.is_complex(signal):
        # Complex noise: real and imaginary parts are independent Gaussian
        noise_real = torch.randn_like(signal.real) * noise_std / math.sqrt(2)
        noise_imag = torch.randn_like(signal.imag) * noise_std / math.sqrt(2)
        noise = torch.complex(noise_real, noise_imag)
    else:
        # Real noise
        noise = torch.randn_like(signal) * noise_std

    return signal + noise


def add_carrier_frequency(baseband_signal: torch.Tensor,
                          carrier_freq: float,
                          sample_rate: float) -> torch.Tensor:
    """
    Modulate baseband signal to carrier frequency.

    Args:
        baseband_signal: Complex baseband signal
        carrier_freq: Carrier frequency in Hz
        sample_rate: Sampling rate in Hz

    Returns:
        Complex signal modulated to carrier frequency
    """
    num_samples = len(baseband_signal)
    device = baseband_signal.device

    # Generate time vector
    t = generate_time_vector(num_samples, sample_rate, device)

    # Generate carrier: e^(j*2*pi*fc*t)
    carrier = torch.exp(1j * 2 * math.pi * carrier_freq * t)

    # Modulate
    return baseband_signal * carrier


def calculate_sinr(estimated_signal: torch.Tensor,
                   reference_signal: torch.Tensor) -> float:
    """
    Calculate Signal-to-Interference-plus-Noise Ratio.

    SINR = 10 * log10(P_signal / P_error)

    Args:
        estimated_signal: Estimated/separated signal
        reference_signal: Ground truth reference signal

    Returns:
        SINR in dB
    """
    # Align signals if needed (simple energy-based alignment)
    # This handles potential scaling differences
    estimated_signal = estimated_signal.flatten()
    reference_signal = reference_signal.flatten()

    # Ensure same length
    min_len = min(len(estimated_signal), len(reference_signal))
    estimated_signal = estimated_signal[:min_len]
    reference_signal = reference_signal[:min_len]

    # Calculate optimal scaling factor
    numerator = torch.sum(torch.conj(reference_signal) * estimated_signal)
    denominator = torch.sum(torch.abs(reference_signal) ** 2)

    if denominator < 1e-12:
        return float('-inf')

    alpha = numerator / denominator

    # Scale estimated signal
    scaled_estimated = alpha * estimated_signal

    # Calculate error
    error = reference_signal - scaled_estimated

    # Calculate powers
    signal_power = torch.mean(torch.abs(reference_signal) ** 2).item()
    error_power = torch.mean(torch.abs(error) ** 2).item()

    # Avoid division by zero
    if error_power < 1e-12:
        return float('inf')

    # SINR in dB
    sinr_db = 10 * math.log10(signal_power / error_power)

    return sinr_db


def calculate_papr(signal: torch.Tensor) -> float:
    """
    Calculate Peak-to-Average Power Ratio.

    PAPR = 10 * log10(P_peak / P_avg)

    Args:
        signal: Input signal (complex or real)

    Returns:
        PAPR in dB
    """
    # Calculate instantaneous power
    power = torch.abs(signal) ** 2

    # Peak and average power
    peak_power = torch.max(power).item()
    avg_power = torch.mean(power).item()

    if avg_power < 1e-12:
        return float('inf')

    papr_db = 10 * math.log10(peak_power / avg_power)

    return papr_db
