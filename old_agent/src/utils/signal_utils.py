"""
Shared signal processing utilities
Consolidates duplicate functions across the codebase
"""
import numpy as np


def normalize_power(signal, target_power=1.0):
    """
    Normalize signal to target power level
    
    Args:
        signal: Input signal (complex or real)
        target_power: Target average power (default: 1.0)
        
    Returns:
        normalized_signal: Power-normalized signal
    """
    current_power = np.mean(np.abs(signal)**2)
    if current_power == 0:
        return signal
    
    scale_factor = np.sqrt(target_power / current_power)
    return signal * scale_factor


def normalize_power_db(signal, target_power_db=0.0):
    """
    Normalize signal to target power level in dB
    
    Args:
        signal: Input signal (complex or real)  
        target_power_db: Target power in dB (default: 0 dB)
        
    Returns:
        normalized_signal: Power-normalized signal
    """
    target_power_linear = 10**(target_power_db / 10)
    return normalize_power(signal, target_power_linear)


def add_awgn_noise(signal, snr_db):
    """
    Add Additive White Gaussian Noise to signal
    
    Args:
        signal: Input signal (complex or real)
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        noisy_signal: Signal with added AWGN
    """
    signal_power = np.mean(np.abs(signal)**2)
    noise_power = signal_power / (10**(snr_db/10))
    
    if np.iscomplexobj(signal):
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 
                                        1j * np.random.randn(len(signal)))
    else:
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    
    return signal + noise


def add_carrier_frequency(baseband_signal, carrier_freq, sample_rate):
    """
    Add carrier frequency to baseband signal
    
    Args:
        baseband_signal: Complex baseband signal
        carrier_freq: Carrier frequency in Hz
        sample_rate: Sample rate in Hz
        
    Returns:
        rf_signal: Signal with carrier frequency applied
    """
    t = np.arange(len(baseband_signal)) / sample_rate
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    return baseband_signal * carrier


def calculate_signal_power(signal):
    """
    Calculate signal power in linear and dB scales
    
    Args:
        signal: Input signal
        
    Returns:
        dict: Power metrics {'linear': float, 'db': float}
    """
    power_linear = np.mean(np.abs(signal)**2)
    power_db = 10 * np.log10(power_linear) if power_linear > 0 else -np.inf
    
    return {'linear': power_linear, 'db': power_db}


def calculate_papr(signal):
    """
    Calculate Peak-to-Average Power Ratio
    
    Args:
        signal: Input signal
        
    Returns:
        papr_db: PAPR in dB
    """
    peak_power = np.max(np.abs(signal)**2)
    avg_power = np.mean(np.abs(signal)**2)
    
    if avg_power == 0:
        return np.inf
        
    papr_db = 10 * np.log10(peak_power / avg_power)
    return papr_db


def resample_signal(signal, original_rate, target_rate):
    """
    Resample signal to new sample rate using linear interpolation
    
    Args:
        signal: Input signal
        original_rate: Original sample rate
        target_rate: Target sample rate
        
    Returns:
        resampled_signal: Signal at new sample rate
    """
    if original_rate == target_rate:
        return signal
    
    # Calculate new length
    duration = len(signal) / original_rate
    new_length = int(duration * target_rate)
    
    # Create time vectors
    t_original = np.arange(len(signal)) / original_rate
    t_new = np.arange(new_length) / target_rate
    
    # Interpolate real and imaginary parts
    if np.iscomplexobj(signal):
        real_part = np.interp(t_new, t_original, signal.real)
        imag_part = np.interp(t_new, t_original, signal.imag)
        resampled_signal = real_part + 1j * imag_part
    else:
        resampled_signal = np.interp(t_new, t_original, signal)
    
    return resampled_signal


def align_signals_length(signals, method='truncate'):
    """
    Align multiple signals to same length
    
    Args:
        signals: List of signals to align
        method: 'truncate', 'pad_zeros', or 'repeat'
        
    Returns:
        aligned_signals: List of signals with same length
    """
    if not signals:
        return signals
    
    lengths = [len(sig) for sig in signals]
    
    if method == 'truncate':
        target_length = min(lengths)
        return [sig[:target_length] for sig in signals]
        
    elif method == 'pad_zeros':
        target_length = max(lengths)
        aligned = []
        for sig in signals:
            if len(sig) < target_length:
                padding = target_length - len(sig)
                if np.iscomplexobj(sig):
                    padded = np.concatenate([sig, np.zeros(padding, dtype=complex)])
                else:
                    padded = np.concatenate([sig, np.zeros(padding)])
                aligned.append(padded)
            else:
                aligned.append(sig)
        return aligned
        
    elif method == 'repeat':
        target_length = max(lengths)
        aligned = []
        for sig in signals:
            if len(sig) < target_length:
                repeats = int(np.ceil(target_length / len(sig)))
                repeated = np.tile(sig, repeats)[:target_length]
                aligned.append(repeated)
            else:
                aligned.append(sig[:target_length])
        return aligned
        
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def frequency_shift(signal, freq_offset, sample_rate):
    """
    Apply frequency shift to signal
    
    Args:
        signal: Input signal
        freq_offset: Frequency offset in Hz (positive = upshift)
        sample_rate: Sample rate in Hz
        
    Returns:
        shifted_signal: Frequency-shifted signal
    """
    t = np.arange(len(signal)) / sample_rate
    shift_factor = np.exp(1j * 2 * np.pi * freq_offset * t)
    return signal * shift_factor


def generate_time_vector(num_samples, sample_rate):
    """
    Generate time vector for given parameters
    
    Args:
        num_samples: Number of samples
        sample_rate: Sample rate in Hz
        
    Returns:
        time_vector: Time vector in seconds
    """
    return np.arange(num_samples) / sample_rate


def db_to_linear(db_value):
    """Convert dB to linear scale"""
    return 10**(db_value / 10)


def linear_to_db(linear_value):
    """Convert linear to dB scale"""
    if linear_value <= 0:
        return -np.inf
    return 10 * np.log10(linear_value)


def rms_value(signal):
    """Calculate RMS value of signal"""
    return np.sqrt(np.mean(np.abs(signal)**2))


def peak_value(signal):
    """Calculate peak value of signal"""
    return np.max(np.abs(signal))


if __name__ == "__main__":
    # Test signal utilities
    print("Testing signal utilities...")
    
    # Generate test signal
    fs = 1000  # Sample rate
    t = np.arange(1000) / fs
    test_signal = np.sin(2*np.pi*10*t) + 0.5j*np.cos(2*np.pi*15*t)
    
    # Test power normalization
    normalized = normalize_power(test_signal, 2.0)
    power_metrics = calculate_signal_power(normalized)
    print(f"Original power: {calculate_signal_power(test_signal)['linear']:.6f}")
    print(f"Normalized power: {power_metrics['linear']:.6f} ({power_metrics['db']:.2f} dB)")
    
    # Test noise addition
    noisy = add_awgn_noise(test_signal, 10)  # 10 dB SNR
    print(f"Noisy signal power: {calculate_signal_power(noisy)['linear']:.6f}")
    
    # Test PAPR
    papr = calculate_papr(test_signal)
    print(f"PAPR: {papr:.2f} dB")
    
    # Test frequency shift
    shifted = frequency_shift(test_signal, 100, fs)  # 100 Hz shift
    print(f"Frequency shifted signal length: {len(shifted)}")
    
    # Test signal alignment
    sig1 = np.random.randn(100)
    sig2 = np.random.randn(80) 
    sig3 = np.random.randn(120)
    aligned = align_signals_length([sig1, sig2, sig3], method='pad_zeros')
    print(f"Aligned lengths: {[len(sig) for sig in aligned]}")
    
    print("All tests completed successfully!")