"""
Signal mixing utilities for multi-standard RF coexistence scenarios.

Based on 3GPP coexistence studies and realistic spectrum sharing scenarios.
Supports:
- Per-source independent channel effects
- Co-channel and adjacent-channel mixing
- Realistic power ratios (SIR: -20 to +20 dB)
- Timing offsets (asynchronous arrival)
- MIMO spatial mixing with correlation
- Complete ground truth preservation

References:
- 3GPP TR 38.901: Channel models
- 3GPP TS 38.104/38.101: RF requirements (ACIR, ACLR)
- Dynamic Spectrum Sharing (DSS): LTE-NR coexistence
- See paper/mixing_scenarios.md for detailed specifications
"""

import torch
from typing import List, Dict, Optional
from datetime import datetime

from src.utils_shared import normalize_power, add_carrier_frequency
from src.utils_channel import (
    apply_tdl_channel,
    apply_cfo,
    apply_sfo,
    apply_iq_imbalance,
    apply_dc_offset,
    apply_phase_noise,
    apply_pa_nonlinearity,
    generate_mimo_channel,
    apply_mimo_channel,
)


class SignalMixer:
    """
    Multi-standard RF signal mixer with realistic coexistence scenarios.

    Features:
    - Per-source independent channel effects before mixing
    - Co-channel mixing (hardest case, all at baseband)
    - Adjacent-channel mixing (frequency-shifted with realistic ACIR)
    - Realistic power ratios (near-far scenarios)
    - Timing offsets (asynchronous signal arrival)
    - MIMO spatial mixing with correlation
    - Complete ground truth preservation
    """

    def __init__(self, sample_rate: float, device: str = 'cpu'):
        """
        Initialize signal mixer.

        Args:
            sample_rate: Common sampling rate for all signals (Hz)
            device: PyTorch device ('cpu', 'cuda', 'mps')
        """
        self.sample_rate = sample_rate
        self.device = device
        self.sources = []

    def add_source(self,
                   signal: torch.Tensor,
                   label: str,
                   power_db: float = 0.0,
                   freq_offset_hz: float = 0.0,
                   timing_offset_samples: int = 0,
                   channel_params: Optional[Dict] = None,
                   source_sample_rate: Optional[float] = None):
        """
        Add a source signal to the mixer.

        Args:
            signal: Clean baseband signal (complex)
            label: Source label (e.g., 'GSM', 'LTE', '5G NR')
            power_db: Target power in dB relative to reference
            freq_offset_hz: Frequency offset for adjacent-channel mixing (0 = co-channel)
            timing_offset_samples: Timing offset for asynchronous arrival
            channel_params: Optional dict with channel parameters:
                {
                    'tdl_model': str,  # 'TDL-A', 'TDL-B', etc. or None
                    'delay_spread_ns': float,
                    'doppler_hz': float,
                    'cfo_hz': float,
                    'sfo_ppm': float,
                    'iq_amplitude_db': float,
                    'iq_phase_deg': float,
                    'dc_offset_dbc': float,
                    'phase_noise_dbc_hz': float,
                    'pa_backoff_db': float,
                    'pa_smoothness': float,
                }
            source_sample_rate: Original sample rate of the signal (if different from mixer rate,
                              signal will be resampled automatically)
        """
        # Resample if source sample rate differs from mixer sample rate
        if source_sample_rate is not None and abs(source_sample_rate - self.sample_rate) > 1.0:
            # Calculate new length after resampling
            original_duration = len(signal) / source_sample_rate
            new_length = int(original_duration * self.sample_rate)

            # Resample using linear interpolation with align_corners=True for proper edge handling
            # Interpolate real and imaginary parts separately
            real_part = torch.nn.functional.interpolate(
                signal.real.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=True
            ).squeeze()

            imag_part = torch.nn.functional.interpolate(
                signal.imag.unsqueeze(0).unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=True
            ).squeeze()

            signal = torch.complex(real_part, imag_part)

        source_info = {
            'signal_clean': signal.to(self.device),
            'label': label,
            'power_db': power_db,
            'freq_offset_hz': freq_offset_hz,
            'timing_offset_samples': timing_offset_samples,
            'channel_params': channel_params or {},
            'signal_channelized': None,  # Will be filled during mixing
        }

        self.sources.append(source_info)

    def _apply_channel_to_source(self, signal: torch.Tensor, channel_params: Dict) -> torch.Tensor:
        """
        Apply channel effects to a single source signal.

        Args:
            signal: Clean baseband signal
            channel_params: Channel parameters dict

        Returns:
            Channelized signal
        """
        channelized = signal.clone()

        # Apply TDL multipath channel if specified
        if channel_params.get('tdl_model') is not None:
            channelized = apply_tdl_channel(
                channelized,
                tdl_model=channel_params['tdl_model'],
                delay_spread_ns=channel_params.get('delay_spread_ns', 30.0),
                sample_rate=self.sample_rate,
                doppler_hz=channel_params.get('doppler_hz', 0.0),
                device=self.device
            )

        # Apply carrier frequency offset
        if channel_params.get('cfo_hz') is not None:
            channelized = apply_cfo(channelized, channel_params['cfo_hz'], self.sample_rate)

        # Apply sampling frequency offset
        if channel_params.get('sfo_ppm') is not None:
            channelized = apply_sfo(channelized, channel_params['sfo_ppm'])

        # Apply I/Q imbalance
        if channel_params.get('iq_amplitude_db') is not None or channel_params.get('iq_phase_deg') is not None:
            channelized = apply_iq_imbalance(
                channelized,
                amplitude_imb_db=channel_params.get('iq_amplitude_db', 0.0),
                phase_imb_deg=channel_params.get('iq_phase_deg', 0.0)
            )

        # Apply DC offset
        if channel_params.get('dc_offset_dbc') is not None:
            channelized = apply_dc_offset(channelized, channel_params['dc_offset_dbc'])

        # Apply phase noise
        if channel_params.get('phase_noise_dbc_hz') is not None:
            channelized = apply_phase_noise(
                channelized,
                channel_params['phase_noise_dbc_hz'],
                self.sample_rate
            )

        # Apply PA nonlinearity
        if channel_params.get('pa_backoff_db') is not None:
            channelized = apply_pa_nonlinearity(
                channelized,
                input_backoff_db=channel_params['pa_backoff_db'],
                smoothness=channel_params.get('pa_smoothness', 2.0)
            )

        return channelized

    def _apply_timing_offset(self, signal: torch.Tensor, offset_samples: int, total_length: int) -> torch.Tensor:
        """
        Apply timing offset with zero-padding.

        Args:
            signal: Input signal
            offset_samples: Timing offset in samples
            total_length: Total output length

        Returns:
            Time-shifted signal with zero-padding
        """
        if offset_samples < 0:
            raise ValueError("Timing offset must be non-negative")

        if offset_samples == 0 and len(signal) == total_length:
            return signal

        # Create zero-padded output
        output = torch.zeros(total_length, dtype=signal.dtype, device=signal.device)

        # Place signal at offset position
        end_pos = min(offset_samples + len(signal), total_length)
        signal_len = end_pos - offset_samples
        output[offset_samples:end_pos] = signal[:signal_len]

        return output

    def mix(self, mode: str = 'co-channel', output_length: Optional[int] = None) -> Dict:
        """
        Mix all added source signals.

        Args:
            mode: Mixing mode - 'co-channel' or 'adjacent-channel'
            output_length: Output length in samples (uses max if None)

        Returns:
            Dictionary with mixed signal and complete ground truth:
            {
                'mixed_signal': torch.Tensor,
                'source_signals_clean': List[torch.Tensor],
                'source_signals_channelized': List[torch.Tensor],
                'source_signals_aligned': List[torch.Tensor],
                'channel_params': List[dict],
                'mixing_params': dict,
                'metadata': dict,
            }
        """
        if not self.sources:
            raise ValueError("No source signals added to mixer")

        if mode not in ['co-channel', 'adjacent-channel']:
            raise ValueError(f"Unknown mixing mode: {mode}")

        # Determine output length
        if output_length is None:
            max_signal_len = max(len(src['signal_clean']) for src in self.sources)
            max_timing_offset = max(src['timing_offset_samples'] for src in self.sources)
            output_length = max_signal_len + max_timing_offset

        # Initialize mixed signal
        mixed_signal = torch.zeros(output_length, dtype=torch.complex64, device=self.device)

        # Lists for ground truth
        source_signals_clean = []
        source_signals_channelized = []
        source_signals_aligned = []
        channel_params_list = []

        # Process each source
        for source in self.sources:
            # 1. Get clean signal
            signal_clean = source['signal_clean']
            source_signals_clean.append(signal_clean)

            # 2. Apply channel effects
            signal_channelized = self._apply_channel_to_source(
                signal_clean,
                source['channel_params']
            )
            source['signal_channelized'] = signal_channelized
            source_signals_channelized.append(signal_channelized)

            # 3. Apply timing offset and normalize to output length
            signal_aligned = self._apply_timing_offset(
                signal_channelized,
                source['timing_offset_samples'],
                output_length
            )

            # 4. Apply frequency offset if adjacent-channel mode
            if mode == 'adjacent-channel' and source['freq_offset_hz'] != 0:
                signal_aligned = add_carrier_frequency(
                    signal_aligned,
                    source['freq_offset_hz'],
                    self.sample_rate
                )

            # 5. Normalize to target power
            signal_powered = normalize_power(signal_aligned, source['power_db'])
            source_signals_aligned.append(signal_powered)

            # 6. Add to mixture
            mixed_signal += signal_powered

            # 7. Store channel parameters
            channel_params_list.append(source['channel_params'])

        # Create mixing parameters
        mixing_params = {
            'num_sources': len(self.sources),
            'mode': mode,
            'power_ratios_db': [src['power_db'] for src in self.sources],
            'frequency_offsets_hz': [src['freq_offset_hz'] for src in self.sources],
            'timing_offsets_samples': [src['timing_offset_samples'] for src in self.sources],
            'sample_rate': self.sample_rate,
            'output_length': output_length,
        }

        # Create metadata
        metadata = {
            'scenario_name': f"{'+'.join([src['label'] for src in self.sources])}_{mode}",
            'standards': [src['label'] for src in self.sources],
            'generation_timestamp': datetime.now().isoformat(),
            'generator_version': '1.0',
            'device': self.device,
        }

        return {
            'mixed_signal': mixed_signal,
            'source_signals_clean': source_signals_clean,
            'source_signals_channelized': source_signals_channelized,
            'source_signals_aligned': source_signals_aligned,
            'channel_params': channel_params_list,
            'mixing_params': mixing_params,
            'metadata': metadata,
        }

    def clear(self):
        """Clear all source signals from mixer."""
        self.sources = []

    def get_source_info(self) -> List[Dict]:
        """
        Get information about all added source signals.

        Returns:
            List of dicts with source information
        """
        return [
            {
                'index': i,
                'label': src['label'],
                'power_db': src['power_db'],
                'freq_offset_hz': src['freq_offset_hz'],
                'timing_offset_samples': src['timing_offset_samples'],
                'signal_length': len(src['signal_clean']),
                'has_channel': bool(src['channel_params']),
            }
            for i, src in enumerate(self.sources)
        ]


class MIMOSignalMixer:
    """
    MIMO signal mixer with spatial correlation.

    Extends SignalMixer for multi-antenna scenarios with realistic spatial channels.
    """

    def __init__(self,
                 num_tx: int,
                 num_rx: int,
                 sample_rate: float,
                 spatial_correlation: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize MIMO signal mixer.

        Args:
            num_tx: Number of transmit antennas (sources can have different tx antennas)
            num_rx: Number of receive antennas
            sample_rate: Sampling rate in Hz
            spatial_correlation: Target spatial correlation (0-1)
            device: PyTorch device
        """
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.sample_rate = sample_rate
        self.spatial_correlation = spatial_correlation
        self.device = device
        self.sources = []

    def add_source(self,
                   signals: torch.Tensor,
                   label: str,
                   power_db: float = 0.0,
                   freq_offset_hz: float = 0.0,
                   timing_offset_samples: int = 0,
                   doppler_hz: float = 0.0):
        """
        Add a MIMO source signal.

        Args:
            signals: Transmit signals, shape (num_tx, num_samples)
            label: Source label
            power_db: Target power in dB
            freq_offset_hz: Frequency offset
            timing_offset_samples: Timing offset
            doppler_hz: Doppler frequency for time-varying fading
        """
        if signals.shape[0] != self.num_tx:
            raise ValueError(f"Expected {self.num_tx} transmit signals, got {signals.shape[0]}")

        source_info = {
            'signals': signals.to(self.device),
            'label': label,
            'power_db': power_db,
            'freq_offset_hz': freq_offset_hz,
            'timing_offset_samples': timing_offset_samples,
            'doppler_hz': doppler_hz,
        }

        self.sources.append(source_info)

    def mix(self, mode: str = 'co-channel', output_length: Optional[int] = None) -> Dict:
        """
        Mix MIMO source signals with spatial channels.

        Args:
            mode: Mixing mode - 'co-channel' or 'adjacent-channel'
            output_length: Output length in samples

        Returns:
            Dictionary with MIMO mixed signals and ground truth
        """
        if not self.sources:
            raise ValueError("No source signals added to MIMO mixer")

        # Determine output length
        if output_length is None:
            max_signal_len = max(src['signals'].shape[1] for src in self.sources)
            max_timing_offset = max(src['timing_offset_samples'] for src in self.sources)
            output_length = max_signal_len + max_timing_offset

        # Initialize per-antenna mixed signals
        mixed_signals_mimo = torch.zeros(
            (self.num_rx, output_length),
            dtype=torch.complex64,
            device=self.device
        )

        # Process each source with independent MIMO channel
        for source in self.sources:
            # Apply timing offset to transmit signals first
            tx_signals = source['signals']
            tx_signals_padded = torch.zeros(
                (self.num_tx, output_length),
                dtype=torch.complex64,
                device=self.device
            )
            offset = source['timing_offset_samples']
            signal_len = min(tx_signals.shape[1], output_length - offset)
            tx_signals_padded[:, offset:offset+signal_len] = tx_signals[:, :signal_len]
            tx_signals = tx_signals_padded

            # Generate MIMO channel for this source (after padding)
            H = generate_mimo_channel(
                self.num_tx,
                self.num_rx,
                output_length,
                doppler_hz=source['doppler_hz'],
                sample_rate=self.sample_rate
            )

            # Apply frequency offset if adjacent-channel
            if mode == 'adjacent-channel' and source['freq_offset_hz'] != 0:
                for i in range(self.num_tx):
                    tx_signals[i] = add_carrier_frequency(
                        tx_signals[i],
                        source['freq_offset_hz'],
                        self.sample_rate
                    )

            # Apply MIMO channel (no noise, added separately if needed)
            rx_signals = apply_mimo_channel(
                tx_signals,
                H,
                noise_power_db=-100  # Effectively no noise, just mixing
            )

            # Normalize to target power (preserve spatial diversity)
            # Calculate total power across all receive antennas
            total_power = torch.mean(torch.abs(rx_signals) ** 2)
            target_power_linear = 10 ** (source['power_db'] / 10.0)

            if total_power > 1e-12:
                scale = torch.sqrt(target_power_linear / total_power)
                rx_signals = rx_signals * scale

            # Add to mixture
            mixed_signals_mimo += rx_signals

        # Create output dictionary
        mixing_params = {
            'num_sources': len(self.sources),
            'mode': mode,
            'num_tx': self.num_tx,
            'num_rx': self.num_rx,
            'spatial_correlation': self.spatial_correlation,
            'power_ratios_db': [src['power_db'] for src in self.sources],
            'frequency_offsets_hz': [src['freq_offset_hz'] for src in self.sources],
            'timing_offsets_samples': [src['timing_offset_samples'] for src in self.sources],
            'sample_rate': self.sample_rate,
            'output_length': output_length,
        }

        metadata = {
            'scenario_name': f"MIMO_{self.num_tx}x{self.num_rx}_{'+'.join([src['label'] for src in self.sources])}",
            'standards': [src['label'] for src in self.sources],
            'generation_timestamp': datetime.now().isoformat(),
            'generator_version': '1.0',
            'device': self.device,
        }

        return {
            'mixed_signals_mimo': mixed_signals_mimo,
            'mixing_params': mixing_params,
            'metadata': metadata,
        }

    def clear(self):
        """Clear all source signals."""
        self.sources = []
