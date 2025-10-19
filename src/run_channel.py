"""
Realistic channel modeling demonstration per 3GPP specifications.

Demonstrates 3GPP TDL models, hardware impairments, and RF effects.
"""

import torch
import argparse
from utils_channel import (
    add_awgn,
    apply_tdl_channel,
    apply_cfo,
    apply_iq_imbalance,
    apply_dc_offset,
    apply_phase_noise,
    apply_pa_nonlinearity,
    generate_mimo_channel,
    apply_mimo_channel,
    TDL_MODELS
)


def demo_tdl_models(device='cpu'):
    """Demonstrate 3GPP TDL channel models."""
    print("\n" + "="*70)
    print("3GPP TDL Channel Models Demonstration (TR 38.901)")
    print("="*70)

    signal = torch.randn(5000, dtype=torch.complex64, device=device)
    signal = signal / torch.sqrt(torch.mean(torch.abs(signal) ** 2))
    sample_rate = 15.36e6

    for model_name in TDL_MODELS.keys():
        model_info = TDL_MODELS[model_name]
        print(f"\n{model_name}: {model_info['description']}")
        print(f"  Number of taps: {len(model_info['delays_normalized'])}")
        print(f"  Typical delay spread: {model_info['typical_ds_ns']} ns")

        faded = apply_tdl_channel(signal, tdl_model=model_name, sample_rate=sample_rate, device=device)

        input_power = torch.mean(torch.abs(signal) ** 2).item()
        output_power = torch.mean(torch.abs(faded) ** 2).item()
        print(f"  Power preservation: {output_power/input_power:.3f}")


def demo_hardware_impairments(device='cpu'):
    """Demonstrate hardware impairments."""
    print("\n" + "="*70)
    print("Hardware Impairments Demonstration")
    print("="*70)

    signal = torch.randn(5000, dtype=torch.complex64, device=device)
    signal = signal / torch.sqrt(torch.mean(torch.abs(signal) ** 2))
    sample_rate = 15.36e6

    print("\n1. Carrier Frequency Offset (CFO) - 3GPP TS 38.104/38.101")
    for cfo_ppm in [0.1, 1.0, 5.0]:
        cfo_hz = cfo_ppm * 3.5e9 / 1e6
        rotated = apply_cfo(signal, cfo_hz, sample_rate, device=device)
        print(f"  CFO {cfo_ppm} ppm ({cfo_hz:.1f} Hz): power = {torch.mean(torch.abs(rotated)**2).item():.6f}")

    print("\n2. I/Q Imbalance - 3GPP TS 36.101 (Image rejection >= 25 dB)")
    for amp_db in [0.5, 1.0, 2.0]:
        distorted = apply_iq_imbalance(signal, amplitude_imb_db=amp_db, phase_imb_deg=5.0, device=device)
        error = torch.mean(torch.abs(signal - distorted)).item()
        print(f"  Amplitude imbalance {amp_db} dB: error = {error:.6f}")

    print("\n3. DC Offset (LO leakage)")
    for dc_dbc in [-30, -35, -40]:
        with_dc = apply_dc_offset(signal, dc_level_dbc=dc_dbc, device=device)
        dc_mag = torch.abs(torch.mean(with_dc - signal)).item()
        print(f"  DC offset {dc_dbc} dBc: magnitude = {dc_mag:.6f}")

    print("\n4. Phase Noise (Oscillator imperfection) - 3GPP TS 25.102")
    for pn_dbc in [-90, -100, -110]:
        noisy = apply_phase_noise(signal, phase_noise_dbc_hz=pn_dbc, sample_rate=sample_rate, device=device)
        phase_var = torch.var(torch.angle(noisy)).item()
        print(f"  Phase noise {pn_dbc} dBc/Hz: phase variance = {phase_var:.6f}")

    print("\n5. Power Amplifier Nonlinearity (Rapp model)")
    for ibo_db in [3, 6, 9]:
        amplified = apply_pa_nonlinearity(signal, input_backoff_db=ibo_db, device=device)
        peak_reduction = (torch.max(torch.abs(signal)) - torch.max(torch.abs(amplified))).item()
        print(f"  Input back-off {ibo_db} dB: peak reduction = {peak_reduction:.6f}")


def demo_realistic_scenario(device='cpu'):
    """Demonstrate realistic combined scenario."""
    print("\n" + "="*70)
    print("Realistic Channel Scenario: TDL + CFO + I/Q Imbalance + AWGN")
    print("="*70)

    signal = torch.randn(10000, dtype=torch.complex64, device=device)
    signal = signal / torch.sqrt(torch.mean(torch.abs(signal) ** 2))
    sample_rate = 15.36e6

    print("\nOriginal signal:")
    print(f"  Power: {torch.mean(torch.abs(signal)**2).item():.6f}")
    print(f"  Peak: {torch.max(torch.abs(signal)).item():.6f}")

    print("\nApplying TDL-A channel (NLOS)...")
    faded = apply_tdl_channel(signal, tdl_model='TDL-A', doppler_hz=100, sample_rate=sample_rate, device=device)
    print(f"  Power after fading: {torch.mean(torch.abs(faded)**2).item():.6f}")

    print("\nApplying CFO (5 ppm at 3.5 GHz = 17.5 kHz)...")
    with_cfo = apply_cfo(faded, cfo_hz=17500, sample_rate=sample_rate, device=device)
    print(f"  Power after CFO: {torch.mean(torch.abs(with_cfo)**2).item():.6f}")

    print("\nApplying I/Q imbalance (1 dB, 5 deg)...")
    with_iq = apply_iq_imbalance(with_cfo, amplitude_imb_db=1.0, phase_imb_deg=5.0, device=device)
    print(f"  Power after I/Q imb: {torch.mean(torch.abs(with_iq)**2).item():.6f}")

    print("\nApplying AWGN (SNR=20 dB)...")
    final = add_awgn(with_iq, snr_db=20, device=device)
    print(f"  Power after noise: {torch.mean(torch.abs(final)**2).item():.6f}")

    degradation_db = 10 * torch.log10(torch.mean(torch.abs(signal)**2) / torch.mean(torch.abs(final - signal)**2))
    print(f"\nTotal degradation: {degradation_db.item():.2f} dB")


def demo_mimo_with_doppler(device='cpu'):
    """Demonstrate MIMO with time-varying channels."""
    print("\n" + "="*70)
    print("MIMO with Time-Varying Fading (Jakes' Model)")
    print("="*70)

    num_tx, num_rx = 4, 4
    num_samples = 5000
    sample_rate = 15.36e6

    print(f"\nConfiguration: {num_tx}x{num_rx} MIMO")
    print(f"Doppler frequency: 100 Hz (30 km/h at 2 GHz)")

    H = generate_mimo_channel(
        num_tx, num_rx, num_samples,
        doppler_hz=100,
        sample_rate=sample_rate,
        device=device
    )

    print(f"Channel matrix shape: {H.shape}")

    h_start = H[0, 0, 0]
    h_middle = H[0, 0, num_samples//2]
    h_end = H[0, 0, -1]

    print(f"\nChannel variation over time (H[0,0,:]):")
    print(f"  Start magnitude: {torch.abs(h_start).item():.6f}")
    print(f"  Middle magnitude: {torch.abs(h_middle).item():.6f}")
    print(f"  End magnitude: {torch.abs(h_end).item():.6f}")
    print(f"  Start-End difference: {torch.abs(h_start - h_end).item():.6f}")

    tx_signals = torch.randn(num_tx, num_samples, dtype=torch.complex64, device=device)
    for i in range(num_tx):
        tx_signals[i] = tx_signals[i] / torch.sqrt(torch.mean(torch.abs(tx_signals[i])**2))

    rx_signals = apply_mimo_channel(tx_signals, H, noise_power_db=-20, device=device)

    print(f"\nReceived signal powers:")
    for i in range(num_rx):
        power = torch.mean(torch.abs(rx_signals[i])**2).item()
        print(f"  RX antenna {i}: {power:.6f}")


def main():
    parser = argparse.ArgumentParser(description='Realistic Channel Modeling Demonstration')
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'tdl', 'impairments', 'realistic', 'mimo'],
                       help='Which demonstration to run')
    parser.add_argument('--device', type=str, default='cpu',
                       help='PyTorch device (cpu, cuda, mps)')

    args = parser.parse_args()

    print("="*70)
    print("Realistic Channel Modeling Demonstration")
    print("Based on 3GPP TR 38.901 and hardware specifications")
    print("="*70)
    print(f"Device: {args.device}")

    if args.demo in ['all', 'tdl']:
        demo_tdl_models(args.device)

    if args.demo in ['all', 'impairments']:
        demo_hardware_impairments(args.device)

    if args.demo in ['all', 'realistic']:
        demo_realistic_scenario(args.device)

    if args.demo in ['all', 'mimo']:
        demo_mimo_with_doppler(args.device)

    print("\n" + "="*70)
    print("All demonstrations completed")
    print("="*70)


if __name__ == '__main__':
    main()
