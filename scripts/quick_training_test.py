#!/usr/bin/env python3
"""
Quick training test to get real SINR performance numbers
Generate synthetic test data and train models for a few epochs
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_algorithms.cnn_lstm import CNNLSTMSeparator, SourceSeparationTrainer
from ml_algorithms.conv_tasnet import ConvTasNet, ConvTasNetTrainer
from ml_algorithms.dprnn import DualPathRNN, DPRNNTrainer
from ml_algorithms.baseline_algorithms import (
    ICASourceSeparation, NMFSourceSeparation, compute_sinr
)
from mixing.signal_mixer import SignalMixer
from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator


def generate_test_data(num_samples: int = 50, signal_length: int = 8000):
    """Generate synthetic test data for training"""
    print(f"Generating {num_samples} test samples...")
    
    sample_rate = 30.72e6
    duration = signal_length / sample_rate  # Calculate duration from signal length
    
    mixed_signals = []
    target_dicts = []
    
    for i in tqdm(range(num_samples)):
        # Set random seed for reproducibility
        np.random.seed(i + 42)
        
        # Generate GSM and LTE signals
        gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        gsm_signal = gsm_gen.generate_baseband()
        
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=20)
        lte_signal = lte_gen.generate_baseband()
        
        # Truncate or pad to exact length
        gsm_signal = gsm_signal[:signal_length] if len(gsm_signal) >= signal_length else np.pad(gsm_signal, (0, signal_length - len(gsm_signal)))
        lte_signal = lte_signal[:signal_length] if len(lte_signal) >= signal_length else np.pad(lte_signal, (0, signal_length - len(lte_signal)))
        
        # Create mixture with SignalMixer
        mixer = SignalMixer(sample_rate)
        mixer.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
        mixer.add_signal(lte_signal, carrier_freq=1.8e9, power_db=-3, label='LTE')
        
        mixed_signal, _ = mixer.mix_signals(duration=duration)
        mixed_signal = mixed_signal[:signal_length]  # Ensure exact length
        
        # Convert to tensors
        def complex_to_tensor(signal):
            return torch.tensor(np.stack([signal.real, signal.imag]), dtype=torch.float32)
        
        mixed_tensor = complex_to_tensor(mixed_signal)
        targets = {
            'GSM': complex_to_tensor(gsm_signal),
            'LTE': complex_to_tensor(lte_signal),
            'UMTS': torch.zeros_like(complex_to_tensor(gsm_signal)),  # Placeholder
            '5G_NR': torch.zeros_like(complex_to_tensor(gsm_signal))  # Placeholder
        }
        
        mixed_signals.append(mixed_tensor)
        target_dicts.append(targets)
    
    return mixed_signals, target_dicts


def evaluate_baselines(mixed_signals, target_dicts, num_test=20):
    """Evaluate baseline algorithms"""
    print("Evaluating baseline algorithms...")
    
    results = {}
    
    # ICA
    print("Testing ICA...")
    ica_sinrs = []
    for i in tqdm(range(min(num_test, len(mixed_signals)))):
        mixed = mixed_signals[i].numpy()
        mixed_complex = mixed[0] + 1j * mixed[1]  # Convert back to complex
        
        targets = target_dicts[i]
        gsm_target = targets['GSM'].numpy()
        lte_target = targets['LTE'].numpy()
        gsm_complex = gsm_target[0] + 1j * gsm_target[1]
        lte_complex = lte_target[0] + 1j * lte_target[1]
        
        try:
            ica = ICASourceSeparation(n_components=2)
            separated = ica.fit_separate(mixed_complex)
            
            if len(separated) >= 2:
                # Test both assignments
                sinr1 = (compute_sinr(separated[0], gsm_complex) + compute_sinr(separated[1], lte_complex)) / 2
                sinr2 = (compute_sinr(separated[1], gsm_complex) + compute_sinr(separated[0], lte_complex)) / 2
                ica_sinrs.append(max(sinr1, sinr2))
            else:
                ica_sinrs.append(-20.0)
        except:
            ica_sinrs.append(-20.0)
    
    results['ICA'] = {
        'mean_sinr': np.mean(ica_sinrs),
        'std_sinr': np.std(ica_sinrs)
    }
    
    # NMF
    print("Testing NMF...")
    nmf_sinrs = []
    for i in tqdm(range(min(num_test, len(mixed_signals)))):
        mixed = mixed_signals[i].numpy()
        mixed_complex = mixed[0] + 1j * mixed[1]
        
        targets = target_dicts[i]
        gsm_target = targets['GSM'].numpy()
        lte_target = targets['LTE'].numpy()
        gsm_complex = gsm_target[0] + 1j * gsm_target[1]
        lte_complex = lte_target[0] + 1j * lte_target[1]
        
        try:
            nmf = NMFSourceSeparation(n_components=2)
            separated = nmf.fit_separate(mixed_complex)
            
            if len(separated) >= 2:
                sinr1 = (compute_sinr(separated[0], gsm_complex) + compute_sinr(separated[1], lte_complex)) / 2
                sinr2 = (compute_sinr(separated[1], gsm_complex) + compute_sinr(separated[0], lte_complex)) / 2
                nmf_sinrs.append(max(sinr1, sinr2))
            else:
                nmf_sinrs.append(-15.0)
        except:
            nmf_sinrs.append(-15.0)
    
    results['NMF'] = {
        'mean_sinr': np.mean(nmf_sinrs),
        'std_sinr': np.std(nmf_sinrs)
    }
    
    return results


def train_model_simple(model, trainer, mixed_tensors, target_dicts, epochs=3):
    """Simplified training function"""
    print(f"Training {model.__class__.__name__}...")
    
    # Split data
    split_idx = int(0.8 * len(mixed_tensors))
    
    best_sinr = -float('inf')
    train_losses = []
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Training
        for i in range(split_idx):
            mixed = mixed_tensors[i].unsqueeze(0).to(trainer.device)
            targets = {k: v.unsqueeze(0).to(trainer.device) for k, v in target_dicts[i].items()}
            
            try:
                loss = trainer.train_step(mixed, targets)
                epoch_losses.append(loss)
            except Exception as e:
                print(f"Training step {i} failed: {e}")
                continue
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            # Validation
            val_sinrs = []
            for i in range(split_idx, len(mixed_tensors)):
                mixed = mixed_tensors[i].unsqueeze(0).to(trainer.device)
                targets = {k: v.unsqueeze(0).to(trainer.device) for k, v in target_dicts[i].items()}
                
                try:
                    metrics = trainer.evaluate(mixed, targets)
                    if hasattr(metrics, 'keys') and 'si_snr_metrics' in metrics:
                        val_sinrs.append(np.mean(list(metrics['si_snr_metrics'].values())))
                    elif hasattr(metrics, 'keys') and 'mean_si_snr' in metrics:
                        val_sinrs.append(metrics['mean_si_snr'])
                    elif isinstance(metrics, tuple):
                        _, sinr_dict = metrics
                        val_sinrs.append(np.mean(list(sinr_dict.values())))
                except Exception as e:
                    continue
            
            val_sinr = np.mean(val_sinrs) if val_sinrs else 0.0
            best_sinr = max(best_sinr, val_sinr)
            
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val SINR={val_sinr:.2f}dB")
    
    return best_sinr, train_losses


def main():
    """Main training function"""
    print("Quick Training Test for RF Source Separation")
    print("=" * 60)
    
    # Device selection
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate test data
    signal_length = 4000  # Smaller for MPS compatibility
    mixed_tensors, target_dicts = generate_test_data(num_samples=40, signal_length=signal_length)
    
    # Evaluate baselines
    baseline_results = evaluate_baselines(mixed_tensors, target_dicts)
    
    # Initialize models (very small configurations for quick testing)
    models = {
        'CNN-LSTM': {
            'model': CNNLSTMSeparator(
                input_length=signal_length,
                num_standards=4,
                cnn_channels=[2, 4, 8, 16],
                lstm_hidden_size=32,
                dropout=0.1
            ),
            'trainer_class': SourceSeparationTrainer
        },
        'Conv-TasNet': {
            'model': ConvTasNet(
                N=32, L=8, B=16, H=32,
                P=3, X=2, R=2, C=4
            ),
            'trainer_class': ConvTasNetTrainer
        },
        'DPRNN': {
            'model': DualPathRNN(
                N=16, B=16, H=8, P=10,
                num_layers=2, C=4
            ),
            'trainer_class': DPRNNTrainer
        }
    }
    
    # Train models
    results = baseline_results.copy()
    
    for model_name, model_config in models.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}")
        
        try:
            model = model_config['model'].to(device)
            trainer = model_config['trainer_class'](model, learning_rate=1e-3, device=device)
            
            best_sinr, losses = train_model_simple(model, trainer, mixed_tensors, target_dicts, epochs=3)
            
            results[model_name] = {
                'best_sinr': best_sinr,
                'train_losses': losses
            }
            
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
            results[model_name] = {
                'best_sinr': 0.0,
                'error': str(e)
            }
    
    # Print results
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON - REAL EXPERIMENTAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15} {'SINR (dB)':<15} {'Paper Claim':<15} {'Status':<20}")
    print(f"{'-'*80}")
    
    paper_claims = {
        'ICA': 15.2,
        'NMF': 18.3,
        'CNN-LSTM': 26.7,
        'Conv-TasNet': None,
        'DPRNN': None
    }
    
    for algo, result in results.items():
        if 'mean_sinr' in result:  # Baseline
            sinr = result['mean_sinr']
            std = result.get('std_sinr', 0)
            sinr_str = f"{sinr:.1f}±{std:.1f}"
        elif 'best_sinr' in result:  # Deep learning
            sinr = result['best_sinr']
            sinr_str = f"{sinr:.1f}"
        else:
            sinr_str = "Failed"
            sinr = 0
        
        claim = paper_claims.get(algo, None)
        claim_str = f"{claim:.1f}" if claim else "N/A"
        
        if claim and sinr != 0:
            diff = sinr - claim
            if abs(diff) < 2.0:
                status = "✓ Close to claim"
            elif diff > 2.0:
                status = "↑ Better than claim"
            else:
                status = "↓ Much worse than claim"
        else:
            status = "New result" if claim is None else "Failed training"
        
        print(f"{algo:<15} {sinr_str:<15} {claim_str:<15} {status:<20}")
    
    # Paper update recommendations
    print(f"\n{'='*80}")
    print("PAPER UPDATE RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print("\nBased on actual experimental results:")
    for algo, result in results.items():
        if algo in ['ICA', 'NMF', 'CNN-LSTM']:
            if 'mean_sinr' in result:
                actual_sinr = result['mean_sinr']
            elif 'best_sinr' in result:
                actual_sinr = result['best_sinr']
            else:
                actual_sinr = 0
            
            paper_claim = paper_claims[algo]
            print(f"\n{algo}:")
            print(f"  Paper claim: {paper_claim:.1f} dB SINR")
            print(f"  Actual result: {actual_sinr:.1f} dB SINR")
            
            if actual_sinr < paper_claim - 5:
                print(f"  🔴 CRITICAL: Actual result is {paper_claim - actual_sinr:.1f} dB WORSE than claimed")
                print(f"  📝 Recommendation: Update paper to {actual_sinr:.1f} dB or improve implementation")
            elif abs(actual_sinr - paper_claim) < 2:
                print(f"  ✅ GOOD: Results match paper claims within 2 dB")
            else:
                print(f"  📝 Recommendation: Update paper claim to {actual_sinr:.1f} dB")
    
    # Save results
    Path("results").mkdir(exist_ok=True)
    output_file = f"results/quick_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for k, v in results.items():
            json_results[k] = {}
            for k2, v2 in v.items():
                if isinstance(v2, (np.ndarray, list)):
                    json_results[k][k2] = [float(x) for x in v2] if hasattr(v2, '__iter__') else float(v2)
                elif isinstance(v2, (np.float32, np.float64, np.int32, np.int64)):
                    json_results[k][k2] = float(v2)
                else:
                    json_results[k][k2] = v2
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"🔬 Experiment completed on {len(mixed_tensors)} samples with {signal_length} sample signals")
    
    return results


if __name__ == "__main__":
    results = main()