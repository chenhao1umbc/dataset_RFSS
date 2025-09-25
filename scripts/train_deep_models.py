#!/usr/bin/env python3
"""
Train deep learning models (CNN-LSTM, Conv-TasNet, DPRNN) and get real SINR numbers
This replaces the fabricated performance claims in the paper with actual experimental results
"""
import sys
import os
import h5py
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_algorithms.cnn_lstm import CNNLSTMSeparator, SourceSeparationTrainer
from ml_algorithms.conv_tasnet import ConvTasNet, ConvTasNetTrainer
from ml_algorithms.dprnn import DualPathRNN, DPRNNTrainer
from ml_algorithms.baseline_algorithms import (
    ICASourceSeparation, NMFSourceSeparation, compute_sinr
)


def load_dataset(dataset_path: str, num_samples: int = None) -> dict:
    """Load HDF5 dataset for training"""
    print(f"Loading dataset from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return None
    
    data = {'mixed_signals': [], 'target_signals': []}
    
    with h5py.File(dataset_path, 'r') as f:
        print(f"Dataset contains {f.attrs.get('total_samples', 'unknown')} samples")
        
        batches = f['batches']
        sample_count = 0
        
        for batch_name in batches.keys():
            if num_samples and sample_count >= num_samples:
                break
                
            batch = batches[batch_name]
            signals = batch['signals']
            components = batch['components']
            
            for signal_name in signals.keys():
                if num_samples and sample_count >= num_samples:
                    break
                
                # Extract sample ID from signal name (e.g., 'mixed_000001')
                sample_id = signal_name.split('_')[-1]
                
                mixed_signal = signals[signal_name][:]
                
                # Get component signals
                if f'sample_{sample_id}' in components:
                    comp_group = components[f'sample_{sample_id}']
                    targets = {}
                    
                    for std_name in comp_group.keys():
                        targets[std_name] = comp_group[std_name][:]
                    
                    data['mixed_signals'].append(mixed_signal)
                    data['target_signals'].append(targets)
                    sample_count += 1
    
    print(f"Loaded {len(data['mixed_signals'])} samples")
    return data


def prepare_pytorch_data(mixed_signals: list, target_signals: list, device: str):
    """Convert to PyTorch tensors"""
    def complex_to_tensor(signal):
        """Convert complex signal to [real, imag] tensor"""
        return torch.tensor(np.stack([signal.real, signal.imag]), dtype=torch.float32)
    
    # Process mixed signals
    mixed_tensors = []
    target_dicts = []
    
    for i in range(len(mixed_signals)):
        # Mixed signal
        mixed_tensor = complex_to_tensor(mixed_signals[i]).to(device)
        mixed_tensors.append(mixed_tensor)
        
        # Target signals
        target_dict = {}
        for std_name, signal in target_signals[i].items():
            target_dict[std_name] = complex_to_tensor(signal).to(device)
        target_dicts.append(target_dict)
    
    return mixed_tensors, target_dicts


def train_model(model, trainer, mixed_tensors, target_dicts, epochs: int = 10):
    """Train a model and return performance metrics"""
    print(f"Training {model.__class__.__name__} for {epochs} epochs...")
    
    # Split into train/val
    split_idx = int(0.8 * len(mixed_tensors))
    train_mixed = mixed_tensors[:split_idx]
    train_targets = target_dicts[:split_idx]
    val_mixed = mixed_tensors[split_idx:]
    val_targets = target_dicts[split_idx:]
    
    train_losses = []
    val_metrics = []
    
    best_val_sinr = -float('inf')
    
    for epoch in range(epochs):
        # Training
        epoch_losses = []
        for i in tqdm(range(len(train_mixed)), desc=f"Epoch {epoch+1}/{epochs}"):
            mixed = train_mixed[i].unsqueeze(0)  # Add batch dim
            targets = {k: v.unsqueeze(0) for k, v in train_targets[i].items()}
            
            loss = trainer.train_step(mixed, targets)
            epoch_losses.append(loss)
        
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)
        
        # Validation
        val_losses = []
        val_sinr_all = []
        
        for i in range(len(val_mixed)):
            mixed = val_mixed[i].unsqueeze(0)
            targets = {k: v.unsqueeze(0) for k, v in val_targets[i].items()}
            
            metrics = trainer.evaluate(mixed, targets)
            val_losses.append(metrics['loss'])
            if 'mean_si_snr' in metrics:
                val_sinr_all.append(metrics['mean_si_snr'])
            elif 'si_snr_metrics' in metrics:
                val_sinr_all.append(np.mean(list(metrics['si_snr_metrics'].values())))
        
        val_loss = np.mean(val_losses)
        val_sinr = np.mean(val_sinr_all) if val_sinr_all else 0.0
        val_metrics.append({'loss': val_loss, 'sinr': val_sinr})
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val SINR={val_sinr:.2f} dB")
        
        # Update learning rate
        if hasattr(trainer, 'scheduler'):
            trainer.scheduler.step(val_loss)
        
        # Track best validation SINR
        if val_sinr > best_val_sinr:
            best_val_sinr = val_sinr
    
    return {
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'best_val_sinr': best_val_sinr,
        'final_val_sinr': val_metrics[-1]['sinr'] if val_metrics else 0.0
    }


def evaluate_baselines(mixed_signals: list, target_signals: list) -> dict:
    """Evaluate baseline algorithms (ICA, NMF)"""
    print("Evaluating baseline algorithms...")
    
    results = {}
    
    # ICA
    print("Testing ICA...")
    ica_sinrs = []
    for i in tqdm(range(min(50, len(mixed_signals)))):  # Test on subset for speed
        mixed = mixed_signals[i]
        targets = target_signals[i]
        
        try:
            ica = ICASourceSeparation(n_components=len(targets))
            separated = ica.fit_separate(mixed)
            
            if len(separated) >= 2:
                # Find best assignment
                best_sinr = -float('inf')
                target_list = list(targets.values())
                
                for perm in [[0, 1], [1, 0]] if len(separated) >= 2 else [[0]]:
                    sinr_sum = 0
                    for j, target_idx in enumerate(perm):
                        if target_idx < len(target_list) and j < len(separated):
                            sinr_sum += compute_sinr(separated[j], target_list[target_idx])
                    
                    if sinr_sum > best_sinr:
                        best_sinr = sinr_sum / len(perm)
                
                ica_sinrs.append(best_sinr)
            else:
                ica_sinrs.append(-20.0)  # Failed
        except:
            ica_sinrs.append(-20.0)
    
    results['ICA'] = {
        'mean_sinr': np.mean(ica_sinrs),
        'std_sinr': np.std(ica_sinrs)
    }
    
    # NMF  
    print("Testing NMF...")
    nmf_sinrs = []
    for i in tqdm(range(min(50, len(mixed_signals)))):
        mixed = mixed_signals[i]
        targets = target_signals[i]
        
        try:
            nmf = NMFSourceSeparation(n_components=len(targets))
            separated = nmf.fit_separate(mixed)
            
            if len(separated) >= 2:
                # Find best assignment
                best_sinr = -float('inf')
                target_list = list(targets.values())
                
                for perm in [[0, 1], [1, 0]] if len(separated) >= 2 else [[0]]:
                    sinr_sum = 0
                    for j, target_idx in enumerate(perm):
                        if target_idx < len(target_list) and j < len(separated):
                            sinr_sum += compute_sinr(separated[j], target_list[target_idx])
                    
                    if sinr_sum > best_sinr:
                        best_sinr = sinr_sum / len(perm)
                
                nmf_sinrs.append(best_sinr)
            else:
                nmf_sinrs.append(-15.0)  # Failed
        except:
            nmf_sinrs.append(-15.0)
    
    results['NMF'] = {
        'mean_sinr': np.mean(nmf_sinrs),
        'std_sinr': np.std(nmf_sinrs)
    }
    
    return results


def main():
    """Main training and evaluation function"""
    parser = argparse.ArgumentParser(description='Train deep learning models for RF source separation')
    parser.add_argument('--dataset', type=str, default='data/large_dataset/rfss_large.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to use for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/mps/cuda/auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args.dataset, args.num_samples)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    # Evaluate baselines first
    baseline_results = evaluate_baselines(dataset['mixed_signals'], dataset['target_signals'])
    
    # Prepare PyTorch data
    mixed_tensors, target_dicts = prepare_pytorch_data(
        dataset['mixed_signals'], dataset['target_signals'], device
    )
    
    # Initialize models
    signal_length = mixed_tensors[0].shape[-1]
    print(f"Original signal length: {signal_length}")
    
    # Downsample signals for MPS memory constraints
    max_length = 8000  # Limit to 8k samples for MPS
    if signal_length > max_length:
        print(f"Downsampling signals from {signal_length} to {max_length} for MPS compatibility")
        # Downsample all signals
        downsampled_mixed = []
        downsampled_targets = []
        
        for i in range(len(mixed_tensors)):
            # Downsample mixed signal
            mixed = mixed_tensors[i]
            step = signal_length // max_length
            downsampled = mixed[:, ::step][:, :max_length]
            downsampled_mixed.append(downsampled)
            
            # Downsample target signals
            downsampled_target = {}
            for std_name, signal in target_dicts[i].items():
                target_down = signal[:, ::step][:, :max_length]
                downsampled_target[std_name] = target_down
            downsampled_targets.append(downsampled_target)
        
        mixed_tensors = downsampled_mixed
        target_dicts = downsampled_targets
        signal_length = max_length
    
    print(f"Using signal length: {signal_length}")
    
    models = {
        'CNN-LSTM': {
            'model': CNNLSTMSeparator(
                input_length=signal_length,
                num_standards=4,
                cnn_channels=[2, 8, 16, 32],  # Even smaller for MPS
                lstm_hidden_size=64,
                dropout=0.1
            ).to(device),
            'trainer_class': SourceSeparationTrainer
        },
        'Conv-TasNet': {
            'model': ConvTasNet(
                N=64, L=8, B=32, H=64,  # Much smaller configuration
                P=3, X=3, R=2, C=4
            ).to(device),
            'trainer_class': ConvTasNetTrainer
        },
        'DPRNN': {
            'model': DualPathRNN(
                N=32, B=32, H=16, P=20,  # Very small configuration
                num_layers=2, C=4
            ).to(device),
            'trainer_class': DPRNNTrainer
        }
    }
    
    # Training results
    results = baseline_results.copy()
    
    # Train each model
    for model_name, model_config in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        try:
            model = model_config['model']
            trainer = model_config['trainer_class'](model, learning_rate=1e-3, device=device)
            
            metrics = train_model(model, trainer, mixed_tensors, target_dicts, args.epochs)
            
            results[model_name] = {
                'best_sinr': metrics['best_val_sinr'],
                'final_sinr': metrics['final_val_sinr'],
                'train_losses': metrics['train_losses'],
                'val_metrics': metrics['val_metrics']
            }
            
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
            results[model_name] = {
                'best_sinr': 0.0,
                'final_sinr': 0.0,
                'error': str(e)
            }
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE COMPARISON")
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
        
        if claim:
            diff = sinr - claim
            if abs(diff) < 2.0:
                status = "✓ Close to claim"
            elif diff > 2.0:
                status = "↑ Better than claim"  
            else:
                status = "↓ Worse than claim"
        else:
            status = "New implementation"
        
        print(f"{algo:<15} {sinr_str:<15} {claim_str:<15} {status:<20}")
    
    # Save results
    output_file = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                json_results[k] = {}
                for k2, v2 in v.items():
                    if isinstance(v2, (np.ndarray, list)):
                        json_results[k][k2] = [float(x) for x in v2] if hasattr(v2, '__iter__') else float(v2)
                    else:
                        json_results[k][k2] = float(v2) if isinstance(v2, (np.float32, np.float64)) else v2
            else:
                json_results[k] = v
                
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"✅ Training experiment completed on {len(mixed_tensors)} samples")
    
    return results


if __name__ == "__main__":
    results = main()