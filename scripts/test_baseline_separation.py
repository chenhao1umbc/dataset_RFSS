#!/usr/bin/env python3
"""
Test baseline source separation algorithms (ICA, NMF) on real signals
Get actual SINR performance numbers for paper validation
"""
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ml_algorithms.baseline_algorithms import (
    ICASourceSeparation, NMFSourceSeparation, compute_sinr
)
from mixing.signal_mixer import SignalMixer
from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator


def create_test_mixtures(num_samples: int = 30) -> list:
    """Create test mixed signals with known ground truth"""
    print(f"Creating {num_samples} test mixtures with ground truth...")
    
    sample_rate = 30.72e6
    duration = 0.005  # 5ms for faster processing
    mixtures = []
    
    for i in tqdm(range(num_samples)):
        # Generate GSM and LTE signals
        gsm_gen = GSMGenerator(sample_rate=sample_rate, duration=duration)
        gsm_signal = gsm_gen.generate_baseband()
        
        lte_gen = LTEGenerator(sample_rate=sample_rate, duration=duration, bandwidth=20)
        lte_signal = lte_gen.generate_baseband()
        
        # Create mixture
        mixer = SignalMixer(sample_rate)
        mixer.add_signal(gsm_signal, carrier_freq=900e6, power_db=0, label='GSM')
        mixer.add_signal(lte_signal, carrier_freq=1.8e9, power_db=-3, label='LTE')
        
        mixed_signal, _ = mixer.mix_signals(duration=duration)
        
        # Store with ground truth
        mixtures.append({
            'mixed': mixed_signal,
            'gsm_true': gsm_signal,
            'lte_true': lte_signal,
            'sample_id': i
        })
    
    return mixtures


def evaluate_ica_performance(mixtures: list) -> dict:
    """Evaluate ICA source separation"""
    print("\n=== Evaluating ICA Performance ===")
    
    sinr_results = []
    success_count = 0
    
    for mixture in tqdm(mixtures):
        mixed = mixture['mixed']
        gsm_true = mixture['gsm_true']
        lte_true = mixture['lte_true']
        
        try:
            # Apply ICA
            ica = ICASourceSeparation(n_components=2, max_iter=50)
            separated = ica.fit_separate(mixed)
            
            if len(separated) >= 2:
                # Try both assignments and take the best
                assignment1_sinr = []
                assignment1_sinr.append(compute_sinr(separated[0], gsm_true))
                assignment1_sinr.append(compute_sinr(separated[1], lte_true))
                
                assignment2_sinr = []
                assignment2_sinr.append(compute_sinr(separated[1], gsm_true))
                assignment2_sinr.append(compute_sinr(separated[0], lte_true))
                
                # Choose better assignment
                if np.mean(assignment1_sinr) > np.mean(assignment2_sinr):
                    sinr_results.append(np.mean(assignment1_sinr))
                else:
                    sinr_results.append(np.mean(assignment2_sinr))
                
                success_count += 1
            else:
                sinr_results.append(-20.0)  # Failed separation
                
        except Exception as e:
            sinr_results.append(-20.0)
    
    return {
        'algorithm': 'ICA',
        'mean_sinr': np.mean(sinr_results),
        'std_sinr': np.std(sinr_results),
        'success_rate': success_count / len(mixtures),
        'results': sinr_results
    }


def evaluate_nmf_performance(mixtures: list) -> dict:
    """Evaluate NMF source separation"""
    print("\n=== Evaluating NMF Performance ===")
    
    sinr_results = []
    success_count = 0
    
    for mixture in tqdm(mixtures):
        mixed = mixture['mixed']
        gsm_true = mixture['gsm_true']
        lte_true = mixture['lte_true']
        
        try:
            # Apply NMF
            nmf = NMFSourceSeparation(n_components=2, max_iter=50)
            separated = nmf.fit_separate(mixed)
            
            if len(separated) >= 2:
                # Try both assignments
                assignment1_sinr = []
                assignment1_sinr.append(compute_sinr(separated[0], gsm_true))
                assignment1_sinr.append(compute_sinr(separated[1], lte_true))
                
                assignment2_sinr = []
                assignment2_sinr.append(compute_sinr(separated[1], gsm_true))
                assignment2_sinr.append(compute_sinr(separated[0], lte_true))
                
                # Choose better assignment
                if np.mean(assignment1_sinr) > np.mean(assignment2_sinr):
                    sinr_results.append(np.mean(assignment1_sinr))
                else:
                    sinr_results.append(np.mean(assignment2_sinr))
                
                success_count += 1
            else:
                sinr_results.append(-15.0)  # Failed separation
                
        except Exception as e:
            sinr_results.append(-15.0)
    
    return {
        'algorithm': 'NMF',
        'mean_sinr': np.mean(sinr_results),
        'std_sinr': np.std(sinr_results),
        'success_rate': success_count / len(mixtures),
        'results': sinr_results
    }


def main():
    """Main evaluation function"""
    print("Baseline Source Separation Algorithm Evaluation")
    print("=" * 60)
    
    # Create test data
    mixtures = create_test_mixtures(num_samples=20)  # Small for testing
    
    # Evaluate algorithms
    ica_results = evaluate_ica_performance(mixtures)
    nmf_results = evaluate_nmf_performance(mixtures)
    
    # Print results
    print("\n" + "=" * 60)
    print("BASELINE ALGORITHM PERFORMANCE")
    print("=" * 60)
    print(f"{'Algorithm':<10} {'SINR (dB)':<12} {'Success Rate':<15} {'Paper Claim':<15}")
    print("-" * 60)
    
    paper_claims = {'ICA': 15.2, 'NMF': 18.3}
    
    for results in [ica_results, nmf_results]:
        algo = results['algorithm']
        sinr = results['mean_sinr']
        std = results['std_sinr'] 
        success = results['success_rate']
        claim = paper_claims[algo]
        
        print(f"{algo:<10} {sinr:>6.1f}±{std:.1f}{'':<3} {success:>10.1%}{'':<5} {claim:>10.1f}")
        
        # Compare with paper claim
        diff = sinr - claim
        if abs(diff) < 3.0:
            status = "✓ Close to claim"
        elif diff > 3.0:
            status = "↑ Better than claim"
        else:
            status = "↓ Worse than claim"
        
        print(f"{'':10} Difference: {diff:+.1f} dB ({status})")
    
    print("\n" + "=" * 60)
    print("PAPER UPDATE RECOMMENDATIONS")
    print("=" * 60)
    
    # ICA recommendations
    ica_sinr = ica_results['mean_sinr']
    print(f"\nICA Algorithm:")
    print(f"  Paper claim: 15.2 dB SINR")
    print(f"  Actual result: {ica_sinr:.1f} ± {ica_results['std_sinr']:.1f} dB")
    print(f"  Recommendation: Update paper to {ica_sinr:.1f} dB")
    
    # NMF recommendations
    nmf_sinr = nmf_results['mean_sinr']
    print(f"\nNMF Algorithm:")
    print(f"  Paper claim: 18.3 dB SINR")
    print(f"  Actual result: {nmf_sinr:.1f} ± {nmf_results['std_sinr']:.1f} dB")
    print(f"  Recommendation: Update paper to {nmf_sinr:.1f} dB")
    
    # CNN-LSTM recommendations
    print(f"\nCNN-LSTM Algorithm:")
    print(f"  Paper claim: 26.7 dB SINR")
    print(f"  Status: Not implemented (requires PyTorch training)")
    print(f"  Recommendation: Implement and train, or reduce claim to realistic level")
    
    print(f"\n✓ Baseline evaluation completed on {len(mixtures)} test samples")
    
    return ica_results, nmf_results


if __name__ == "__main__":
    ica_results, nmf_results = main()