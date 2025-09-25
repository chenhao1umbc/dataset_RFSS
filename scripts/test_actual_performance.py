#!/usr/bin/env python3
"""
Test actual code performance to get real benchmark numbers
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.umts_generator import UMTSGenerator 
from signal_generation.lte_generator import LTEGenerator
from signal_generation.nr_generator import NRGenerator


def benchmark_generator(generator_class, name, **kwargs):
    """Benchmark a signal generator"""
    print(f"\n=== Benchmarking {name} ===")
    
    # Test parameters
    sample_rate = 30.72e6  # 30.72 MHz
    duration = 0.01  # 10 ms
    num_runs = 10
    
    try:
        # Create generator
        generator = generator_class(sample_rate=sample_rate, duration=duration, **kwargs)
        
        # Warmup run
        _ = generator.generate_baseband()
        
        # Benchmark runs
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            start_time = time.time()
            signal = generator.generate_baseband()
            end_time = time.time()
            
            generation_time = end_time - start_time
            times.append(generation_time)
            
            # Memory usage (approximate)
            memory_mb = signal.nbytes / (1024 * 1024)
            memory_usage.append(memory_mb)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_memory = np.mean(memory_usage)
        
        # Real-time factor calculation
        realtime_factor = duration / avg_time
        
        print(f"Average generation time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"Real-time factor: {realtime_factor:.1f}×")
        print(f"Memory usage: {avg_memory:.1f} MB per {duration*1000:.0f}ms")
        print(f"Signal length: {len(signal)} samples")
        print(f"Signal power: {np.mean(np.abs(signal)**2):.6f}")
        
        return {
            'name': name,
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000, 
            'realtime_factor': realtime_factor,
            'memory_mb': avg_memory,
            'signal_length': len(signal),
            'signal_power': np.mean(np.abs(signal)**2)
        }
        
    except Exception as e:
        print(f"Error benchmarking {name}: {e}")
        return None


def main():
    """Main benchmarking function"""
    print("RF Signal Generation Performance Benchmark")
    print("=" * 50)
    
    results = []
    
    # Benchmark GSM
    result = benchmark_generator(GSMGenerator, "GSM (2G)")
    if result:
        results.append(result)
    
    # Benchmark UMTS
    result = benchmark_generator(UMTSGenerator, "UMTS (3G)")
    if result:
        results.append(result)
    
    # Benchmark LTE
    result = benchmark_generator(LTEGenerator, "LTE (4G)", bandwidth=20, modulation='64QAM')
    if result:
        results.append(result)
    
    # Benchmark 5G NR
    result = benchmark_generator(NRGenerator, "5G NR", bandwidth=50, modulation='256QAM', numerology=1)
    if result:
        results.append(result)
    
    # Summary table
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Standard':<15} {'Time (ms)':<12} {'Real-time':<12} {'Memory (MB)':<12} {'Power':<10}")
    print("-" * 80)
    
    for result in results:
        if result:
            time_str = f"{result['avg_time_ms']:.1f}±{result['std_time_ms']:.1f}"
            rt_str = f"{result['realtime_factor']:.0f}×"
            mem_str = f"{result['memory_mb']:.1f}"
            power_str = f"{result['signal_power']:.4f}"
            print(f"{result['name']:<15} {time_str:<12} {rt_str:<12} {mem_str:<12} {power_str:<10}")
    
    print("\nTest Configuration:")
    print(f"- Sample Rate: 30.72 MHz")
    print(f"- Duration: 10 ms")
    print(f"- Runs: 10 per standard")
    
    return results


if __name__ == "__main__":
    results = main()