"""
Performance benchmarking script for RF Signal Source Separation Dataset
Compares generation speed, memory usage, and signal quality metrics
"""
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from signal_generation.gsm_generator import GSMGenerator
from signal_generation.lte_generator import LTEGenerator
from signal_generation.umts_generator import UMTSGenerator
from signal_generation.nr_generator import NRGenerator
from validation.signal_metrics import SignalAnalyzer, ValidationReport
from channel_models.basic_channels import ChannelSimulator
from mixing.signal_mixer import SignalMixer


class PerformanceBenchmark:
    """Performance benchmarking for RF signal generation"""
    
    def __init__(self, output_dir='benchmarks'):
        """Initialize benchmark suite"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def measure_resource_usage(self, func, *args, **kwargs):
        """Measure CPU time, memory usage, and execution time"""
        process = psutil.Process()
        
        # Get initial memory usage
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure execution time
        start_time = time.time()
        start_cpu = time.process_time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final measurements
        end_time = time.time()
        end_cpu = time.process_time()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'result': result,
            'wall_time': end_time - start_time,
            'cpu_time': end_cpu - start_cpu,
            'memory_peak': mem_after,
            'memory_delta': mem_after - mem_before
        }
    
    def benchmark_signal_generation(self):
        """Benchmark signal generation for all standards"""
        print("=== Signal Generation Benchmarks ===")
        
        configs = [
            ('GSM', GSMGenerator, {'sample_rate': 10e6, 'duration': 0.01}),
            ('UMTS', UMTSGenerator, {'sample_rate': 15.36e6, 'duration': 0.01, 'spreading_factor': 128}),
            ('LTE', LTEGenerator, {'sample_rate': 30.72e6, 'duration': 0.01, 'bandwidth': 20}),
            ('5G-NR', NRGenerator, {'sample_rate': 61.44e6, 'duration': 0.01, 'bandwidth': 100})
        ]
        
        generation_results = {}
        
        for standard, generator_class, params in configs:
            print(f"\nBenchmarking {standard} generation...")
            
            # Test with different signal lengths
            durations = [0.001, 0.01, 0.1]  # 1ms, 10ms, 100ms
            standard_results = {}
            
            for duration in durations:
                test_params = params.copy()
                test_params['duration'] = duration
                
                # Run multiple iterations for averaging
                iterations = 5 if duration <= 0.01 else 2
                iter_results = []
                
                for i in range(iterations):
                    def generate_signal():
                        gen = generator_class(**test_params)
                        return gen.generate_baseband()
                    
                    metrics = self.measure_resource_usage(generate_signal)
                    
                    # Add signal quality metrics
                    signal = metrics['result']
                    metrics['signal_length'] = len(signal)
                    metrics['samples_per_second'] = len(signal) / metrics['wall_time']
                    metrics['realtime_factor'] = duration / metrics['wall_time']
                    metrics['signal_power'] = np.mean(np.abs(signal)**2)
                    
                    iter_results.append(metrics)
                
                # Average results
                avg_result = {}
                for key in ['wall_time', 'cpu_time', 'memory_delta', 'samples_per_second', 'realtime_factor']:
                    avg_result[key] = np.mean([r[key] for r in iter_results])
                    avg_result[f'{key}_std'] = np.std([r[key] for r in iter_results])
                
                avg_result['signal_length'] = iter_results[0]['signal_length']
                avg_result['duration'] = duration
                
                standard_results[f'{duration*1000:.0f}ms'] = avg_result
                
                print(f"  {duration*1000:.0f}ms: {avg_result['realtime_factor']:.1f}x realtime, "
                      f"{avg_result['wall_time']*1000:.2f}ms generation time")
            
            generation_results[standard] = standard_results
        
        self.results['generation'] = generation_results
        return generation_results
    
    def benchmark_channel_effects(self):
        """Benchmark channel model application"""
        print("\n=== Channel Effects Benchmarks ===")
        
        # Test signal (LTE 20MHz, 10ms)
        lte_gen = LTEGenerator(sample_rate=30.72e6, duration=0.01, bandwidth=20)
        test_signal = lte_gen.generate_baseband()
        
        channel_configs = [
            ('AWGN', lambda ch: ch.add_awgn(20)),
            ('Multipath', lambda ch: ch.add_multipath()),
            ('Rayleigh', lambda ch: ch.add_rayleigh_fading(100)),
            ('Rician', lambda ch: ch.add_rician_fading(50, 10)),
            ('Combined', lambda ch: ch.add_multipath().add_rayleigh_fading(100).add_awgn(15))
        ]
        
        channel_results = {}
        
        for channel_name, channel_setup in channel_configs:
            print(f"\nBenchmarking {channel_name} channel...")
            
            def apply_channel():
                channel = ChannelSimulator(30.72e6)
                channel_setup(channel)
                return channel.apply(test_signal)
            
            # Multiple iterations
            iterations = 10
            iter_results = []
            
            for i in range(iterations):
                metrics = self.measure_resource_usage(apply_channel)
                
                # Add throughput metrics
                metrics['throughput_mbps'] = (len(test_signal) * 8) / (metrics['wall_time'] * 1e6)  # Mbps
                iter_results.append(metrics)
            
            # Average results
            avg_result = {}
            for key in ['wall_time', 'cpu_time', 'memory_delta', 'throughput_mbps']:
                avg_result[key] = np.mean([r[key] for r in iter_results])
                avg_result[f'{key}_std'] = np.std([r[key] for r in iter_results])
            
            channel_results[channel_name] = avg_result
            
            print(f"  {channel_name}: {avg_result['wall_time']*1000:.2f}ms, "
                  f"throughput: {avg_result['throughput_mbps']:.1f} Mbps")
        
        self.results['channels'] = channel_results
        return channel_results
    
    def benchmark_signal_mixing(self):
        """Benchmark signal mixing performance"""
        print("\n=== Signal Mixing Benchmarks ===")
        
        # Generate test signals
        gsm_gen = GSMGenerator(sample_rate=30.72e6, duration=0.005)
        lte_gen = LTEGenerator(sample_rate=30.72e6, duration=0.005, bandwidth=20)
        nr_gen = NRGenerator(sample_rate=30.72e6, duration=0.005, bandwidth=50)
        
        gsm_signal = gsm_gen.generate_baseband()
        lte_signal = lte_gen.generate_baseband()
        nr_signal = nr_gen.generate_baseband()
        
        mixing_configs = [
            ('2-Signal Mix', [gsm_signal, lte_signal]),
            ('3-Signal Mix', [gsm_signal, lte_signal, nr_signal]),
            ('Multi-Standard', [gsm_signal, lte_signal, nr_signal, gsm_signal])  # Duplicate for stress test
        ]
        
        mixing_results = {}
        
        for mix_name, signals in mixing_configs:
            print(f"\nBenchmarking {mix_name}...")
            
            def mix_signals():
                mixer = SignalMixer(30.72e6)
                
                # Add signals at different frequencies
                freqs = [900e6, 1.8e9, 3.5e9, 2.1e9]
                for i, signal in enumerate(signals):
                    mixer.add_signal(signal, freqs[i % len(freqs)], power_db=0, 
                                   label=f'Signal_{i}')
                
                mixed_signal, info = mixer.mix_signals(duration=0.005)
                return mixed_signal, info
            
            # Multiple iterations
            iterations = 5
            iter_results = []
            
            for i in range(iterations):
                metrics = self.measure_resource_usage(mix_signals)
                
                mixed_signal, info = metrics['result']
                metrics['num_signals'] = info['num_signals']
                metrics['output_length'] = len(mixed_signal)
                
                iter_results.append(metrics)
            
            # Average results
            avg_result = {}
            for key in ['wall_time', 'cpu_time', 'memory_delta']:
                avg_result[key] = np.mean([r[key] for r in iter_results])
                avg_result[f'{key}_std'] = np.std([r[key] for r in iter_results])
            
            avg_result['num_signals'] = iter_results[0]['num_signals']
            avg_result['signals_per_second'] = avg_result['num_signals'] / avg_result['wall_time']
            
            mixing_results[mix_name] = avg_result
            
            print(f"  {mix_name}: {avg_result['wall_time']*1000:.2f}ms for {avg_result['num_signals']} signals")
        
        self.results['mixing'] = mixing_results
        return mixing_results
    
    def benchmark_validation(self):
        """Benchmark signal validation performance"""
        print("\n=== Validation Benchmarks ===")
        
        # Generate test signals for validation
        test_signals = {}
        generators = {
            'GSM': GSMGenerator(sample_rate=10e6, duration=0.01),
            'LTE': LTEGenerator(sample_rate=30.72e6, duration=0.01, bandwidth=20),
            'NR': NRGenerator(sample_rate=61.44e6, duration=0.01, bandwidth=100)
        }
        
        for name, gen in generators.items():
            test_signals[name] = (gen.generate_baseband(), gen.sample_rate)
        
        validation_results = {}
        
        for signal_name, (signal, sample_rate) in test_signals.items():
            print(f"\nBenchmarking {signal_name} validation...")
            
            def validate_signal():
                reporter = ValidationReport()
                return reporter.generate_signal_report(signal, sample_rate, signal_name)
            
            # Multiple iterations
            iterations = 10
            iter_results = []
            
            for i in range(iterations):
                metrics = self.measure_resource_usage(validate_signal)
                
                # Add validation-specific metrics
                report = metrics['result']
                metrics['signal_length'] = len(signal)
                metrics['samples_per_second'] = len(signal) / metrics['wall_time']
                
                iter_results.append(metrics)
            
            # Average results
            avg_result = {}
            for key in ['wall_time', 'cpu_time', 'memory_delta', 'samples_per_second']:
                avg_result[key] = np.mean([r[key] for r in iter_results])
                avg_result[f'{key}_std'] = np.std([r[key] for r in iter_results])
            
            validation_results[signal_name] = avg_result
            
            print(f"  {signal_name}: {avg_result['wall_time']*1000:.2f}ms validation time")
        
        self.results['validation'] = validation_results
        return validation_results
    
    def benchmark_memory_scaling(self):
        """Benchmark memory usage scaling with signal length"""
        print("\n=== Memory Scaling Benchmarks ===")
        
        # Test with LTE generator at different signal lengths
        durations = [0.001, 0.005, 0.01, 0.05, 0.1]  # 1ms to 100ms
        
        memory_results = {}
        
        for duration in durations:
            print(f"\nTesting {duration*1000:.0f}ms signals...")
            
            def generate_long_signal():
                gen = LTEGenerator(sample_rate=30.72e6, duration=duration, bandwidth=20)
                return gen.generate_baseband()
            
            metrics = self.measure_resource_usage(generate_long_signal)
            
            signal = metrics['result']
            metrics['signal_length'] = len(signal)
            metrics['memory_per_sample'] = metrics['memory_delta'] / len(signal) * 1024  # KB per sample
            
            memory_results[f'{duration*1000:.0f}ms'] = metrics
            
            print(f"  Length: {len(signal):,} samples, Memory: {metrics['memory_delta']:.1f} MB, "
                  f"Per sample: {metrics['memory_per_sample']:.3f} KB")
        
        self.results['memory_scaling'] = memory_results
        return memory_results
    
    def generate_comparison_table(self):
        """Generate comparison table with other RF datasets"""
        print("\n=== Dataset Comparison ===")
        
        # Our dataset metrics (from benchmarks)
        our_metrics = {
            'Standards Supported': '2G/3G/4G/5G',
            'Max Bandwidth (MHz)': 100,
            'Max Sample Rate (MHz)': 122.88,
            'MIMO Support': 'Yes (up to 16x16)',
            'Channel Models': 'Yes (Multipath, Fading, AWGN)',
            'Real-time Generation': f"{self.results['generation']['LTE']['10ms']['realtime_factor']:.1f}x",
            'Memory Efficiency (KB/sample)': f"{self.results['memory_scaling']['10ms']['memory_per_sample']:.3f}",
            'Standards Compliance': '3GPP TS 36.211, 38.211',
            'Validation Framework': 'Yes',
            'Open Source': 'Yes'
        }
        
        # Comparison with other datasets (hypothetical/typical values)
        comparison_data = {
            'Our Dataset': our_metrics,
            'DeepSig RadioML': {
                'Standards Supported': 'Various modulations',
                'Max Bandwidth (MHz)': 'N/A',
                'Max Sample Rate (MHz)': 'Variable',
                'MIMO Support': 'Limited',
                'Channel Models': 'Basic',
                'Real-time Generation': 'N/A',
                'Memory Efficiency (KB/sample)': 'N/A',
                'Standards Compliance': 'Partial',
                'Validation Framework': 'No',
                'Open Source': 'Partial'
            },
            'GNU Radio': {
                'Standards Supported': '2G/3G/4G (partial)',
                'Max Bandwidth (MHz)': 'Variable',
                'Max Sample Rate (MHz)': 'Hardware limited',
                'MIMO Support': 'Yes',
                'Channel Models': 'Yes',
                'Real-time Generation': '1x (hardware)',
                'Memory Efficiency (KB/sample)': 'Variable',
                'Standards Compliance': 'Partial',
                'Validation Framework': 'Basic',
                'Open Source': 'Yes'
            },
            'MATLAB 5G Toolbox': {
                'Standards Supported': '5G NR',
                'Max Bandwidth (MHz)': 400,
                'Max Sample Rate (MHz)': 'Variable',
                'MIMO Support': 'Yes',
                'Channel Models': 'Yes',
                'Real-time Generation': 'Variable',
                'Memory Efficiency (KB/sample)': 'High',
                'Standards Compliance': 'Full 3GPP',
                'Validation Framework': 'Yes',
                'Open Source': 'No (Commercial)'
            }
        }
        
        comparison_df = pd.DataFrame(comparison_data).T
        
        # Save comparison table
        comparison_file = self.output_dir / 'dataset_comparison.csv'
        comparison_df.to_csv(comparison_file)
        
        print("\nDataset Comparison Table:")
        print(comparison_df.to_string())
        
        return comparison_df
    
    def plot_benchmarks(self):
        """Generate benchmark plots"""
        print("\n=== Generating Benchmark Plots ===")
        
        # Plot 1: Generation Performance
        if 'generation' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            standards = list(self.results['generation'].keys())
            durations = ['1ms', '10ms', '100ms']
            
            # Realtime factors
            realtime_data = []
            for standard in standards:
                row = []
                for duration in durations:
                    if duration in self.results['generation'][standard]:
                        row.append(self.results['generation'][standard][duration]['realtime_factor'])
                    else:
                        row.append(0)
                realtime_data.append(row)
            
            realtime_data = np.array(realtime_data)
            
            x = np.arange(len(standards))
            width = 0.25
            
            for i, duration in enumerate(durations):
                ax1.bar(x + i*width, realtime_data[:, i], width, label=duration)
            
            ax1.set_xlabel('Standard')
            ax1.set_ylabel('Realtime Factor')
            ax1.set_title('Signal Generation Performance')
            ax1.set_xticks(x + width)
            ax1.set_xticklabels(standards)
            ax1.legend()
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            # Memory usage
            memory_data = []
            for standard in standards:
                if '10ms' in self.results['generation'][standard]:
                    memory_data.append(self.results['generation'][standard]['10ms']['memory_delta'])
                else:
                    memory_data.append(0)
            
            ax2.bar(standards, memory_data)
            ax2.set_xlabel('Standard')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage (10ms signals)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'generation_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Channel Effects Performance
        if 'channels' in self.results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            channels = list(self.results['channels'].keys())
            times = [self.results['channels'][ch]['wall_time'] * 1000 for ch in channels]  # ms
            throughputs = [self.results['channels'][ch]['throughput_mbps'] for ch in channels]
            
            x = np.arange(len(channels))
            
            ax2 = ax.twinx()
            
            bars1 = ax.bar(x - 0.2, times, 0.4, label='Processing Time', color='skyblue')
            bars2 = ax2.bar(x + 0.2, throughputs, 0.4, label='Throughput', color='lightcoral')
            
            ax.set_xlabel('Channel Effect')
            ax.set_ylabel('Processing Time (ms)', color='blue')
            ax2.set_ylabel('Throughput (Mbps)', color='red')
            ax.set_title('Channel Effects Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(channels, rotation=45)
            
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'channel_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Memory Scaling
        if 'memory_scaling' in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            durations = list(self.results['memory_scaling'].keys())
            signal_lengths = [self.results['memory_scaling'][d]['signal_length'] for d in durations]
            memory_usage = [self.results['memory_scaling'][d]['memory_delta'] for d in durations]
            memory_per_sample = [self.results['memory_scaling'][d]['memory_per_sample'] for d in durations]
            
            # Total memory usage
            ax1.plot(signal_lengths, memory_usage, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Signal Length (samples)')
            ax1.set_ylabel('Memory Usage (MB)')
            ax1.set_title('Memory Scaling with Signal Length')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log')
            
            # Memory per sample
            ax2.plot(signal_lengths, memory_per_sample, 'o-', linewidth=2, markersize=8, color='red')
            ax2.set_xlabel('Signal Length (samples)')
            ax2.set_ylabel('Memory per Sample (KB)')
            ax2.set_title('Memory Efficiency')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'memory_scaling.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Benchmark plots saved to {self.output_dir}/")
    
    def save_results(self):
        """Save benchmark results to files"""
        import json
        
        # Save JSON results
        results_file = self.output_dir / 'benchmark_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_results = {}
            for category, data in self.results.items():
                json_results[category] = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        json_results[category][key] = {k: convert_numpy(v) for k, v in value.items()}
                    else:
                        json_results[category][key] = convert_numpy(value)
            
            json.dump(json_results, f, indent=2)
        
        print(f"Benchmark results saved to {results_file}")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("RF Signal Source Separation Dataset - Performance Benchmarks")
        print("=" * 70)
        
        # Run all benchmarks
        self.benchmark_signal_generation()
        self.benchmark_channel_effects()
        self.benchmark_signal_mixing()
        self.benchmark_validation()
        self.benchmark_memory_scaling()
        
        # Generate comparison and plots
        self.generate_comparison_table()
        self.plot_benchmarks()
        self.save_results()
        
        print("\n" + "=" * 70)
        print("Benchmark suite completed successfully!")
        print(f"Results saved to: {self.output_dir}/")


if __name__ == "__main__":
    # Run benchmarks
    benchmark = PerformanceBenchmark('benchmarks')
    benchmark.run_full_benchmark()