#!/usr/bin/env python3
"""
Test MPS vs CPU performance for PyTorch training
Benchmark CNN-LSTM architecture on Apple Silicon
"""
import time
import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import torch
    import torch.nn as nn
    from ml_algorithms.cnn_lstm import CNNLSTMSeparator
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    print("PyTorch not installed - installing via uv...")


def install_pytorch_with_uv():
    """Install PyTorch using uv"""
    import subprocess
    
    print("Installing PyTorch with MPS support...")
    try:
        # Install PyTorch for macOS with MPS support
        subprocess.run([
            "uv", "add", "torch", "torchvision", "torchaudio"
        ], check=True)
        
        print("PyTorch installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install PyTorch: {e}")
        return False


def check_mps_availability():
    """Check if MPS is available and functional"""
    print("=== MPS Availability Check ===")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        return False
    
    if not torch.backends.mps.is_built():
        print("❌ MPS not built in this PyTorch version")
        return False
    
    print("✅ MPS is available and built")
    
    # Test basic MPS functionality
    try:
        device = torch.device('mps')
        test_tensor = torch.randn(100, 100, device=device)
        result = torch.matmul(test_tensor, test_tensor.t())
        print("✅ MPS basic operations working")
        return True
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False


def benchmark_device(device_name: str, num_iterations: int = 10) -> dict:
    """Benchmark training on specified device"""
    print(f"\n=== Benchmarking {device_name.upper()} ===")
    
    device = torch.device(device_name)
    
    # Create model and move to device
    model = CNNLSTMSeparator(input_length=1024, num_standards=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Create dummy data
    batch_size = 8
    input_data = torch.randn(batch_size, 2, 1024, device=device)
    
    # Target signals for 2 standards
    targets = {
        'GSM': torch.randn(batch_size, 2, 1024, device=device),
        'LTE': torch.randn(batch_size, 2, 1024, device=device)
    }
    
    # Warmup runs
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Sync for accurate timing
    if device_name == 'mps':
        torch.mps.synchronize()
    elif device_name == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    print(f"Running {num_iterations} benchmark iterations...")
    times = []
    memory_usage = []
    
    for i in range(num_iterations):
        # Record memory before
        if device_name == 'mps':
            mem_allocated = torch.mps.current_allocated_memory() / 1024**2
        else:
            mem_allocated = 0  # CPU memory not easily tracked
        
        start_time = time.time()
        
        # Training step
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = model.compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Sync before timing
        if device_name == 'mps':
            torch.mps.synchronize()
        elif device_name == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        step_time = end_time - start_time
        times.append(step_time)
        memory_usage.append(mem_allocated)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {step_time*1000:.1f}ms")
    
    return {
        'device': device_name,
        'mean_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'min_time_ms': np.min(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'mean_memory_mb': np.mean(memory_usage) if memory_usage[0] > 0 else 0,
        'total_params': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    }


def compare_devices():
    """Compare CPU vs MPS performance"""
    print("PyTorch Device Performance Comparison")
    print("=" * 60)
    
    results = []
    
    # Test CPU
    try:
        cpu_results = benchmark_device('cpu', num_iterations=10)
        results.append(cpu_results)
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
    
    # Test MPS if available
    if check_mps_availability():
        try:
            mps_results = benchmark_device('mps', num_iterations=10)
            results.append(mps_results)
        except Exception as e:
            print(f"MPS benchmark failed: {e}")
    
    # Print comparison
    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Device':<10} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10}")
        print("-" * 60)
        
        cpu_time = None
        for result in results:
            device = result['device'].upper()
            mean_time = result['mean_time_ms']
            std_time = result['std_time_ms']
            memory = result['mean_memory_mb']
            
            if device == 'CPU':
                cpu_time = mean_time
                speedup = '1.0×'
            else:
                speedup = f"{cpu_time/mean_time:.1f}×" if cpu_time else 'N/A'
            
            time_str = f"{mean_time:.1f}±{std_time:.1f}"
            memory_str = f"{memory:.1f}" if memory > 0 else "N/A"
            
            print(f"{device:<10} {time_str:<12} {memory_str:<12} {speedup:<10}")
        
        # Recommendations
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        
        if len(results) >= 2 and results[1]['device'] == 'mps':
            mps_time = results[1]['mean_time_ms']
            cpu_time = results[0]['mean_time_ms']
            speedup = cpu_time / mps_time
            
            if speedup > 2.0:
                print(f"🚀 MPS shows significant speedup ({speedup:.1f}×)")
                print("   Recommendation: Use MPS for large-scale training")
                return 'mps'
            elif speedup > 1.2:
                print(f"✅ MPS shows moderate speedup ({speedup:.1f}×)")
                print("   Recommendation: Use MPS for training")
                return 'mps'
            else:
                print(f"⚠️ MPS speedup is minimal ({speedup:.1f}×)")
                print("   Recommendation: CPU may be sufficient for small models")
                return 'cpu'
    
    else:
        print("\n❌ Could not compare devices - using CPU as fallback")
        return 'cpu'


def test_large_batch_scaling():
    """Test how performance scales with batch size"""
    if not check_mps_availability():
        print("Skipping batch scaling test - MPS not available")
        return
    
    print("\n=== Testing Batch Size Scaling ===")
    
    batch_sizes = [1, 2, 4, 8, 16]
    device = torch.device('mps')
    
    print(f"{'Batch Size':<12} {'Time (ms)':<12} {'Memory (MB)':<12} {'Throughput':<12}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        try:
            model = CNNLSTMSeparator(input_length=512, num_standards=2).to(device)
            input_data = torch.randn(batch_size, 2, 512, device=device)
            
            # Warmup
            for _ in range(3):
                _ = model(input_data)
            
            torch.mps.synchronize()
            
            # Benchmark
            times = []
            for _ in range(5):
                start = time.time()
                outputs = model(input_data)
                torch.mps.synchronize()
                times.append(time.time() - start)
            
            mean_time = np.mean(times) * 1000
            memory = torch.mps.current_allocated_memory() / 1024**2
            throughput = batch_size / (mean_time / 1000)
            
            print(f"{batch_size:<12} {mean_time:<8.1f}{'':<4} {memory:<8.1f}{'':<4} {throughput:<8.1f}{' samples/s':<4}")
            
            # Clean up
            del model, input_data
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"{batch_size:<12} Failed: {str(e)[:40]}...")


def main():
    """Main performance testing function"""
    if not HAS_PYTORCH:
        print("Installing PyTorch...")
        if install_pytorch_with_uv():
            print("Please restart the script after PyTorch installation")
            return
        else:
            print("Failed to install PyTorch")
            return
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    # Main comparison
    recommended_device = compare_devices()
    
    # Batch scaling test
    test_large_batch_scaling()
    
    print(f"\n✅ Performance testing completed")
    print(f"💡 Recommended device for training: {recommended_device.upper()}")
    
    return recommended_device


if __name__ == "__main__":
    recommended_device = main()