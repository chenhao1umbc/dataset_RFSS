"""
MIMO channel models and spatial processing
"""
import numpy as np
from src.utils.config_loader import get_mimo_config


class MIMOChannel:
    """MIMO channel implementation with spatial correlation"""
    
    def __init__(self, num_tx=2, num_rx=2, correlation='low'):
        """
        Initialize MIMO channel
        
        Args:
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas
            correlation: Spatial correlation level ('low', 'medium', 'high')
        """
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.correlation = correlation
        
        # Load MIMO configuration
        self.mimo_config = get_mimo_config()
        self.correlation_coeff = self.mimo_config['spatial_correlation'][correlation]
        
        # Channel matrix (num_rx x num_tx)
        self.H = None
        
    def generate_correlation_matrix(self, num_antennas, correlation_coeff):
        """Generate spatial correlation matrix"""
        R = np.zeros((num_antennas, num_antennas), dtype=complex)
        
        for i in range(num_antennas):
            for j in range(num_antennas):
                # Exponential correlation model
                R[i, j] = correlation_coeff ** abs(i - j)
        
        return R
    
    def _apply_spatial_correlation(self, H_iid):
        """Apply spatial correlation to channel matrix (shared method)"""
        if self.correlation == 'low':
            return H_iid
        
        # Generate correlation matrices (cached for efficiency)
        if not hasattr(self, '_R_tx_sqrt') or not hasattr(self, '_R_rx_sqrt'):
            R_tx = self.generate_correlation_matrix(self.num_tx, self.correlation_coeff)
            R_rx = self.generate_correlation_matrix(self.num_rx, self.correlation_coeff)
            
            # Cache Cholesky decompositions for reuse
            self._R_tx_sqrt = np.linalg.cholesky(R_tx)
            self._R_rx_sqrt = np.linalg.cholesky(R_rx)
        
        # Apply correlation: H = R_rx^(1/2) * H_iid * R_tx^(1/2)
        return self._R_rx_sqrt @ H_iid @ self._R_tx_sqrt.T
    
    def generate_channel_matrix(self, num_samples=1):
        """Generate MIMO channel matrix"""
        if num_samples == 1:
            # Single channel realization
            H_iid = (np.random.randn(self.num_rx, self.num_tx) + 
                    1j * np.random.randn(self.num_rx, self.num_tx)) / np.sqrt(2)
            
            # Apply spatial correlation using shared method
            H = self._apply_spatial_correlation(H_iid)
            self.H = H
            return H
        
        else:
            # Multiple channel realizations for time-varying channel
            H_sequence = []
            for _ in range(num_samples):
                H_iid = (np.random.randn(self.num_rx, self.num_tx) + 
                        1j * np.random.randn(self.num_rx, self.num_tx)) / np.sqrt(2)
                
                # Apply spatial correlation using shared method
                H = self._apply_spatial_correlation(H_iid)
                H_sequence.append(H)
            
            return np.array(H_sequence)
    
    def apply_channel(self, tx_signals):
        """
        Apply MIMO channel to transmitted signals
        
        Args:
            tx_signals: Transmitted signals (num_tx x num_samples)
            
        Returns:
            rx_signals: Received signals (num_rx x num_samples)
        """
        if tx_signals.ndim == 1:
            # Single antenna case
            tx_signals = tx_signals.reshape(1, -1)
        
        num_samples = tx_signals.shape[1]
        
        # Generate channel matrix
        if self.H is None:
            self.H = self.generate_channel_matrix()
        
        # Apply channel: y = H * x
        rx_signals = self.H @ tx_signals
        
        return rx_signals
    
    def get_channel_capacity(self, snr_db=10):
        """Calculate MIMO channel capacity"""
        if self.H is None:
            self.H = self.generate_channel_matrix()
        
        # SNR in linear scale
        snr_linear = 10**(snr_db/10)
        
        # Channel capacity: C = log2(det(I + (SNR/num_tx) * H * H^H))
        HH = self.H @ self.H.conj().T
        I = np.eye(self.num_rx)
        
        capacity_matrix = I + (snr_linear / self.num_tx) * HH
        capacity = np.log2(np.linalg.det(capacity_matrix)).real
        
        return capacity
    
    def get_condition_number(self):
        """Get channel matrix condition number"""
        if self.H is None:
            self.H = self.generate_channel_matrix()
        
        return np.linalg.cond(self.H)


class MIMOPreprocessing:
    """MIMO preprocessing techniques"""
    
    @staticmethod
    def zero_forcing(H):
        """Zero-forcing precoding"""
        # For square matrices: W = H^-1
        # For overdetermined: W = (H^H * H)^-1 * H^H
        
        if H.shape[0] == H.shape[1]:  # Square matrix
            W = np.linalg.inv(H)
        else:  # Overdetermined
            W = np.linalg.pinv(H)
        
        return W
    
    @staticmethod
    def mmse(H, noise_power=1.0):
        """MMSE (Minimum Mean Square Error) precoding"""
        # W = (H^H * H + noise_power * I)^-1 * H^H
        HH = H.conj().T @ H
        I = np.eye(H.shape[1])
        
        W = np.linalg.inv(HH + noise_power * I) @ H.conj().T
        
        return W
    
    @staticmethod
    def svd_precoding(H):
        """SVD-based precoding"""
        # H = U * S * V^H
        U, S, Vh = np.linalg.svd(H)
        
        # Precoding matrix is V
        # Detection matrix is U^H
        return Vh.conj().T, U.conj().T, S


class MIMOSystemSimulator:
    """Complete MIMO system simulator"""
    
    def __init__(self, num_tx=2, num_rx=2, correlation='low'):
        """Initialize MIMO system"""
        self.mimo_channel = MIMOChannel(num_tx, num_rx, correlation)
        self.num_tx = num_tx
        self.num_rx = num_rx
    
    def simulate_transmission(self, data_symbols, precoding='none', snr_db=10):
        """
        Simulate complete MIMO transmission
        
        Args:
            data_symbols: Data symbols to transmit (num_streams x num_symbols)
            precoding: Precoding method ('none', 'zf', 'mmse', 'svd')
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            received_symbols: Received symbols after channel and processing
            channel_matrix: Channel matrix used
        """
        if data_symbols.ndim == 1:
            # Single stream
            num_symbols = len(data_symbols)
            data_symbols = data_symbols.reshape(1, -1)
        else:
            num_symbols = data_symbols.shape[1]
        
        # Generate channel matrix
        H = self.mimo_channel.generate_channel_matrix()
        
        # Apply precoding
        if precoding == 'none':
            tx_signals = data_symbols
        elif precoding == 'zf':
            W = MIMOPreprocessing.zero_forcing(H)
            tx_signals = W @ data_symbols
        elif precoding == 'mmse':
            noise_power = 10**(-snr_db/10)
            W = MIMOPreprocessing.mmse(H, noise_power)
            tx_signals = W @ data_symbols
        elif precoding == 'svd':
            V, U, S = MIMOPreprocessing.svd_precoding(H)
            tx_signals = V @ data_symbols
        else:
            raise ValueError(f"Unknown precoding method: {precoding}")
        
        # Normalize transmit power
        tx_power = np.mean(np.abs(tx_signals)**2)
        if tx_power > 0:
            tx_signals = tx_signals / np.sqrt(tx_power)
        
        # Apply channel
        rx_signals = self.mimo_channel.apply_channel(tx_signals)
        
        # Add noise
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(*rx_signals.shape) + 
                                         1j * np.random.randn(*rx_signals.shape))
        rx_signals_noisy = rx_signals + noise
        
        # Detection (simplified - just return noisy received signals)
        received_symbols = rx_signals_noisy
        
        return received_symbols, H
    
    def calculate_performance_metrics(self, tx_symbols, rx_symbols):
        """Calculate performance metrics"""
        # Symbol error rate (simplified)
        if tx_symbols.shape != rx_symbols.shape:
            min_len = min(tx_symbols.shape[1], rx_symbols.shape[1])
            tx_symbols = tx_symbols[:, :min_len]
            rx_symbols = rx_symbols[:, :min_len]
        
        # Calculate MSE
        mse = np.mean(np.abs(tx_symbols - rx_symbols)**2)
        
        # Calculate SNR
        signal_power = np.mean(np.abs(tx_symbols)**2)
        noise_power = mse
        snr_measured = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        return {
            'mse': mse,
            'snr_measured': snr_measured,
            'signal_power': signal_power,
            'noise_power': noise_power
        }


if __name__ == "__main__":
    # Test MIMO system
    print("Testing MIMO system...")
    
    # Test different antenna configurations
    configs = [(2, 2), (2, 4), (4, 4)]
    
    for num_tx, num_rx in configs:
        print(f"\nTesting {num_tx}x{num_rx} MIMO...")
        
        # Create MIMO system
        mimo_sim = MIMOSystemSimulator(num_tx, num_rx, correlation='medium')
        
        # Generate test data
        num_symbols = 1000
        data_symbols = (np.random.randn(num_tx, num_symbols) + 
                       1j * np.random.randn(num_tx, num_symbols)) / np.sqrt(2)
        
        # Test different precoding methods
        for precoding in ['none', 'zf', 'mmse']:
            try:
                rx_symbols, H = mimo_sim.simulate_transmission(
                    data_symbols, precoding=precoding, snr_db=15
                )
                
                # Calculate metrics
                metrics = mimo_sim.calculate_performance_metrics(data_symbols, rx_symbols)
                
                print(f"  {precoding.upper()} precoding:")
                print(f"    MSE: {metrics['mse']:.6f}")
                print(f"    SNR: {metrics['snr_measured']:.2f} dB")
                
                # Channel properties
                capacity = mimo_sim.mimo_channel.get_channel_capacity(snr_db=15)
                cond_num = mimo_sim.mimo_channel.get_condition_number()
                
                print(f"    Capacity: {capacity:.2f} bits/s/Hz")
                print(f"    Condition number: {cond_num:.2f}")
                
            except np.linalg.LinAlgError:
                print(f"  {precoding.upper()} precoding: Singular matrix, skipped")
    
    print("MIMO system test completed")