"""
CNN-LSTM architecture for RF signal source separation
Based on paper claims: achieving 26.7 dB SINR improvement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution layer"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv_real = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for complex convolution
        x shape: (batch, 2, length) where dim 1 is [real, imag]
        """
        real, imag = x[:, 0:1], x[:, 1:2]
        
        # Complex multiplication: (a+bi) * (c+di) = (ac-bd) + (ad+bc)i
        out_real = self.conv_real(real) - self.conv_imag(imag)  
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        
        return torch.cat([out_real, out_imag], dim=1)


class CNNLSTMSeparator(nn.Module):
    """CNN-LSTM architecture for RF signal source separation"""
    
    def __init__(self, 
                 input_length: int = 30720,  # 10ms at 30.72MHz
                 num_standards: int = 4,     # GSM, UMTS, LTE, 5G
                 cnn_channels: List[int] = [2, 32, 64, 128],
                 lstm_hidden_size: int = 256,
                 lstm_num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_length = input_length
        self.num_standards = num_standards
        
        # CNN feature extraction layers  
        self.cnn_layers = nn.ModuleList()
        for i in range(len(cnn_channels) - 1):
            if i == 0:
                # First layer: 2 channels (real+imag) -> next channels, but process each separately
                self.cnn_layers.append(
                    nn.Sequential(
                        nn.Conv1d(2, cnn_channels[i+1], kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm1d(cnn_channels[i+1]),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
            else:
                self.cnn_layers.append(
                    nn.Sequential(
                        nn.Conv1d(cnn_channels[i], cnn_channels[i+1], 
                                kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm1d(cnn_channels[i+1]),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                )
        
        # Calculate LSTM input size after CNN layers
        cnn_output_length = input_length
        for _ in range(len(cnn_channels) - 1):
            cnn_output_length = cnn_output_length // 2  # stride=2
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers for each standard
        self.output_layers = nn.ModuleDict({
            'GSM': nn.Linear(lstm_hidden_size * 2, 2),      # Complex output
            'UMTS': nn.Linear(lstm_hidden_size * 2, 2),
            'LTE': nn.Linear(lstm_hidden_size * 2, 2),  
            '5G_NR': nn.Linear(lstm_hidden_size * 2, 2)
        })
        
        # Upsample to original length
        self.upsample_factor = input_length // cnn_output_length
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, 2, length) for complex signals
            
        Returns:
            Dictionary of separated signals for each standard
        """
        batch_size = x.size(0)
        
        # CNN feature extraction
        features = x
        for cnn_layer in self.cnn_layers:
            features = cnn_layer(features)
        
        # Reshape for LSTM: (batch, seq_len, features)
        # features shape: (batch, channels, length)
        features = features.permute(0, 2, 1)  # (batch, length, channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Generate separated signals for each standard
        separated_signals = {}
        for standard, output_layer in self.output_layers.items():
            # Apply output layer
            signal_features = output_layer(attended)  # (batch, seq_len, 2)
            
            # Upsample to original length
            signal_upsampled = F.interpolate(
                signal_features.permute(0, 2, 1),  # (batch, 2, seq_len)
                size=self.input_length,
                mode='linear',
                align_corners=False
            )
            
            separated_signals[standard] = signal_upsampled
        
        return separated_signals
    
    def compute_loss(self, predictions: dict, targets: dict, 
                    loss_weights: Optional[dict] = None) -> torch.Tensor:
        """
        Compute separation loss
        
        Args:
            predictions: Dict of predicted signals
            targets: Dict of target signals  
            loss_weights: Optional weights for each standard
            
        Returns:
            Total loss
        """
        if loss_weights is None:
            loss_weights = {std: 1.0 for std in predictions.keys()}
        
        total_loss = 0
        for standard in predictions.keys():
            if standard in targets:
                pred = predictions[standard]
                target = targets[standard]
                
                # MSE loss for complex signals
                loss = F.mse_loss(pred, target)
                total_loss += loss_weights[standard] * loss
        
        return total_loss


class SourceSeparationTrainer:
    """Trainer for CNN-LSTM source separation"""
    
    def __init__(self, model: CNNLSTMSeparator, 
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_step(self, mixed_signals: torch.Tensor, 
                  target_signals: dict) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(mixed_signals)
        
        # Compute loss
        loss = self.model.compute_loss(predictions, target_signals)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mixed_signals: torch.Tensor, 
                target_signals: dict) -> Tuple[float, dict]:
        """Evaluate model performance"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(mixed_signals)
            loss = self.model.compute_loss(predictions, target_signals)
            
            # Compute SINR for each standard
            sinr_metrics = {}
            for standard in predictions.keys():
                if standard in target_signals:
                    pred = predictions[standard]
                    target = target_signals[standard]
                    
                    # SINR = 10 * log10(signal_power / noise_power)
                    signal_power = torch.mean(target ** 2)
                    noise_power = torch.mean((pred - target) ** 2) + 1e-10
                    sinr_db = 10 * torch.log10(signal_power / noise_power)
                    sinr_metrics[standard] = sinr_db.item()
            
            return loss.item(), sinr_metrics


def create_training_data(mixed_signal: np.ndarray, 
                        component_signals: dict) -> Tuple[torch.Tensor, dict]:
    """
    Convert numpy arrays to PyTorch tensors for training
    
    Args:
        mixed_signal: Complex mixed signal array
        component_signals: Dict of individual component signals
        
    Returns:
        Tuple of (mixed_tensor, targets_dict)
    """
    # Convert complex signal to real/imaginary representation
    def complex_to_tensor(signal):
        return torch.tensor(np.stack([signal.real, signal.imag]), dtype=torch.float32)
    
    mixed_tensor = complex_to_tensor(mixed_signal).unsqueeze(0)  # Add batch dim
    
    targets_dict = {}
    for standard, signal in component_signals.items():
        targets_dict[standard] = complex_to_tensor(signal).unsqueeze(0)
    
    return mixed_tensor, targets_dict


if __name__ == "__main__":
    # Test the model
    print("Testing CNN-LSTM Source Separation Architecture")
    
    # Create model
    model = CNNLSTMSeparator(input_length=1024)  # Smaller for testing
    
    # Create dummy data
    batch_size = 2
    dummy_input = torch.randn(batch_size, 2, 1024)  # (batch, [real,imag], length)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print("Output shapes:")
    for standard, output in outputs.items():
        print(f"  {standard}: {output.shape}")
    
    print("✓ CNN-LSTM architecture test passed")