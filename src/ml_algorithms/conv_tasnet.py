"""
Conv-TasNet: Convolutional Time-domain Audio Source Separation Network
Adapted for RF signal source separation
Original paper: https://arxiv.org/abs/1809.07454

WARNING: EXPERIMENTAL CODE - TRAINING INSTABILITY ISSUES
This implementation is under active development and currently exhibits:
- Tensor shape incompatibility errors during training
- Unvalidated separation performance
- May require architecture modifications for RF signals

Use for research and development only. Contributions welcome.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization for Conv-TasNet"""
    
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))
        
    def forward(self, y):
        """
        Args:
            y: [B, N, T] tensor
        """
        mean = y.mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        var = ((y - mean) ** 2).mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        y_norm = (y - mean) / (var + 1e-8).sqrt()  # [B, N, T]
        y_norm = self.gamma * y_norm + self.beta  # [B, N, T]
        return y_norm


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1D convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=False):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, dilation=dilation, bias=bias
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network (TCN) block"""
    
    def __init__(self, in_channels, hidden_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # 1D Conv layers
        self.conv1x1_in = nn.Conv1d(in_channels, hidden_channels, 1)
        
        self.depthwise_conv = DepthwiseSeparableConv1d(
            hidden_channels, hidden_channels, kernel_size, dilation=dilation
        )
        
        self.norm = GlobalLayerNorm(hidden_channels)
        self.activation = nn.PReLU()
        
        self.conv1x1_out = nn.Conv1d(hidden_channels, in_channels + skip_channels, 1)
        
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T]
        Returns:
            residual: [B, in_channels, T]
            skip: [B, skip_channels, T] 
        """
        y = self.conv1x1_in(x)
        
        # Apply padding for causal convolution
        padding = (self.kernel_size - 1) * self.dilation
        y = F.pad(y, (padding, 0))
        
        y = self.depthwise_conv(y)
        y = self.norm(y)
        y = self.activation(y)
        
        y = self.conv1x1_out(y)
        
        # Split residual and skip connections
        residual, skip = y.split([x.size(1), y.size(1) - x.size(1)], dim=1)
        
        return x + residual, skip


class ConvTasNet(nn.Module):
    """
    Conv-TasNet for RF signal source separation
    
    Args:
        N: Number of filters in autoencoder (default: 512)
        L: Length of filters in autoencoder (default: 16)
        B: Bottleneck dimension (default: 128)
        H: Hidden dimension (default: 512)
        P: Kernel size in conv blocks (default: 3)
        X: Number of conv blocks in each repeat (default: 8)
        R: Number of repeats (default: 3)
        C: Number of sources (default: 4)
        norm_type: Normalization type (default: 'gLN')
    """
    
    def __init__(self, 
                 N: int = 512,
                 L: int = 16,
                 B: int = 128,
                 H: int = 512,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 C: int = 4,
                 norm_type: str = 'gLN'):
        super().__init__()
        
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.C = C
        
        # Encoder: 1D conv to transform waveform to representation
        self.encoder = nn.Conv1d(2, N, L, stride=L//2)  # 2 channels for complex input (real+imag)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(N, B, 1)
        
        # TCN separator
        self.tcn = self._build_tcn()
        
        # Mask generation
        self.mask_net = nn.Conv1d(B, N * C, 1)
        
        # Decoder: transposed conv to reconstruct waveform
        self.decoder = nn.ConvTranspose1d(N, 2, L, stride=L//2)
        
    def _build_tcn(self):
        """Build the Temporal Convolutional Network"""
        tcn_blocks = nn.ModuleList()
        
        for r in range(self.R):
            for x in range(self.X):
                dilation = 2 ** x
                tcn_blocks.append(
                    TemporalConvNet(
                        in_channels=self.B,
                        hidden_channels=self.H,
                        skip_channels=self.B,
                        kernel_size=self.P,
                        dilation=dilation
                    )
                )
        
        return tcn_blocks
    
    def forward(self, mixture):
        """
        Args:
            mixture: [B, 2, T] complex waveform (real+imag channels)
            
        Returns:
            separated_sources: dict with separated sources
        """
        batch_size, n_channels, n_samples = mixture.shape
        
        # Encode
        encoded = self.encoder(mixture)  # [B, N, T']
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)  # [B, B, T']
        
        # TCN processing
        output = bottleneck
        skip_connections = 0
        
        for tcn_block in self.tcn:
            output, skip = tcn_block(output)
            skip_connections = skip_connections + skip
        
        # Generate masks
        masks = self.mask_net(skip_connections)  # [B, N*C, T']
        masks = masks.view(batch_size, self.C, self.N, -1)  # [B, C, N, T']
        masks = torch.sigmoid(masks)
        
        # Apply masks to encoded representation
        masked_encoded = encoded.unsqueeze(1) * masks  # [B, C, N, T']
        
        # Decode each source
        separated_sources = {}
        source_names = ['GSM', 'UMTS', 'LTE', '5G_NR']
        
        for c in range(self.C):
            if c < len(source_names):
                source_name = source_names[c]
            else:
                source_name = f'Source_{c}'
            
            decoded = self.decoder(masked_encoded[:, c])  # [B, 2, T]
            
            # Ensure same length as input
            if decoded.size(-1) > n_samples:
                decoded = decoded[:, :, :n_samples]
            elif decoded.size(-1) < n_samples:
                padding = n_samples - decoded.size(-1)
                decoded = F.pad(decoded, (0, padding))
            
            separated_sources[source_name] = decoded
        
        return separated_sources
    
    def compute_loss(self, predictions: dict, targets: dict, 
                    loss_type: str = 'si_snr') -> torch.Tensor:
        """
        Compute separation loss
        
        Args:
            predictions: Dict of predicted signals
            targets: Dict of target signals
            loss_type: 'si_snr' or 'mse'
            
        Returns:
            Total loss
        """
        if loss_type == 'si_snr':
            return self._si_snr_loss(predictions, targets)
        else:
            return self._mse_loss(predictions, targets)
    
    def _si_snr_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Scale-Invariant Signal-to-Noise Ratio loss"""
        total_loss = 0
        count = 0
        
        for source_name in predictions.keys():
            if source_name in targets:
                pred = predictions[source_name]  # [B, 2, T]
                target = targets[source_name]    # [B, 2, T]
                
                # Flatten to [B, -1] for SI-SNR computation
                pred_flat = pred.view(pred.size(0), -1)
                target_flat = target.view(target.size(0), -1)
                
                # Compute SI-SNR
                si_snr = self._compute_si_snr(pred_flat, target_flat)
                total_loss = total_loss - si_snr.mean()  # Negative because we want to maximize SI-SNR
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss
    
    def _mse_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Mean Squared Error loss"""
        total_loss = 0
        count = 0
        
        for source_name in predictions.keys():
            if source_name in targets:
                pred = predictions[source_name]
                target = targets[source_name]
                loss = F.mse_loss(pred, target)
                total_loss += loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return total_loss
    
    def _compute_si_snr(self, pred, target):
        """Compute Scale-Invariant SNR"""
        # Zero mean
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        # Optimal scaling factor
        dot_product = (pred * target).sum(dim=1, keepdim=True)
        target_energy = (target ** 2).sum(dim=1, keepdim=True)
        alpha = dot_product / (target_energy + 1e-8)
        
        # Scaled target
        scaled_target = alpha * target
        
        # SI-SNR computation
        signal_energy = (scaled_target ** 2).sum(dim=1)
        noise_energy = ((pred - scaled_target) ** 2).sum(dim=1)
        
        si_snr = 10 * torch.log10(signal_energy / (noise_energy + 1e-8) + 1e-8)
        
        return si_snr


class ConvTasNetTrainer:
    """Trainer for Conv-TasNet"""
    
    def __init__(self, model: ConvTasNet, 
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train_step(self, mixed_signals: torch.Tensor, 
                  target_signals: dict) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move to device
        mixed_signals = mixed_signals.to(self.device)
        target_signals = {k: v.to(self.device) for k, v in target_signals.items()}
        
        # Forward pass
        predictions = self.model(mixed_signals)
        
        # Compute loss
        loss = self.model.compute_loss(predictions, target_signals)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, mixed_signals: torch.Tensor, 
                target_signals: dict) -> dict:
        """Evaluate model performance"""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            mixed_signals = mixed_signals.to(self.device)
            target_signals = {k: v.to(self.device) for k, v in target_signals.items()}
            
            predictions = self.model(mixed_signals)
            loss = self.model.compute_loss(predictions, target_signals)
            
            # Compute SI-SNR for each source
            si_snr_metrics = {}
            for source_name in predictions.keys():
                if source_name in target_signals:
                    pred = predictions[source_name]
                    target = target_signals[source_name]
                    
                    # Flatten and compute SI-SNR
                    pred_flat = pred.view(pred.size(0), -1)
                    target_flat = target.view(target.size(0), -1)
                    
                    si_snr = self.model._compute_si_snr(pred_flat, target_flat)
                    si_snr_metrics[source_name] = si_snr.mean().item()
            
            return {
                'loss': loss.item(),
                'si_snr_metrics': si_snr_metrics,
                'mean_si_snr': np.mean(list(si_snr_metrics.values())) if si_snr_metrics else 0.0
            }


if __name__ == "__main__":
    # Test Conv-TasNet
    print("Testing Conv-TasNet Architecture")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = ConvTasNet(N=256, L=16, B=128, H=256, C=2).to(device)  # Smaller for testing
    
    # Create dummy data
    batch_size = 4
    n_samples = 8000  # 0.25s at 32kHz equivalent
    dummy_input = torch.randn(batch_size, 2, n_samples).to(device)  # Complex signal
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print("Output shapes:")
    for source_name, output in outputs.items():
        print(f"  {source_name}: {output.shape}")
    
    # Test training step
    trainer = ConvTasNetTrainer(model, device=device)
    
    # Dummy targets
    targets = {}
    for source_name in outputs.keys():
        targets[source_name] = torch.randn_like(outputs[source_name])
    
    loss = trainer.train_step(dummy_input, targets)
    print(f"Training loss: {loss:.4f}")
    
    print("✓ Conv-TasNet architecture test passed")