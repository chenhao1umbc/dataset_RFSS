"""
Dual-Path RNN (DPRNN) for RF signal source separation
Based on "Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation"
Paper: https://arxiv.org/abs/1910.06379
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class DualRNN(nn.Module):
    """Dual-Path RNN block with intra-chunk and inter-chunk processing"""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Intra-chunk RNN (processes within each chunk)
        self.intra_rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Inter-chunk RNN (processes across chunks)
        self.inter_rnn = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Layer normalization
        self.intra_norm = nn.LayerNorm(hidden_size * 2)
        self.inter_norm = nn.LayerNorm(hidden_size * 2)
        
        # Linear projection layers
        self.intra_linear = nn.Linear(hidden_size * 2, input_size)
        self.inter_linear = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, input_size]
            chunk_size: Size of chunks for dual-path processing
        Returns:
            Output tensor [B, T, input_size]
        """
        B, T, N = x.shape
        
        # Padding if needed
        if T % chunk_size != 0:
            padding = chunk_size - (T % chunk_size)
            x = F.pad(x, (0, 0, 0, padding))
            T_padded = T + padding
        else:
            T_padded = T
        
        num_chunks = T_padded // chunk_size
        
        # Reshape for chunk processing: [B, num_chunks, chunk_size, N]
        x_chunks = x[:, :T_padded].view(B, num_chunks, chunk_size, N)
        
        # Intra-chunk RNN processing
        # Process each chunk independently
        intra_out = []
        for i in range(num_chunks):
            chunk = x_chunks[:, i]  # [B, chunk_size, N]
            chunk_out, _ = self.intra_rnn(chunk)  # [B, chunk_size, hidden_size*2]
            chunk_out = self.intra_norm(chunk_out)
            chunk_out = self.intra_linear(chunk_out)  # [B, chunk_size, N]
            chunk_out = chunk + chunk_out  # Residual connection
            intra_out.append(chunk_out)
        
        intra_output = torch.stack(intra_out, dim=1)  # [B, num_chunks, chunk_size, N]
        
        # Inter-chunk RNN processing
        # Process across chunks at each time step within chunks
        inter_out = []
        for t in range(chunk_size):
            # Get all chunks at time step t: [B, num_chunks, N]
            time_step = intra_output[:, :, t, :]
            step_out, _ = self.inter_rnn(time_step)  # [B, num_chunks, hidden_size*2]
            step_out = self.inter_norm(step_out)
            step_out = self.inter_linear(step_out)  # [B, num_chunks, N]
            step_out = time_step + step_out  # Residual connection
            inter_out.append(step_out)
        
        inter_output = torch.stack(inter_out, dim=2)  # [B, num_chunks, chunk_size, N]
        
        # Reshape back to original format
        output = inter_output.view(B, T_padded, N)
        
        # Remove padding
        if T_padded != T:
            output = output[:, :T]
        
        return output


class DualPathRNN(nn.Module):
    """
    Dual-Path RNN for RF signal source separation
    
    Args:
        N: Number of encoder filters
        L: Length of encoder filters  
        B: Bottleneck dimension
        H: Hidden dimension for RNN
        P: Chunk size for dual-path processing
        num_layers: Number of DPRNN blocks
        C: Number of sources
    """
    
    def __init__(self,
                 N: int = 256,
                 L: int = 20,
                 B: int = 256,
                 H: int = 128,
                 P: int = 50,
                 num_layers: int = 6,
                 C: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.N = N
        self.L = L
        self.B = B
        self.H = H
        self.P = P
        self.num_layers = num_layers
        self.C = C
        
        # Encoder: 1D conv
        self.encoder = nn.Conv1d(2, N, L, stride=L//2, bias=False)
        
        # Layer normalization after encoder
        self.ln = nn.LayerNorm(N)
        
        # Bottleneck
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)
        
        # Dual-Path RNN layers
        self.dprnn_layers = nn.ModuleList([
            DualRNN(B, H, dropout) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Conv1d(B, N * C, 1, bias=False)
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(N, 2, L, stride=L//2, bias=False)
        
    def forward(self, mixture: torch.Tensor) -> dict:
        """
        Args:
            mixture: [B, 2, T] mixed signal (real+imag)
        Returns:
            Dictionary of separated sources
        """
        B, _, T = mixture.shape
        
        # Encoding
        encoded = self.encoder(mixture)  # [B, N, T']
        T_encoded = encoded.shape[-1]
        
        # Layer normalization
        encoded = encoded.transpose(1, 2)  # [B, T', N]
        encoded = self.ln(encoded)
        encoded = encoded.transpose(1, 2)  # [B, N, T']
        
        # Bottleneck
        bottleneck = self.bottleneck(encoded)  # [B, B, T']
        
        # DPRNN processing
        # Convert to [B, T', B] for RNN processing
        x = bottleneck.transpose(1, 2)  # [B, T', B]
        
        for dprnn_layer in self.dprnn_layers:
            x = dprnn_layer(x, chunk_size=self.P)
        
        # Convert back to [B, B, T']
        x = x.transpose(1, 2)
        
        # Output layer
        masks = self.output_layer(x)  # [B, N*C, T']
        masks = masks.view(B, self.C, self.N, T_encoded)  # [B, C, N, T']
        masks = torch.sigmoid(masks)
        
        # Apply masks
        masked = encoded.unsqueeze(1) * masks  # [B, C, N, T']
        
        # Decoding
        separated_sources = {}
        source_names = ['GSM', 'UMTS', 'LTE', '5G_NR']
        
        for c in range(self.C):
            if c < len(source_names):
                source_name = source_names[c]
            else:
                source_name = f'Source_{c}'
            
            decoded = self.decoder(masked[:, c])  # [B, 2, T_decoded]
            
            # Trim or pad to match input length
            if decoded.size(-1) > T:
                decoded = decoded[:, :, :T]
            elif decoded.size(-1) < T:
                padding = T - decoded.size(-1)
                decoded = F.pad(decoded, (0, padding))
            
            separated_sources[source_name] = decoded
        
        return separated_sources
    
    def compute_loss(self, predictions: dict, targets: dict) -> torch.Tensor:
        """Compute SI-SNR loss"""
        total_loss = 0
        count = 0
        
        for source_name in predictions.keys():
            if source_name in targets:
                pred = predictions[source_name]
                target = targets[source_name]
                
                # SI-SNR loss
                loss = self._si_snr_loss(pred, target)
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else total_loss
    
    def _si_snr_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Scale-Invariant SNR loss"""
        # Flatten
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Zero mean
        pred = pred - pred.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        
        # SI-SNR
        s_target = ((pred * target).sum(dim=1, keepdim=True) * target) / (target ** 2).sum(dim=1, keepdim=True)
        e_noise = pred - s_target
        
        si_snr = 10 * torch.log10((s_target ** 2).sum(dim=1) / ((e_noise ** 2).sum(dim=1) + 1e-8))
        return -si_snr.mean()  # Negative because we want to maximize SI-SNR


class DPRNNTrainer:
    """Trainer for Dual-Path RNN"""
    
    def __init__(self, model: DualPathRNN,
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
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            mixed_signals = mixed_signals.to(self.device)
            target_signals = {k: v.to(self.device) for k, v in target_signals.items()}
            
            predictions = self.model(mixed_signals)
            loss = self.model.compute_loss(predictions, target_signals)
            
            # Compute SI-SNR metrics
            si_snr_metrics = {}
            for source_name in predictions.keys():
                if source_name in target_signals:
                    pred = predictions[source_name].view(mixed_signals.size(0), -1)
                    target = target_signals[source_name].view(mixed_signals.size(0), -1)
                    
                    # SI-SNR calculation
                    pred_zm = pred - pred.mean(dim=1, keepdim=True)
                    target_zm = target - target.mean(dim=1, keepdim=True)
                    
                    s_target = ((pred_zm * target_zm).sum(dim=1, keepdim=True) * target_zm) / (target_zm ** 2).sum(dim=1, keepdim=True)
                    e_noise = pred_zm - s_target
                    
                    si_snr = 10 * torch.log10((s_target ** 2).sum(dim=1) / ((e_noise ** 2).sum(dim=1) + 1e-8))
                    si_snr_metrics[source_name] = si_snr.mean().item()
            
            return {
                'loss': loss.item(),
                'si_snr_metrics': si_snr_metrics,
                'mean_si_snr': np.mean(list(si_snr_metrics.values())) if si_snr_metrics else 0.0
            }


if __name__ == "__main__":
    # Test Dual-Path RNN
    print("Testing Dual-Path RNN Architecture")
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = DualPathRNN(N=128, B=128, H=64, P=25, num_layers=4, C=2).to(device)  # Smaller for testing
    
    # Create dummy data
    batch_size = 2
    n_samples = 4000  # 4000 samples
    dummy_input = torch.randn(batch_size, 2, n_samples).to(device)
    
    # Forward pass
    outputs = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print("Output shapes:")
    for source_name, output in outputs.items():
        print(f"  {source_name}: {output.shape}")
    
    # Test training step
    trainer = DPRNNTrainer(model, device=device)
    
    # Dummy targets
    targets = {}
    for source_name in outputs.keys():
        targets[source_name] = torch.randn_like(outputs[source_name])
    
    loss = trainer.train_step(dummy_input, targets)
    print(f"Training loss: {loss:.4f}")
    
    print("✓ Dual-Path RNN architecture test passed")