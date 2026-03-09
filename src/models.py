"""Neural network models for RF source separation.

Implements three architectures for separating mixed RF signals into
individual source estimates using Permutation Invariant Training (PIT)
with Scale-Invariant Signal-to-Noise Ratio (SI-SINR) loss.

All models:
- Accept complex RF signals as 2-channel real (real+imaginary), shape (B, 2, T)
- Output n_sources estimates, shape (B, n_sources, 2, T)
- Are trained with PIT-SI-SINR loss (permutation invariant, no fixed label assignment)

References:
- Conv-TasNet: Luo & Mesgarani, "Conv-TasNet: Surpassing ideal time-frequency
  magnitude masking for speech separation", IEEE TASLP 2019
- Dual-Path RNN: Luo et al., "Dual-path RNN: efficient long sequence modeling
  for time-domain single-channel speech separation", ICASSP 2020
- SI-SINR: Le Roux et al., "SDR - Half-baked or Well Done?", ICASSP 2019
"""

from itertools import permutations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sinr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample SI-SINR for batched 1D signals.

    Computes Scale-Invariant Signal-to-Noise Ratio by projecting the estimate
    onto the target, then measuring the power ratio of the projected signal
    versus the residual error.

    Args:
        estimate: Estimated signal, shape (B, T) real-valued.
        target: Ground-truth signal, shape (B, T) real-valued.

    Returns:
        SI-SINR in dB, shape (B,).
    """
    eps = 1e-8
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Scale-invariant projection: s_target = <s_hat, s> / <s, s> * s
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_power = (target * target).sum(dim=-1, keepdim=True) + eps
    s_target = dot / target_power * target

    # Noise residual
    e_noise = estimate - s_target

    # SI-SINR in dB
    signal_power = (s_target * s_target).sum(dim=-1)
    noise_power = (e_noise * e_noise).sum(dim=-1) + eps
    return 10.0 * torch.log10(signal_power / noise_power + eps)


def pit_si_sinr_loss(estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Permutation Invariant Training loss using SI-SINR.

    Evaluates all C! permutations and picks the best (highest mean SI-SINR)
    per sample, then returns the negative mean across the batch.

    Args:
        estimates: Separated source estimates, shape (B, C, 2, T).
        targets: Ground-truth sources, shape (B, C, 2, T).

    Returns:
        Scalar loss: negative mean of per-sample best-permutation SI-SINR.
    """
    B, C, _, T = estimates.shape

    # Flatten (2, T) -> (2T,) for SI-SINR
    est_flat = estimates.reshape(B, C, -1)  # (B, C, 2T)
    tgt_flat = targets.reshape(B, C, -1)    # (B, C, 2T)

    # Build C x C SI-SINR matrix for each batch item
    # sisnr_matrix[b, i, j] = SI-SINR(estimate i, target j)
    sisnr_matrix = torch.zeros(B, C, C, device=estimates.device)
    for i in range(C):
        for j in range(C):
            sisnr_matrix[:, i, j] = si_sinr(est_flat[:, i, :], tgt_flat[:, j, :])

    # Evaluate all C! permutations, find per-sample best
    perms = list(permutations(range(C)))
    perm_scores = torch.stack(
        [sisnr_matrix[:, range(C), list(p)].mean(dim=-1) for p in perms],
        dim=-1
    )  # (B, n_perms)

    best_scores = perm_scores.max(dim=-1).values  # (B,)
    return -best_scores.mean()


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization over both channel and time dimensions.

    Normalizes over dims (1, 2) — channels and time jointly — then
    applies learnable scale (gamma) and shift (beta).

    Args:
        channel_size: Number of channels to normalize.
    """

    def __init__(self, channel_size: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channel_size, 1))
        self.beta = nn.Parameter(torch.zeros(1, channel_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / (var + 1e-8).sqrt() + self.beta


class _TCNBlock(nn.Module):
    """Single dilated temporal convolutional block used in Conv-TasNet.

    Each block applies a 1x1 conv (bottleneck expansion), a causal depthwise
    conv with exponentially increasing dilation, Global Layer Normalization,
    and PReLU. Two parallel 1x1 convs produce residual and skip outputs.

    Args:
        in_channels: Input channel count (B in Conv-TasNet notation).
        hidden_channels: Expanded channel count (H in Conv-TasNet notation).
        kernel_size: Depthwise conv kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.conv1x1 = nn.Conv1d(in_channels, hidden_channels, 1)
        padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size, dilation=dilation,
            padding=padding, groups=hidden_channels
        )
        self.norm = GlobalLayerNorm(hidden_channels)
        self.prelu = nn.PReLU()
        self.res_conv = nn.Conv1d(hidden_channels, in_channels, 1)
        self.skip_conv = nn.Conv1d(hidden_channels, in_channels, 1)
        self.causal_padding = padding

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv1x1(x)
        h = self.depthwise(h)
        # Remove the right side of causal padding
        h = h[:, :, :x.shape[-1]]
        h = self.norm(h)
        h = self.prelu(h)
        residual = self.res_conv(h)
        skip = self.skip_conv(h)
        return x + residual, skip


class ConvTasNet(nn.Module):
    """Conv-TasNet for RF source separation.

    Encoder-separator-decoder architecture:
    - Encoder: Conv1d(2, N, L, stride=L//2, bias=False)
    - Layer norm + Bottleneck Conv1d(N, B, 1)
    - TCN separator: R repeats x X dilated blocks (dilation = 2^x)
    - Mask: Conv1d(B, N*n_sources, 1) -> sigmoid -> (B, n_sources, N, T')
    - Decoder: shared ConvTranspose1d(N, 2, L, stride=L//2, bias=False)

    Args:
        N: Encoder filter count (default 256).
        L: Encoder filter length (default 16); must be even.
        B: Bottleneck channels (default 128).
        H: TCN hidden channels (default 256).
        P: TCN kernel size (default 3).
        X: TCN blocks per repeat (default 8).
        R: Number of TCN repeats (default 3).
        n_sources: Number of sources to separate (default 2).
    """

    def __init__(
        self,
        N: int = 256,
        L: int = 16,
        B: int = 128,
        H: int = 256,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        n_sources: int = 2,
    ):
        super().__init__()
        self.N = N
        self.L = L
        self.n_sources = n_sources

        self.encoder = nn.Conv1d(2, N, L, stride=L // 2, bias=False)
        self.layer_norm = nn.GroupNorm(1, N)
        self.bottleneck = nn.Conv1d(N, B, 1)

        self.tcn = nn.ModuleList([
            _TCNBlock(B, H, P, dilation=2 ** x)
            for _ in range(R)
            for x in range(X)
        ])

        self.mask_conv = nn.Conv1d(B, N * n_sources, 1)
        self.decoder = nn.ConvTranspose1d(N, 2, L, stride=L // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, _, T = x.shape

        # Encode
        encoded = self.encoder(x)  # (B, N, T')
        T_enc = encoded.shape[-1]

        # Layer norm + bottleneck
        h = self.layer_norm(encoded)
        h = self.bottleneck(h)  # (B, B_ch, T')

        # TCN with skip accumulation
        skip_sum = torch.zeros_like(h)
        for block in self.tcn:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        # Mask
        masks = self.mask_conv(skip_sum)              # (B, N*n_sources, T')
        masks = masks.view(B_batch, self.n_sources, self.N, T_enc)
        masks = torch.sigmoid(masks)

        # Apply masks to encoded
        encoded_exp = encoded.unsqueeze(1)            # (B, 1, N, T')
        masked = encoded_exp * masks                  # (B, n_sources, N, T')

        # Decode each source
        outputs = []
        for i in range(self.n_sources):
            dec = self.decoder(masked[:, i, :, :])    # (B, 2, T_decoded)
            # Trim or pad to original length T
            if dec.shape[-1] > T:
                dec = dec[:, :, :T]
            elif dec.shape[-1] < T:
                dec = F.pad(dec, (0, T - dec.shape[-1]))
            outputs.append(dec)

        return torch.stack(outputs, dim=1)            # (B, n_sources, 2, T)


class CNNLSTMSeparator(nn.Module):
    """CNN-BiLSTM for RF source separation.

    CNN downsampling encoder, bidirectional LSTM temporal modeling,
    transposed-conv upsampling decoder outputting n_sources signals.

    Args:
        n_sources: Number of sources (default 2).
        cnn_channels: Channel progression [2, 64, 128, 256] (including input).
        lstm_hidden: LSTM hidden size (default 256).
        lstm_layers: LSTM layers (default 2).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        n_sources: int = 2,
        cnn_channels: List[int] = None,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [2, 64, 128, 256]

        self.n_sources = n_sources

        # CNN encoder
        encoder_layers = []
        for i in range(len(cnn_channels) - 1):
            encoder_layers.append(
                nn.Conv1d(cnn_channels[i], cnn_channels[i + 1], 7, stride=2, padding=3)
            )
            encoder_layers.append(nn.BatchNorm1d(cnn_channels[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        lstm_dropout = dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        # Output head: 2 channels per source (real+imag)
        self.output_conv = nn.Conv1d(lstm_hidden * 2, n_sources * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, _, T = x.shape

        # CNN encode
        h = self.encoder(x)          # (B, C, T_down)

        # BiLSTM
        h = h.transpose(1, 2)        # (B, T_down, C)
        h, _ = self.lstm(h)          # (B, T_down, 2*lstm_hidden)
        h = h.transpose(1, 2)        # (B, 2*lstm_hidden, T_down)

        # Output projection
        h = self.output_conv(h)      # (B, n_sources*2, T_down)

        # Upsample to original length
        h = F.interpolate(h, size=T, mode='linear', align_corners=False)

        # Reshape to (B, n_sources, 2, T)
        h = h.view(B_batch, self.n_sources, 2, T)
        return h


class _DualRNNBlock(nn.Module):
    """Single Dual-Path RNN block with intra- and inter-chunk BiLSTM processing.

    Applies intra-chunk (local) and inter-chunk (global) recurrent processing
    with residual connections, as described in Luo et al., ICASSP 2020.

    Args:
        channels: Feature channel count (B in DPRNN notation).
        hidden_size: RNN hidden size (H in DPRNN notation).
    """

    def __init__(self, channels: int, hidden_size: int):
        super().__init__()
        self.intra_rnn = nn.LSTM(channels, hidden_size, bidirectional=True, batch_first=True)
        self.inter_rnn = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.intra_norm = nn.LayerNorm(hidden_size * 2)
        self.inter_norm = nn.LayerNorm(hidden_size * 2)
        self.intra_linear = nn.Linear(hidden_size * 2, channels)
        self.inter_linear = nn.Linear(hidden_size * 2, channels)

    def forward(self, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B_batch, T_len, C = x.shape

        # Pad T to be divisible by chunk_size
        pad_len = (chunk_size - T_len % chunk_size) % chunk_size
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        T_pad = x_pad.shape[1]
        n_chunks = T_pad // chunk_size

        # Reshape: (B, n_chunks, chunk_size, C)
        x_chunks = x_pad.reshape(B_batch, n_chunks, chunk_size, C)

        # Intra-chunk: process each chunk independently
        # Merge batch and n_chunks dims: (B*n_chunks, chunk_size, C)
        intra_in = x_chunks.reshape(B_batch * n_chunks, chunk_size, C)
        intra_h, _ = self.intra_rnn(intra_in)                     # (B*n_chunks, chunk_size, H*2)
        intra_h = self.intra_norm(intra_h)                         # (B*n_chunks, chunk_size, H*2)
        intra_res = self.intra_linear(intra_h)                     # (B*n_chunks, chunk_size, C)
        intra_res = intra_res.reshape(B_batch, n_chunks, chunk_size, C)
        x_chunks = x_chunks + intra_res

        # Inter-chunk: each position across chunks uses the pre-linear intra output (H*2)
        # Reshape intra_h: (B*n_chunks, chunk_size, H*2) -> (B, n_chunks, chunk_size, H*2)
        H2 = intra_h.shape[-1]
        intra_h = intra_h.reshape(B_batch, n_chunks, chunk_size, H2)
        # Transpose to (B, chunk_size, n_chunks, H*2) for inter processing
        inter_in = intra_h.permute(0, 2, 1, 3)                    # (B, chunk_size, n_chunks, H*2)
        inter_in = inter_in.reshape(B_batch * chunk_size, n_chunks, H2)
        inter_h, _ = self.inter_rnn(inter_in)                      # (B*chunk_size, n_chunks, H*2)
        inter_h = self.inter_norm(inter_h)
        inter_res = self.inter_linear(inter_h)                     # (B*chunk_size, n_chunks, C)
        inter_res = inter_res.reshape(B_batch, chunk_size, n_chunks, C)
        inter_res = inter_res.permute(0, 2, 1, 3)                  # (B, n_chunks, chunk_size, C)
        x_chunks = x_chunks + inter_res

        # Reshape back and remove padding
        out = x_chunks.reshape(B_batch, T_pad, C)
        return out[:, :T_len, :]


class DualPathRNN(nn.Module):
    """Dual-Path RNN for RF source separation.

    Based on Luo et al., "Dual-path RNN: efficient long sequence modeling
    for time-domain single-channel speech separation", ICASSP 2020.

    Args:
        N: Encoder filters (default 64).
        L: Encoder filter length (default 16).
        B: Bottleneck channels (default 64).
        H: RNN hidden size (default 64).
        P: Chunk size (default 50).
        num_layers: DPRNN blocks (default 6).
        n_sources: Number of sources (default 2).
        dropout: Dropout rate (default 0.0).
    """

    def __init__(
        self,
        N: int = 64,
        L: int = 16,
        B: int = 64,
        H: int = 64,
        P: int = 50,
        num_layers: int = 6,
        n_sources: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.N = N
        self.L = L
        self.P = P
        self.n_sources = n_sources

        self.encoder = nn.Conv1d(2, N, L, stride=L // 2, bias=False)
        self.layer_norm = nn.LayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1)

        self.dprnn_blocks = nn.ModuleList([
            _DualRNNBlock(B, H) for _ in range(num_layers)
        ])

        self.mask_conv = nn.Conv1d(B, N * n_sources, 1)
        self.decoder = nn.ConvTranspose1d(N, 2, L, stride=L // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_batch, _, T = x.shape

        # Encode
        encoded = self.encoder(x)          # (B, N, T')
        T_enc = encoded.shape[-1]

        # LayerNorm over channel dim (transpose for LayerNorm)
        h = encoded.transpose(1, 2)        # (B, T', N)
        h = self.layer_norm(h)
        h = h.transpose(1, 2)              # (B, N, T')

        # Bottleneck
        h = self.bottleneck(h)             # (B, B_ch, T')

        # DPRNN blocks (operate in (B, T', B_ch) layout)
        h = h.transpose(1, 2)             # (B, T', B_ch)
        for block in self.dprnn_blocks:
            h = block(h, self.P)
        h = h.transpose(1, 2)             # (B, B_ch, T')

        # Mask
        masks = self.mask_conv(h)                           # (B, N*n_sources, T')
        masks = masks.view(B_batch, self.n_sources, self.N, T_enc)
        masks = torch.sigmoid(masks)

        # Apply masks
        encoded_exp = encoded.unsqueeze(1)                  # (B, 1, N, T')
        masked = encoded_exp * masks                        # (B, n_sources, N, T')

        # Decode each source
        outputs = []
        for i in range(self.n_sources):
            dec = self.decoder(masked[:, i, :, :])          # (B, 2, T_decoded)
            if dec.shape[-1] > T:
                dec = dec[:, :, :T]
            elif dec.shape[-1] < T:
                dec = F.pad(dec, (0, T - dec.shape[-1]))
            outputs.append(dec)

        return torch.stack(outputs, dim=1)                  # (B, n_sources, 2, T)
