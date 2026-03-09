"""Training infrastructure for RF source separation models (Phase 4/5).

Provides SeparationDataset for HDF5-backed data loading, Trainer for
training/evaluation loops with checkpoint management, and a main() entry
point with argparse for running experiments.
"""

import argparse
import json
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import scipy.signal as sp_signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import ConvTasNet, CNNLSTMSeparator, DualPathRNN, pit_si_sinr_loss, si_sinr


def _checkpoint_loss(p: Path) -> float:
    """Parse val_loss from checkpoint filename for sorting."""
    try:
        return float(p.stem.split('_loss_')[-1])
    except ValueError:
        return float('inf')


def _resample_complex(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a complex signal to target_len via Fourier resampling."""
    if np.iscomplexobj(signal):
        r = sp_signal.resample(signal.real, target_len)
        i = sp_signal.resample(signal.imag, target_len)
        return r + 1j * i
    return sp_signal.resample(signal.real, target_len).astype(np.complex128)


class SeparationDataset(Dataset):
    """Dataset for source separation training/evaluation.

    Filters samples by num_sources, upsamples reference sources to the
    mixed signal rate, and random-crops to train_length for training.

    Args:
        h5_path: Path to rfss_dataset.h5.
        split: 'train', 'val', or 'test'.
        n_sources: Filter to samples with exactly this many sources.
        train_length: If set, random-crop signals to this length; None=full length.
    """

    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        n_sources: int = 2,
        train_length: Optional[int] = None,
    ):
        self.h5_path = str(h5_path)
        self.split = split
        self.n_sources = n_sources
        self.train_length = train_length
        self._h5 = None

        # Scan for valid indices
        h5file = h5py.File(self.h5_path, 'r')
        total_samples = int(h5file.attrs.get('actual_samples', h5file.attrs['max_samples']))

        train_end = int(0.70 * total_samples)
        val_end = int(0.85 * total_samples)

        if split == 'train':
            start_idx, end_idx = 0, train_end
        elif split == 'val':
            start_idx, end_idx = train_end, val_end
        else:
            start_idx, end_idx = val_end, total_samples

        self.indices = []
        for global_idx in range(start_idx, end_idx):
            metadata_str = h5file['metadata'][global_idx]
            meta = json.loads(metadata_str)
            if meta['num_sources'] == n_sources:
                self.indices.append(global_idx)

        h5file.close()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        if not hasattr(self, '_h5') or self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')

        global_idx = self.indices[idx]
        signal_len = int(self._h5['signal_lengths'][global_idx])
        metadata_str = self._h5['metadata'][global_idx]
        meta = json.loads(metadata_str)

        # Determine output length and crop window for the mixed signal
        if self.train_length is not None:
            out_len = self.train_length
            if signal_len >= out_len:
                crop_start = np.random.randint(0, signal_len - out_len + 1)
            else:
                crop_start = 0
        else:
            out_len = signal_len
            crop_start = 0

        # Load only the needed mixed signal slice
        crop_end = min(crop_start + out_len, signal_len)
        mixed_raw = self._h5['mixed_signals'][global_idx, crop_start:crop_end].astype(np.complex128)
        if len(mixed_raw) < out_len:
            mixed_raw = np.pad(mixed_raw, (0, out_len - len(mixed_raw)))

        # Load each source efficiently based on whether rate matches mixed signal rate
        sources = []
        for i in range(self.n_sources):
            src_meta = meta['sources'][i]
            sample_rate = src_meta['signal_params']['sample_rate']
            native_len = int(round(sample_rate * 0.001))

            if native_len == signal_len:
                # Same rate as mixed signal: load the exact crop slice (no resampling)
                src_end = min(crop_start + out_len, native_len)
                src_slice = self._h5['source_signals'][global_idx, i, crop_start:src_end].astype(np.complex128)
                if len(src_slice) < out_len:
                    src_slice = np.pad(src_slice, (0, out_len - len(src_slice)))
                sources.append(src_slice)
            else:
                # Different rate: compute corresponding native-rate slice, resample to out_len
                ratio = native_len / signal_len
                native_start = int(crop_start * ratio)
                native_needed = int(out_len * ratio) + 2
                native_end = min(native_start + native_needed, native_len)
                src_slice = self._h5['source_signals'][global_idx, i, native_start:native_end].astype(np.complex128)
                sources.append(_resample_complex(src_slice, out_len))

        sources_arr = np.stack(sources, axis=0)  # (n_sources, out_len)

        # Normalize by mixed RMS
        rms = float(np.sqrt(np.mean(np.abs(mixed_raw) ** 2)))
        denom = rms + 1e-8
        mixed_norm = mixed_raw / denom
        sources_norm = sources_arr / denom

        # Convert to float32 tensors
        mixed_2ch = torch.from_numpy(
            np.stack([mixed_norm.real, mixed_norm.imag], axis=0).astype(np.float32)
        )  # (2, out_len)

        sources_list = []
        for i in range(self.n_sources):
            s = sources_norm[i]
            s_2ch = torch.from_numpy(
                np.stack([s.real, s.imag], axis=0).astype(np.float32)
            )  # (2, out_len)
            sources_list.append(s_2ch)
        sources_2ch = torch.stack(sources_list, dim=0)  # (n_sources, 2, out_len)

        return {'mixed': mixed_2ch, 'sources': sources_2ch, 'signal_len': out_len}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __setstate__(self, state: dict):
        self.__dict__.update(state)


class Trainer:
    """Training and evaluation loop for source separation models.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        scheduler: LR scheduler (.step() called each epoch, no arguments).
        device: Torch device string.
        checkpoint_dir: Directory for checkpoint files.
        log_dir: Directory for tensorboard logs (None to disable).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str,
        checkpoint_dir: str,
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = log_dir
        self.writer = None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if log_dir is not None and _TENSORBOARD_AVAILABLE:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir)

        self._global_step = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            mixed = batch['mixed'].to(self.device)       # (B, 2, T)
            sources = batch['sources'].to(self.device)   # (B, n_sources, 2, T)

            self.optimizer.zero_grad()
            estimates = self.model(mixed)                # (B, n_sources, 2, T)
            loss = pit_si_sinr_loss(estimates, sources)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            if self.writer is not None and self._global_step % 50 == 0:
                self.writer.add_scalar('train/batch_loss', loss.item(), self._global_step)

        return total_loss / max(n_batches, 1)

    def evaluate(self, dataloader: DataLoader, epoch: Optional[int] = None, split: str = 'val') -> dict:
        self.model.eval()
        total_loss = 0.0
        total_sisnr = 0.0
        n_batches = 0
        n_samples = 0  # track sample count for unbiased mean

        with torch.no_grad():
            for batch in dataloader:
                mixed = batch['mixed'].to(self.device)
                sources = batch['sources'].to(self.device)

                estimates = self.model(mixed)
                loss = pit_si_sinr_loss(estimates, sources)

                # Best-permutation mean SI-SINR per sample, then sum over batch
                B, C, _, T = estimates.shape
                est_flat = estimates.reshape(B, C, -1)
                tgt_flat = sources.reshape(B, C, -1)

                sisnr_matrix = torch.zeros(B, C, C, device=self.device)
                for i in range(C):
                    for j in range(C):
                        sisnr_matrix[:, i, j] = si_sinr(est_flat[:, i, :], tgt_flat[:, j, :])

                perms = list(permutations(range(C)))
                perm_scores = torch.stack(
                    [sisnr_matrix[:, range(C), list(p)].mean(dim=-1) for p in perms],
                    dim=-1
                )
                # Sum over samples in batch (divide by n_samples at the end)
                total_sisnr += perm_scores.max(dim=-1).values.sum().item()

                total_loss += loss.item() * B
                n_batches += 1
                n_samples += B

        mean_loss = total_loss / max(n_samples, 1)
        mean_sisnr = total_sisnr / max(n_samples, 1)

        if self.writer is not None and epoch is not None:
            self.writer.add_scalar(f'{split}/loss', mean_loss, epoch)
            self.writer.add_scalar(f'{split}/mean_si_sinr_db', mean_sisnr, epoch)

        return {'loss': mean_loss, 'mean_si_sinr_db': mean_sisnr}

    def save_checkpoint(self, epoch: int, val_loss: float):
        filename = self.checkpoint_dir / f'epoch_{epoch:03d}_loss_{val_loss:.4f}.pt'
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }, filename)

        # Keep only 3 best checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob('epoch_*.pt'))
        if len(checkpoints) > 3:
            checkpoints_by_loss = sorted(checkpoints, key=_checkpoint_loss)
            for ckpt in checkpoints_by_loss[3:]:
                ckpt.unlink()

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        return int(ckpt['epoch'])


def main():
    parser = argparse.ArgumentParser(description='Train RF source separation model')
    parser.add_argument('--model', choices=['conv_tasnet', 'cnn_lstm', 'dprnn'],
                        default='conv_tasnet')
    parser.add_argument('--n-sources', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-length', type=int, default=7680)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='runs')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--data', type=str, default='data/rfss_dataset.h5')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    if args.smoke_test:
        args.epochs = 2

    # Build model
    if args.model == 'conv_tasnet':
        model = ConvTasNet(N=256, L=16, B=128, H=256, P=3, X=8, R=3, n_sources=args.n_sources)
    elif args.model == 'cnn_lstm':
        model = CNNLSTMSeparator(n_sources=args.n_sources)
    else:
        model = DualPathRNN(N=64, L=16, B=64, H=64, P=50, num_layers=6, n_sources=args.n_sources)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Build datasets
    train_ds = SeparationDataset(
        args.data, split='train', n_sources=args.n_sources, train_length=args.train_length
    )
    val_ds = SeparationDataset(
        args.data, split='val', n_sources=args.n_sources, train_length=args.train_length
    )

    if args.smoke_test:
        train_ds.indices = train_ds.indices[:50]
        val_ds.indices = val_ds.indices[:10]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # CosineAnnealingLR: smooth LR decay from lr to eta_min over args.epochs steps.
    # Avoids premature LR reduction from noisy validation loss (ReduceLROnPlateau issue).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    log_dir = args.log_dir if _TENSORBOARD_AVAILABLE else None
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=log_dir,
    )

    start_epoch = 0
    if args.resume is not None:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f'Resumed from {args.resume}, epoch {start_epoch}', flush=True)

    print(f'Config: model={args.model}, n_sources={args.n_sources}, device={device}', flush=True)
    print(f'Model parameters: {n_params:,}', flush=True)
    print(f'Train samples: {len(train_ds)}, Val samples: {len(val_ds)}', flush=True)
    print(f'Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.evaluate(val_loader, epoch=epoch, split='val')
        elapsed = time.time() - t0

        scheduler.step()  # CosineAnnealingLR takes no arguments
        trainer.save_checkpoint(epoch, val_metrics['loss'])

        print(
            f'Epoch {epoch + 1}/{args.epochs} | '
            f'train_loss={train_loss:.4f} | '
            f'val_loss={val_metrics["loss"]:.4f} | '
            f'val_SI-SINR={val_metrics["mean_si_sinr_db"]:.2f} dB | '
            f't={elapsed:.1f}s',
            flush=True,
        )

    if trainer.writer is not None:
        trainer.writer.close()


if __name__ == '__main__':
    main()
