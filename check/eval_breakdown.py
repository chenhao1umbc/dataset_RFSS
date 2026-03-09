"""Co-channel vs adjacent-channel breakdown evaluation.

Loads the best checkpoint for a given model/n_sources combination and
evaluates PI-SI-SINR on the test split, stratified by mixing_mode.
Also post-processes baseline_results.json for the same breakdown.

Usage:
    uv run python check/eval_breakdown.py --model conv_tasnet --n-sources 2 \\
        --checkpoint checkpoints/conv_tasnet_2src/<best>.pt
    uv run python check/eval_breakdown.py --baselines-only
"""

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import ConvTasNet, CNNLSTMSeparator, DualPathRNN, si_sinr
from src.train import SeparationDataset

DATASET_PATH = Path("data/rfss_dataset.h5")
BASELINE_RESULTS = Path("check/baseline_results.json")
BREAKDOWN_OUTPUT = Path("check/breakdown_results.json")
TRAIN_LENGTH = 7680
N_EVAL = 300  # per source count; stratified post-hoc by mixing_mode
EVAL_SEED = 42


def best_permutation_si_sinr(estimates: torch.Tensor, sources: torch.Tensor) -> float:
    """PI-SI-SINR for a single sample. estimates/sources: (C, 2, T)."""
    C = estimates.shape[0]
    est_flat = estimates.reshape(C, -1)
    tgt_flat = sources.reshape(C, -1)

    sinr_matrix = torch.zeros(C, C, device=estimates.device)
    for i in range(C):
        for j in range(C):
            sinr_matrix[i, j] = si_sinr(est_flat[i:i+1], tgt_flat[j:j+1]).squeeze()

    best = max(
        sum(sinr_matrix[i, perm[i]].item() for i in range(C)) / C
        for perm in permutations(range(C))
    )
    return best


def eval_model_breakdown(model, device: str, n_sources: int) -> dict:
    """Evaluate model on test set, return per-sample results with mixing_mode."""
    import h5py, json as _json

    ds = SeparationDataset(DATASET_PATH, split='test', n_sources=n_sources,
                           train_length=TRAIN_LENGTH)
    rng = np.random.RandomState(EVAL_SEED)
    chosen = rng.choice(len(ds), size=min(N_EVAL, len(ds)), replace=False)
    ds.indices = [ds.indices[i] for i in sorted(chosen)]

    # Need mixing_mode per sample from metadata
    h5file = h5py.File(str(DATASET_PATH), 'r')
    mixing_modes = {}
    for global_idx in ds.indices:
        meta = _json.loads(h5file['metadata'][global_idx])
        mixing_modes[global_idx] = meta['mixing_params']['mixing_mode']
    h5file.close()

    np.random.seed(EVAL_SEED)  # make random crops deterministic
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    model.eval()

    all_results = []
    sample_ptr = 0

    with torch.no_grad():
        for batch in loader:
            mixed = batch['mixed'].to(device)
            sources_t = batch['sources'].to(device)
            estimates = model(mixed)
            B = mixed.shape[0]

            for b in range(B):
                global_idx = ds.indices[sample_ptr]
                sinr_val = best_permutation_si_sinr(estimates[b], sources_t[b])
                all_results.append({
                    'global_idx': global_idx,
                    'mixing_mode': mixing_modes[global_idx],
                    'si_sinr_db': sinr_val,
                })
                sample_ptr += 1

    return all_results


def breakdown_stats(results: list, mixing_mode: str) -> dict:
    """Compute stats for a subset filtered by mixing_mode."""
    vals = [r['si_sinr_db'] for r in results if r['mixing_mode'] == mixing_mode]
    if not vals:
        return {'n': 0, 'mean': None, 'std': None}
    return {
        'n': len(vals),
        'mean_si_sinr_db': round(float(np.mean(vals)), 4),
        'std_si_sinr_db': round(float(np.std(vals)), 4),
        'min_si_sinr_db': round(float(np.min(vals)), 4),
        'max_si_sinr_db': round(float(np.max(vals)), 4),
    }


def baseline_breakdown() -> dict:
    """Post-process baseline_results.json to produce mixing_mode breakdown."""
    if not BASELINE_RESULTS.exists():
        print(f"  {BASELINE_RESULTS} not found — run check/run_baselines.py first")
        return {}

    raw = json.loads(BASELINE_RESULTS.read_text())
    out = {}
    for key, data in raw.items():
        co = breakdown_stats(data['samples'], 'co-channel')
        adj = breakdown_stats(data['samples'], 'adjacent-channel')
        out[key] = {
            'overall': {k: v for k, v in data.items() if k != 'samples'},
            'co_channel': co,
            'adjacent_channel': adj,
        }
    return out


def print_breakdown_table(results: dict):
    """Print a readable breakdown table."""
    print(f"\n{'Config':<25} {'Mode':<20} {'N':>5} {'Mean SI-SINR':>14} {'Std':>8}")
    print("-" * 76)
    for key in sorted(results.keys()):
        r = results[key]
        overall = r.get('overall', r)
        n_total = overall.get('n_samples', overall.get('n', '?'))
        mean_all = overall.get('mean_si_sinr_db', '?')
        print(f"{key:<25} {'overall':<20} {n_total:>5} {mean_all:>13.2f} dB")
        for mode_key, label in [('co_channel', 'co-channel'), ('adjacent_channel', 'adjacent-channel')]:
            if mode_key in r and r[mode_key]['n'] > 0:
                s = r[mode_key]
                print(f"{'':25} {label:<20} {s['n']:>5} {s['mean_si_sinr_db']:>13.2f} dB  {s['std_si_sinr_db']:>6.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['conv_tasnet', 'cnn_lstm', 'dprnn'],
                        default='conv_tasnet')
    parser.add_argument('--n-sources', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--baselines-only', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    all_breakdown = {}

    # Always include baseline breakdown
    print("=== Baseline breakdown (co-channel vs adjacent-channel) ===")
    baseline_bd = baseline_breakdown()
    all_breakdown.update(baseline_bd)
    print_breakdown_table(baseline_bd)

    if not args.baselines_only and args.checkpoint:
        if args.device == 'auto':
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        else:
            device = args.device

        if args.model == 'conv_tasnet':
            model = ConvTasNet(N=256, L=16, B=128, H=256, P=3, X=8, R=3,
                               n_sources=args.n_sources)
        elif args.model == 'cnn_lstm':
            model = CNNLSTMSeparator(n_sources=args.n_sources)
        else:
            model = DualPathRNN(N=64, L=16, B=64, H=64, P=50, num_layers=6,
                                n_sources=args.n_sources)

        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)

        key = f"{args.n_sources}src_{args.model}"
        print(f"\n=== {key} breakdown ({args.checkpoint}) ===")
        results = eval_model_breakdown(model, device, args.n_sources)

        co = breakdown_stats(results, 'co-channel')
        adj = breakdown_stats(results, 'adjacent-channel')
        all_breakdown[key] = {
            'overall': {
                'n_samples': len(results),
                'mean_si_sinr_db': round(float(np.mean([r['si_sinr_db'] for r in results])), 4),
                'std_si_sinr_db': round(float(np.std([r['si_sinr_db'] for r in results])), 4),
            },
            'co_channel': co,
            'adjacent_channel': adj,
            'samples': results,
        }
        print_breakdown_table({key: all_breakdown[key]})

    # Merge with existing results so sequential runs accumulate
    if BREAKDOWN_OUTPUT.exists():
        existing = json.loads(BREAKDOWN_OUTPUT.read_text())
        for k, v in existing.items():
            if k not in all_breakdown:
                all_breakdown[k] = v

    BREAKDOWN_OUTPUT.write_text(json.dumps(all_breakdown, indent=2, default=float))
    print(f"\nResults saved to {BREAKDOWN_OUTPUT}")


if __name__ == '__main__':
    main()
