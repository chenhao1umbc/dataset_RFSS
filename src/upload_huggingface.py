"""
Upload RFSS dataset files and dataset card to a HuggingFace Hub repository.

Usage:
    uv run python src/upload_huggingface.py \
        --repo username/rfss-dataset \
        --token hf_... \
        [--private] [--skip-multi] [--skip-single]
"""

import argparse
import os

from huggingface_hub import HfApi


DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
- audio-classification
- other
language:
- en
tags:
- RF
- signal-processing
- source-separation
- 5G
- LTE
- GSM
- UMTS
- wireless
- telecommunications
pretty_name: RFSS - RF Signal Source Separation Dataset
size_categories:
- 100K<n<1M
---

# RFSS: RF Signal Source Separation Dataset

## Dataset Description

RFSS is a synthetic RF signal source separation dataset covering 2G (GSM), 3G (UMTS), 4G (LTE), and 5G NR standards with realistic channel models and hardware impairments.

### Files

| File | Samples | Size | Description |
|------|---------|------|-------------|
| rfss_dataset.h5 | 100,000 | ~103 GB | Multi-source mixtures (2-4 sources) |
| rfss_single.h5 | 4,000 | ~1.3 GB | Single-standard samples (1,000/standard) |

## Dataset Structure

Each HDF5 file contains:
- `mixed_signals`: (N, 122880) complex64 -- received observation
- `source_signals`: (N, 4, 122880) complex64 -- ground-truth sources
- `signal_lengths`: (N,) int32 -- valid samples per signal
- `metadata`: (N,) UTF-8 JSON -- full configuration per sample

See [dataset_spec.md](paper/dataset_spec.md) for complete schema.

## Parameter Coverage (multi-source dataset)

| Parameter | Distribution |
|-----------|-------------|
| Source count | 2: 50%, 3: 35%, 4: 15% |
| Mixing mode | Co-channel: 40%, Adjacent-channel: 60% |
| MIMO config | 1x1: 50%, 2x2: 30%, 4x4: 20% |
| SNR | Uniform -10 to +40 dB |
| Channel models | TDL-A/B/C/D/E per 3GPP TR 38.901 |

## Usage

```python
from src.utils_dataset import RFSSDataset, create_dataloader

# Single item
ds = RFSSDataset('rfss_dataset.h5', split='train')  # 70k samples
item = ds[0]
# item['mixed_signal']:  torch.complex64, shape (L,)
# item['source_signals']: list of torch.complex64, each (L,)
# item['metadata']:       dict

# DataLoader
loader = create_dataloader('rfss_dataset.h5', split='train', batch_size=32)
batch = next(iter(loader))
# batch['mixed_signals']:  (B, max_L) complex64
# batch['source_signals']: (B, max_sources, max_L) complex64
# batch['signal_lengths']: (B,) int32
```

## Train/Val/Test Split

70/15/15 applied at load time by `RFSSDataset` (sequential index split).

## Citation

If you use this dataset, please cite:
```
@dataset{rfss2026,
  title  = {RFSS: RF Signal Source Separation Dataset},
  year   = {2026},
}
```

## License

CC-BY-4.0
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload RFSS dataset files to a HuggingFace Hub repository."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID, e.g. username/rfss-dataset",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create a private repository (default: public)",
    )
    parser.add_argument(
        "--skip-multi",
        action="store_true",
        default=False,
        help="Skip uploading rfss_dataset.h5 (the ~103 GB multi-source file)",
    )
    parser.add_argument(
        "--skip-single",
        action="store_true",
        default=False,
        help="Skip uploading rfss_single.h5 (the ~1.3 GB single-standard file)",
    )
    return parser.parse_args()


def resolve_token(token_arg: str | None) -> str | None:
    """Return the HF token from the CLI argument or the HF_TOKEN environment variable."""
    if token_arg:
        return token_arg
    return os.environ.get("HF_TOKEN", None)


def upload_dataset_card(api: HfApi, repo_id: str, token: str | None) -> None:
    """Upload the dataset card as README.md to the repository root."""
    print("Uploading dataset card (README.md) ...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card",
    )
    print("Dataset card uploaded.")


def upload_hdf5(
    api: HfApi,
    repo_id: str,
    token: str | None,
    local_path: str,
    repo_filename: str,
) -> None:
    """Upload a single HDF5 file to the repository."""
    print(f"Uploading {local_path} -> {repo_filename} ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"Add {repo_filename}",
    )
    print(f"Uploaded {repo_filename}.")


def main() -> None:
    """Entry point: create the HF repo (if needed) and upload all selected files."""
    args = parse_args()
    token = resolve_token(args.token)

    api = HfApi()

    print(f"Creating or verifying repository: {args.repo}")
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        token=token,
        exist_ok=True,
    )

    upload_dataset_card(api, args.repo, token)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not args.skip_multi:
        multi_path = os.path.join(project_root, "data", "rfss_dataset.h5")
        upload_hdf5(api, args.repo, token, multi_path, "rfss_dataset.h5")
    else:
        print("Skipping rfss_dataset.h5 (--skip-multi specified).")

    if not args.skip_single:
        single_path = os.path.join(project_root, "data", "rfss_single.h5")
        upload_hdf5(api, args.repo, token, single_path, "rfss_single.h5")
    else:
        print("Skipping rfss_single.h5 (--skip-single specified).")

    print("All uploads complete.")


if __name__ == "__main__":
    main()
