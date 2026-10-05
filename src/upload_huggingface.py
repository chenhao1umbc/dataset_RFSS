"""
Release RFSS v1.1 on the HuggingFace Hub: tag the existing state as v1.0, then add the dataset card, the data licence and
the trained checkpoints in one commit. The HDF5 files are uploaded only when --upload-hdf5 is given (v1.1 keeps the v1.0 bytes).

Usage:
    HF_TOKEN=hf_... uv run python src/upload_huggingface.py --repo Chrishao/rfss [--upload-hdf5] [--dry-run]
"""

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent
CARD = ROOT / "docs" / "hf_dataset_card.md"
LICENSE_DATA = ROOT / "LICENSE-DATA"
TEST_PASSES = ROOT / "check" / "run_test_passes.sh"
HDF5_FILES = ("rfss_dataset.h5", "rfss_single.h5")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Release RFSS v1.1 on a HuggingFace dataset repository."
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo ID, e.g. Chrishao/rfss",
    )
    parser.add_argument(
        "--upload-hdf5",
        action="store_true",
        help="also upload the two HDF5 files to data/",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list the files of the commit and stop"
    )
    return parser.parse_args()


def scored_checkpoints() -> list[Path]:
    """The checkpoints scored in the paper: every final/<run>/ckpt/<file>.pt named in check/run_test_passes.sh."""
    paths = sorted(
        set(re.findall(r"final/[\w.-]+/ckpt/[\w.-]+\.pt", TEST_PASSES.read_text()))
    )
    return [ROOT / p for p in paths]


def build_operations(upload_hdf5: bool) -> list[CommitOperationAdd]:
    """Files of the v1.1 commit: card as README.md, data licence, checkpoints as checkpoints/<run>/ckpt/<file>.pt."""
    operations = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(CARD)),
        CommitOperationAdd(path_in_repo="LICENSE", path_or_fileobj=str(LICENSE_DATA)),
    ]
    for path in scored_checkpoints():
        operations.append(
            CommitOperationAdd(
                path_in_repo="checkpoints/"
                + path.relative_to(ROOT / "final").as_posix(),
                path_or_fileobj=str(path),
            )
        )
    if upload_hdf5:
        operations += [
            CommitOperationAdd(
                path_in_repo=f"data/{name}", path_or_fileobj=str(ROOT / "data" / name)
            )
            for name in HDF5_FILES
        ]
    return operations


def main() -> None:
    """Tag the current state as v1.0 (if not tagged yet), then commit the v1.1 files."""
    args = parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN") or os.environ.get("HF_Token"))
    operations = build_operations(args.upload_hdf5)
    for op in operations:
        print(
            f"{op.path_in_repo}  {Path(op.path_or_fileobj).stat().st_size / 1e6:.1f} MB"
        )
    if args.dry_run:
        return

    tags = {t.name for t in api.list_repo_refs(args.repo, repo_type="dataset").tags}
    if "v1.0" not in tags:
        head = api.dataset_info(args.repo).sha
        api.create_tag(
            args.repo,
            tag="v1.0",
            revision=head,
            repo_type="dataset",
            tag_message="Files of February 2026, no dataset card",
        )
        print(f"Tagged {head} as v1.0")
    commit = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        operations=operations,
        commit_message="v1.1: dataset card, data licence (CC BY-NC 4.0) and trained checkpoints; HDF5 files unchanged",
    )
    api.create_tag(
        args.repo,
        tag="v1.1",
        revision=commit.oid,
        repo_type="dataset",
        tag_message="Dataset card, licence, checkpoints",
    )
    print(f"Committed {commit.oid} and tagged v1.1")


if __name__ == "__main__":
    main()
