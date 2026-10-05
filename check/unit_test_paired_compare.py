"""
Bin sizes of check/paired_compare.py on the fixed validation crops: they depend on the source count, so the script must
read it from the stored run records (a wrong default once gave the 2-source bins for a 3-source comparison).
Skipped automatically if the HDF5 file is absent.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from encoder_sweep import DATASET_PATH, load_items  # noqa: E402
from paired_compare import bin_indices, source_count  # noqa: E402

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason=f"HDF5 not found: {DATASET_PATH}")


@pytest.mark.parametrize("n_sources, adjacent, co_channel", [(2, 106, 83), (3, 129, 83), (4, 113, 68)])
def test_bin_sizes_per_source_count(n_sources, adjacent, co_channel):
    _, _, info = load_items("val", 800, n_sources)
    sizes = {name: len(idx) for name, idx in bin_indices(info).items()}
    assert sizes["all"] == 800
    assert sizes["adjacent_snr_gt_20"] == adjacent
    assert sizes["co_snr_gt_20"] == co_channel


def test_source_count_comes_from_the_records():
    results = {"a": {"n_sources": 3}, "b": {"n_sources": 3}, "c": {"n_sources": 2}, "d": {}}
    assert source_count(results, "a", "b:2") == 3
    with pytest.raises(ValueError):
        source_count(results, "a", "c")
    with pytest.raises(ValueError):
        source_count(results, "a", "d")
