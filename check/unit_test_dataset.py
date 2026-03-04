"""
End-to-end DataLoader test for RFSS dataset.

Instantiates RFSSDataset on rfss_dataset.h5, creates a DataLoader
(num_workers=0), iterates one batch, and verifies tensor shapes and dtypes.
Skipped automatically if the HDF5 file is absent.
"""

import json
from pathlib import Path

import pytest
import torch

MULTI_H5 = Path('data/rfss_dataset.h5')
SINGLE_H5 = Path('data/rfss_single.h5')

pytestmark = pytest.mark.skipif(
    not MULTI_H5.exists(),
    reason=f"Multi-source HDF5 not found: {MULTI_H5}"
)


# ---------------------------------------------------------------------------
# RFSSDataset
# ---------------------------------------------------------------------------

class TestRFSSDataset:
    def test_split_sizes(self):
        from src.utils_dataset import RFSSDataset
        train_ds = RFSSDataset(MULTI_H5, split='train')
        val_ds = RFSSDataset(MULTI_H5, split='val')
        test_ds = RFSSDataset(MULTI_H5, split='test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100000, f"Expected 100000, got {total}"
        assert len(train_ds) == 70000
        assert len(val_ds) == 15000
        assert len(test_ds) == 15000

    def test_getitem_types(self):
        from src.utils_dataset import RFSSDataset
        ds = RFSSDataset(MULTI_H5, split='test')
        item = ds[0]
        assert isinstance(item['mixed_signal'], torch.Tensor)
        assert item['mixed_signal'].dtype == torch.complex64
        assert item['mixed_signal'].ndim == 1
        assert item['mixed_signal'].shape[0] > 0
        assert isinstance(item['source_signals'], list)
        assert len(item['source_signals']) >= 2
        for src in item['source_signals']:
            assert isinstance(src, torch.Tensor)
            assert src.dtype == torch.complex64
        assert isinstance(item['metadata'], dict)
        assert 'num_sources' in item['metadata']
        assert 'sources' in item['metadata']

    def test_metadata_fields(self):
        from src.utils_dataset import RFSSDataset
        ds = RFSSDataset(MULTI_H5, split='test')
        item = ds[0]
        meta = item['metadata']
        for field in ('sample_id', 'num_sources', 'sources', 'snr_db',
                      'mixing_params', 'mimo_config'):
            assert field in meta, f"Missing metadata field: {field}"

    def test_source_count_matches_metadata(self):
        from src.utils_dataset import RFSSDataset
        ds = RFSSDataset(MULTI_H5, split='test')
        for idx in range(min(10, len(ds))):
            item = ds[idx]
            ns_meta = item['metadata']['num_sources']
            ns_actual = len(item['source_signals'])
            assert ns_actual == ns_meta, (
                f"Sample {idx}: metadata says {ns_meta} sources, "
                f"got {ns_actual} non-zero sources"
            )

    def test_no_nan_inf(self):
        from src.utils_dataset import RFSSDataset
        ds = RFSSDataset(MULTI_H5, split='test')
        for idx in range(min(5, len(ds))):
            item = ds[idx]
            assert item['mixed_signal'].isfinite().all(), \
                f"NaN/Inf in mixed_signal at index {idx}"
            for src in item['source_signals']:
                assert src.isfinite().all(), \
                    f"NaN/Inf in source_signal at index {idx}"


# ---------------------------------------------------------------------------
# DataLoader / batch collation
# ---------------------------------------------------------------------------

class TestDataLoader:
    @pytest.fixture(scope='class')
    def one_batch(self):
        from src.utils_dataset import create_dataloader
        loader = create_dataloader(
            MULTI_H5, split='test', batch_size=4,
            shuffle=False, num_workers=0
        )
        return next(iter(loader))

    def test_batch_keys(self, one_batch):
        for key in ('mixed_signals', 'source_signals',
                    'signal_lengths', 'metadata', 'sample_ids'):
            assert key in one_batch, f"Missing batch key: {key}"

    def test_mixed_signals_shape(self, one_batch):
        ms = one_batch['mixed_signals']
        assert isinstance(ms, torch.Tensor)
        assert ms.dtype == torch.complex64
        assert ms.ndim == 2               # (batch, max_len)
        assert ms.shape[0] == 4

    def test_source_signals_shape(self, one_batch):
        ss = one_batch['source_signals']
        assert isinstance(ss, torch.Tensor)
        assert ss.dtype == torch.complex64
        assert ss.ndim == 3               # (batch, max_sources, max_len)
        assert ss.shape[0] == 4

    def test_signal_lengths(self, one_batch):
        sl = one_batch['signal_lengths']
        assert isinstance(sl, torch.Tensor)
        assert sl.dtype == torch.int32
        assert sl.shape == (4,)
        assert (sl > 0).all()
        # lengths must not exceed the padded dimension
        assert (sl <= one_batch['mixed_signals'].shape[1]).all()

    def test_padding_consistency(self, one_batch):
        """Samples past their declared length should be zero-padded."""
        ms = one_batch['mixed_signals']
        sl = one_batch['signal_lengths']
        for i in range(ms.shape[0]):
            tail = ms[i, int(sl[i]):]
            assert (tail == 0).all(), \
                f"Non-zero padding in sample {i} after length {sl[i]}"

    def test_metadata_list(self, one_batch):
        meta_list = one_batch['metadata']
        assert isinstance(meta_list, list)
        assert len(meta_list) == 4
        for m in meta_list:
            assert isinstance(m, dict)

    def test_batch_size_multiple_batches(self):
        from src.utils_dataset import create_dataloader
        loader = create_dataloader(
            MULTI_H5, split='test', batch_size=8,
            shuffle=False, num_workers=0
        )
        seen = 0
        for batch in loader:
            bs = batch['mixed_signals'].shape[0]
            assert bs <= 8
            seen += bs
            if seen >= 32:
                break
        assert seen >= 32


# ---------------------------------------------------------------------------
# Single-source DataLoader
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not SINGLE_H5.exists(),
    reason=f"Single-source HDF5 not found: {SINGLE_H5}"
)
class TestSingleSourceDataLoader:
    def test_single_source_getitem(self):
        from src.utils_dataset import RFSSDataset
        ds = RFSSDataset(SINGLE_H5, split='train')
        item = ds[0]
        assert item['metadata']['num_sources'] == 1
        assert len(item['source_signals']) == 1

    def test_single_source_split_sizes(self):
        from src.utils_dataset import RFSSDataset
        train_ds = RFSSDataset(SINGLE_H5, split='train')
        val_ds = RFSSDataset(SINGLE_H5, split='val')
        test_ds = RFSSDataset(SINGLE_H5, split='test')
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 4000

    def test_single_source_batch(self):
        from src.utils_dataset import create_dataloader
        loader = create_dataloader(
            SINGLE_H5, split='train', batch_size=4,
            shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        assert batch['mixed_signals'].dtype == torch.complex64
        assert batch['mixed_signals'].shape[0] == 4
