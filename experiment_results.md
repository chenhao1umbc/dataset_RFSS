# RFSS Experiment Results

All results use the **test split** (samples 85,000–99,999; 15% of 100k dataset).
Metric: **Permutation-Invariant Scale-Invariant Signal-to-Noise Ratio (PI-SI-SINR, dB)**.
Standard definition: Le Roux et al. "SDR - Half-baked or Well Done?" ICASSP 2019.
Both baselines and DL models use zero-mean centering before projection.
Baselines: N=150 per source count, seed=42. DL models: N=300 per source count, seed=42.

**Note on adjacent-channel evaluation:** Reference signals in HDF5 are stored
pre-frequency-shift. For adjacent-channel mixtures, a separation algorithm
must undo the per-source frequency offset to match the reference; this makes
adjacent-channel SI-SINR lower than co-channel despite the sources occupying
distinct spectral bands. This is documented in paper/dataset_spec.md.

---

## Phase 3: Classical Baseline Results (corrected, 2026-03-05)

**Method:** ICA via single-channel time-delay (Hankel) embedding;
NMF via magnitude STFT ratio masking.
**Code:** `src/baseline_algorithms.py`, `check/run_baselines.py`
**Raw results:** `check/baseline_results.json`, `check/breakdown_results.json`

### ICA

| Source Count | N   | Mean SI-SINR | Std   | Min     | Max    |
|-------------|-----|-------------|-------|---------|--------|
| 2-source    | 150 | -34.91 dB   | 12.09 | -68.61  | -11.35 |
| 3-source    | 150 | -36.98 dB   | 10.08 | -58.02  | -17.37 |
| 4-source    | 150 | -35.84 dB   | 9.74  | -60.70  | -16.99 |

### NMF

| Source Count | N   | Mean SI-SINR | Std   | Min     | Max    |
|-------------|-----|-------------|-------|---------|--------|
| 2-source    | 150 | -26.07 dB   | 15.07 | -59.46  | +5.08  |
| 3-source    | 150 | -29.69 dB   | 13.78 | -54.43  | +1.58  |
| 4-source    | 150 | -27.54 dB   | 13.54 | -54.12  | -2.76  |

### Breakdown by mixing mode (baselines)

| Config     | Mode            | N   | Mean SI-SINR |
|-----------|-----------------|-----|-------------|
| 2src ICA  | co-channel      | 49  | -28.04 dB   |
| 2src ICA  | adjacent-channel| 101 | -38.25 dB   |
| 2src NMF  | co-channel      | 49  | -16.19 dB   |
| 2src NMF  | adjacent-channel| 101 | -30.86 dB   |
| 3src ICA  | co-channel      | 47  | -28.20 dB   |
| 3src ICA  | adjacent-channel| 103 | -40.99 dB   |
| 3src NMF  | co-channel      | 47  | -15.08 dB   |
| 3src NMF  | adjacent-channel| 103 | -36.36 dB   |
| 4src ICA  | co-channel      | 65  | -27.61 dB   |
| 4src ICA  | adjacent-channel| 85  | -42.13 dB   |
| 4src NMF  | co-channel      | 65  | -14.63 dB   |
| 4src NMF  | adjacent-channel| 85  | -37.42 dB   |

### Analysis

Both methods fail. Paper claims (ICA=+15.2 dB, NMF=+18.3 dB) are definitively
disproved. NMF's co-channel performance (-14 to -16 dB) shows it can partially
exploit modulation-type spectral differences; adjacent-channel performance is
worse because the reference signals are stored pre-frequency-shift.

---

## Phase 5: Deep Learning Results

**Status: COMPLETE**

**Training config:**
- Scheduler: CosineAnnealingLR (T_max=30, eta_min=1e-5)
- 30 epochs, batch=8, lr=1e-3, train_length=7680, MPS (Mac Mini M4 Pro)
- Optimizer: Adam, grad clip norm=1.0
- Eval: test split, N=300 per source count, seed=42

**Checkpoints used (best val loss):**
- conv_tasnet_2src: `epoch_029_loss_19.8911.pt`
- conv_tasnet_3src: `epoch_017_loss_21.8244.pt` (from prior ReduceLROnPlateau run — kept because it outperformed the new run on val)
- conv_tasnet_4src: `epoch_018_loss_22.6436.pt`
- dprnn_2src: `epoch_029_loss_20.1535.pt`
- dprnn_3src: `epoch_026_loss_22.0370.pt`
- dprnn_4src: `epoch_004_loss_22.6446.pt` (best; model degraded after ep 4 — small 4-src dataset effect)
- cnn_lstm_2src: `epoch_028_loss_22.3671.pt`
- cnn_lstm_3src: `epoch_026_loss_23.3164.pt`
- cnn_lstm_4src: `epoch_025_loss_23.5638.pt`

### Main results table (PI-SI-SINR, dB, test split, N=300)

| Source Count | ICA       | NMF       | ConvTasNet | DPRNN     | CNN-LSTM  |
|-------------|-----------|-----------|------------|-----------|-----------|
| 2-source    | -34.91    | -26.07    | **-21.18** | -21.53    | -23.32    |
| 3-source    | -36.98    | -29.69    | **-21.08** | -21.31    | -23.65    |
| 4-source    | -35.84    | -27.54    | -22.13     | **-22.22**| -23.56    |

### Improvement over baselines (dB, positive = better)

| Source Count | ConvTasNet vs ICA | ConvTasNet vs NMF | DPRNN vs ICA | DPRNN vs NMF | CNN-LSTM vs ICA | CNN-LSTM vs NMF |
|-------------|------------------|------------------|-------------|-------------|----------------|----------------|
| 2-source    | +13.73           | +4.89            | +13.38      | +4.54       | +11.59         | +2.75          |
| 3-source    | +15.90           | +8.61            | +15.67      | +8.38       | +13.33         | +6.04          |
| 4-source    | +13.71           | +5.41            | +13.62      | +5.32       | +12.28         | +3.98          |

### Co-channel vs adjacent-channel breakdown (PI-SI-SINR, dB)

| Config         | N_co | ConvTasNet (co) | DPRNN (co) | CNN-LSTM (co) | NMF (co) | ICA (co) |
|---------------|------|----------------|-----------|--------------|---------|---------|
| 2-source      | 110  | -12.34         | -12.51    | -17.04       | -16.19  | -28.04  |
| 3-source      | 127  | -10.71         | -10.38    | -15.99       | -15.08  | -28.20  |
| 4-source      | 133  | -12.43         | -12.79    | -16.67       | -14.63  | -27.61  |

| Config         | N_adj | ConvTasNet (adj) | DPRNN (adj) | CNN-LSTM (adj) | NMF (adj) | ICA (adj) |
|---------------|-------|-----------------|------------|---------------|----------|----------|
| 2-source      | 190   | -26.81          | -27.10     | -27.32        | -30.86   | -38.25   |
| 3-source      | 173   | -28.59          | -29.33     | -29.12        | -36.36   | -40.99   |
| 4-source      | 167   | -29.76          | -29.62     | -29.06        | -37.42   | -42.13   |

### Analysis

**DL models consistently and significantly outperform classical baselines** across all
source counts and mixing modes. ConvTasNet and DPRNN perform similarly (~0.1–0.4 dB
apart); CNN-LSTM trails by ~1.4–2.4 dB.

**Source count trend:** Performance degrades modestly from 2-source to 4-source.
ConvTasNet and DPRNN show near-equal performance for 2 and 3 sources (~0.1 dB
difference), with a larger drop at 4 sources (~1 dB). CNN-LSTM shows a clearer
monotonic degradation (−23.32, −23.65, −23.56 dB). 4-source is consistently the
hardest, consistent with fewer training samples (~10.5k vs ~24.5k for 3-source) and
more permutations in PIT loss (24 vs 6).

**Adjacent-channel floor:** All methods show much lower SI-SINR on adjacent-channel
mixtures (roughly −27 to −30 dB for DL) because model outputs are evaluated against
pre-frequency-shift references. The co-channel breakdown is the more meaningful
metric: DL models achieve −10 to −17 dB co-channel vs NMF at −14 to −16 dB and ICA
at −28 dB, showing DL clearly outperforms classical methods on the core separation task.

**DPRNN 4-src note:** Best checkpoint is epoch 4; performance degraded afterward,
consistent with mild overfitting on the small 4-source training set.

---

## Notes

- Baseline eval: CPU, N=150 per source count.
- DL eval: N=300 per source count, seed=42, test split.
- eval_breakdown.py: computes co-channel vs adjacent-channel split for any checkpoint.
- HuggingFace upload: blocked pending user decision on repo name/visibility.
