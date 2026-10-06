# Reviewer Notes — Phase 5 Re-run Preparation

Code review of `src/train.py`, `src/models.py`, `src/baseline_algorithms.py`,
`check/run_baselines.py`, and `train_all.sh`. Updated after the other agent
applied fixes. Status column reflects current state of the code.

---

## Part 1: Bugs — Status After Agent Fixes

### B1. `scheduler.step()` argument incompatibility — RESOLVED

`train.py` line 413 now correctly calls `scheduler.step()` with no arguments,
and the Trainer docstring was updated accordingly. The scheduler is
`CosineAnnealingLR(T_max=args.epochs, eta_min=1e-5)`. Fix is correct.

### B2. SI-SINR zero-mean inconsistency between DL and baselines — RESOLVED

`baseline_algorithms.py` `compute_si_sinr()` now subtracts the mean (lines
43-44), matching the Le Roux et al. 2019 standard definition used by
`models.py`. Both sides now apply zero-mean centering before the projection.

**Consequence:** The previously reported baseline numbers (ICA -31.95 to
-36.82 dB, NMF -23.04 to -30.16 dB) were computed without zero-mean. They
must be regenerated. For near-zero-mean RF signals the difference will be
small, but the numbers in `baseline_results.json` and `experiment_results.md`
are now stale and cannot be cited in the paper.

### B3. SI-SINR computed on flattened real+imag — OPEN (design note)

`pit_si_sinr_loss` and `evaluate()` still flatten `(B, C, 2, T)` to `(B, C, 2T)`
and call `si_sinr()` on the concatenated real+imag vector. After B2 is fixed,
both DL and baselines now apply zero-mean — but zero-mean is applied to the 2T
flattened vector in DL (mean over real+imag jointly) and to the complex signal
in baselines (mean of complex samples, which is `mean_real + j*mean_imag`).
Numerically these are equivalent only if `mean_real == mean_imag`, which is
not guaranteed.

This is unlikely to cause a large numerical difference for RF baseband signals,
but it should be noted clearly in the paper's metric definition section to avoid
reviewer questions.

### B4. Import inside evaluation loop — RESOLVED

`si_sinr` is now imported at module level (line 30):
```python
from src.models import ConvTasNet, CNNLSTMSeparator, DualPathRNN, pit_si_sinr_loss, si_sinr
```
The local import inside the evaluation loop is gone.

---

## Part 2: Configuration Errors — Status After Agent Fixes

### C1. `train_all.sh` uses `--num-workers 4` — RESOLVED

All invocations now use `--num-workers 0`. Correct for macOS.

### C2. `train_all.sh` batch-size mismatch — RESOLVED

All invocations now use `--batch-size 8`, matching the original experiment
configuration.

### C3. Checkpoint does not save scheduler state — RESOLVED

`save_checkpoint()` now includes `'scheduler': self.scheduler.state_dict()`.
`load_checkpoint()` conditionally restores it. Correct.

### C4. Baseline uses first-N sequential samples — RESOLVED

`collect_test_indices()` now scans all 15k test samples, then draws 150 random
indices per group using `np.random.RandomState(SAMPLE_SEED=42)`.
`N_PER_GROUP` raised from 30 to 150 to match DL evaluation.

**Note:** The full scan (15k HDF5 metadata reads) adds ~1-2 minutes to baseline
startup, which is acceptable.

### C5. Unused `rng` dead code — RESOLVED

Removed from `evaluate_algorithm()` signature and from `main()`.

### C6. Mean-of-batch-means bias — RESOLVED

`evaluate()` now tracks `n_samples` and accumulates
`perm_scores.max(dim=-1).values.sum()` (sum over batch items), then divides by
`n_samples` at the end. Per-sample mean is now unbiased.

---

## Part 3: Design Observations (unchanged, still applicable)

### D1. `SeparationDataset.__init__` slow metadata scan

Scans up to 70k HDF5 metadata strings at init (train split). One-time cost
but takes ~minutes. Unavoidable given the current filter-by-n_sources design.
Not a correctness issue.

### D2. `CosineAnnealingLR` with `T_max=epochs`: single cycle

After `T_max=30` steps the LR would oscillate back up. Since training runs
exactly 30 epochs this is fine, but the run must not be extended without
updating `T_max` or switching to `CosineAnnealingWarmRestarts`.

### D3. 30 epochs may still not converge for 3/4-source

With the old scheduler, the 3-source model improved -22.41 → -21.82 dB over
18 epochs (~0.6 dB). Cosine annealing should help, but 30 epochs may still
not be full convergence. Plan to report results honestly as "30-epoch run"
rather than claiming convergence.

### D4. Estimated wall time for the new train_all.sh is ~40 hours

The script now trains 3 models × 3 source counts = 9 runs. Based on original
per-epoch timings (986/858/418 s for 2/3/4-src):
- ConvTasNet 30 epochs: ~19h
- CNN-LSTM (simpler, fewer params): estimated ~12–14h
- DPRNN (small N/B/H=64): estimated ~8–12h
- **Total: ~40–45h continuous runtime on Mac Mini M4 Pro**

This is a ~2 day unattended run. Ensure the Mac is on a stable power source,
screen sleep is disabled, and the log file is monitored: `runs/train_all.log`.

---

## Part 4: Dataset Paper Requirements — Updated Status

### What we currently have (after planned re-run)

| Required | Status | Gap |
|----------|--------|-----|
| Dataset stats/characterisation | Partial (check/ figures) | Need publication-quality plots in paper |
| Classical baselines (ICA, NMF) | Needs re-run (metric fixed, N=150) | Re-run required before citing numbers |
| DL — ConvTasNet | Needs re-run (scheduler fixed, 30 epochs) | Re-run required |
| DL — CNN-LSTM | Planned in new train_all.sh | Will be done in the 40h run |
| DL — DPRNN | Planned in new train_all.sh | Will be done in the 40h run |
| Per-mixing-mode breakdown | Not yet implemented | Needs analysis script after training |
| Per-standard performance | Not yet implemented | Needs analysis script after training |
| SNR/SIR stratified analysis | Not yet implemented | Needs analysis script after training |
| Dataset availability (HuggingFace) | Deferred | Hard blocker for submission |

### Missing experiments still required before submission

**ME1. CNN-LSTM and DPRNN — ADDRESSED in train_all.sh**
Both architectures will be trained as part of the new 9-run script.

**ME2. Per-mixing-mode breakdown (co-channel vs adjacent-channel)**
The `mixing_mode` field is stored in each sample's metadata and in
`baseline_results.json`. After training, a post-hoc analysis script should
group test results by mixing_mode and report mean SI-SINR for each cell:
`{co-channel, adjacent-channel} × {2,3,4-source} × {ICA, NMF, CNN-LSTM, ConvTasNet, DPRNN}`.
This is the most important analysis for the paper — reviewers will ask why
the variance is so high (it is entirely explained by the co-/adjacent-channel split).

**ME3. Per-standard performance**
Which source pairs are hardest? (e.g., GSM+5G co-channel vs LTE+LTE adjacent)
At minimum, report SI-SINR grouped by source type combination for 2-source.
Metadata contains the signal standards for each source.

**ME4. SNR/SIR stratified analysis**
Show how model SI-SINR varies with per-sample SIR (-20 to +20 dB).
A binned curve (5 dB bins) of output SI-SINR vs input SIR would directly
demonstrate the dataset's value and the dynamic range of difficulty.

**ME5. Dataset availability — hard blocker**
No venue will accept a dataset paper without a public link.
`src/upload_huggingface.py` is ready. This must be done before or at submission.

### Minimum viable experiment set for submission

1. Re-run baselines (ICA, NMF) with fixed metric and N=150
2. ConvTasNet 30-epoch re-run (cosine annealing)
3. CNN-LSTM 30-epoch run
4. Co-channel vs adjacent-channel breakdown for all methods
5. Dataset uploaded to HuggingFace

DPRNN, per-standard breakdown, and SNR curves are strongly recommended but
can go to supplementary if time is tight.

---

## Summary: What Was Fixed, What Remains

| Item | Was | Now |
|------|-----|-----|
| `scheduler.step()` bug | BLOCKING | RESOLVED |
| SI-SINR zero-mean in baselines | Bug | RESOLVED — re-run baselines required |
| Import inside eval loop | Minor | RESOLVED |
| `--num-workers` on macOS | 4 (slow) | 0 (correct) |
| `--batch-size` mismatch | 16 | 8 (matches original) |
| Scheduler state in checkpoint | Not saved | Saved and restored |
| Baseline N and sampling | 30, sequential | 150, random seed=42 |
| Dead `rng` code | Present | Removed |
| Evaluate() mean bias | Batch-mean | Per-sample mean |
| CNN-LSTM and DPRNN training | Not planned | In new train_all.sh |
| **Wall time warning** | ~12.5h | **~40h for 9 runs** |
| B3: 2T SI-SINR flattening | Open | Still open (paper note) |
| Per-mixing-mode analysis | Missing | Still missing (needs script) |
| HuggingFace upload | Deferred | Still deferred (hard blocker) |
