# RFSS Project Task Tracking

## Project Status: Phase 5 complete — all experiments done, results in experiment_results.md
- Phase 1 (all subphases): COMPLETE
- Phase 2.1: COMPLETE
- Phase 2.2: COMPLETE
- Phase 2.3: COMPLETE
- Phase 2.4: COMPLETE — HuggingFace upload deferred until paper is finished
- Datasets: data/rfss_dataset.h5 (100k, 103 GB) + data/rfss_single.h5 (4k, 1.3 GB)
- Phase 3 (baselines): COMPLETE — ICA/NMF fail (SI-SINR -23 to -37 dB); paper claims fabricated
- Phase 4 (deep learning): COMPLETE — 3 models + training infrastructure; smoke test verified
- Phase 5 (DL experiments): COMPLETE — ConvTasNet beats ICA/NMF on all 3 source counts
- Phase 6 (paper revision): IN PROGRESS — venue decided (NeurIPS D&B primary, IEEE TWC fallback); rewrite from scratch in progress

## Phase 1: Code Foundation Verification
### 1.1 Signal Generators - COMPLETE
- [x] Review GSM generator from old_agent
- [x] Test GSM generator produces valid signals
- [x] Review UMTS generator from old_agent
- [x] Test UMTS generator produces valid signals
- [x] Review LTE generator from old_agent
- [x] Test LTE generator produces valid signals
- [x] Review 5G NR generator from old_agent
- [x] Test 5G NR generator produces valid signals
- [x] Validate all generators against 3GPP specs

### 1.2 Channel Models - COMPLETE
**Note**: Complete redesign from scratch based on 3GPP specifications
- [x] Research 3GPP TR 38.901 channel models (TDL/CDL specifications)
- [x] Extract numerical parameters for all TDL models from official sources
- [x] Implement 3GPP TDL-A model (NLOS, low delay spread, 23 taps)
- [x] Implement 3GPP TDL-B model (NLOS, medium delay spread, 23 taps)
- [x] Implement 3GPP TDL-C model (NLOS, high delay spread, 24 taps)
- [x] Implement 3GPP TDL-D model (LOS, low delay spread, 13 taps, K=13.3 dB)
- [x] Implement 3GPP TDL-E model (LOS, high delay spread, 14 taps, K=22 dB)
- [x] Implement Jakes' sum-of-sinusoids model for time-varying fading
- [x] Implement Rayleigh fading with Doppler effects
- [x] Implement Rician fading with Doppler effects
- [x] Implement CFO (Carrier Frequency Offset) per 3GPP TS 38.104/38.101
- [x] Implement SFO (Sampling Frequency Offset) with resampling
- [x] Implement I/Q imbalance per 3GPP TS 36.101 (image rejection requirements)
- [x] Implement DC offset (LO leakage modeling)
- [x] Implement phase noise as Wiener process per 3GPP TS 25.102
- [x] Implement PA nonlinearity using Rapp model
- [x] Implement MIMO channel generation with time-varying fading
- [x] Implement MIMO channel application with spatial correlation
- [x] Create comprehensive unit tests for all TDL models
- [x] Create unit tests for Jakes' fading models (Rayleigh/Rician)
- [x] Create unit tests for all hardware impairments (CFO/SFO/IQ/DC/PN/PA)
- [x] Create unit tests for MIMO channel generation and application
- [x] Create demonstration script (run_channel.py) showing all effects
- [x] Document all implementations with 3GPP citations in paper/amendment.md
- [x] Cross-verify numerical parameters with HermesPy implementation
- [x] Remove unused imports (numpy, Optional) for code cleanliness

### 1.3 Signal Mixing - COMPLETE

**A. Fix Current Issues**
- [x] Fix import path in check/unit_test_channel.py (utils_channel → src.utils_channel)
- [x] Update pyproject.toml testpaths to point to check/ instead of tests/
- [x] Fix all import paths in src/run_*.py files (added src prefix)
- [x] Fix import paths in src/utils_*.py files (added src prefix)

**B. Research and Specifications**
- [x] Research 3GPP coexistence scenarios (LTE-NR DSS, GSM-UMTS-LTE, spectrum sharing)
- [x] Extract realistic interference parameters (ACIR ~32 dB, ACLR 30-45 dB, SIR -20 to +20 dB)
- [x] Document mixing scenarios in paper/mixing_scenarios.md

**C. Core Mixing Infrastructure (src/utils_mixing.py)**
- [x] Implement SignalMixer class with per-source independent channels
- [x] Implement per-source channel application (different TDL/CFO/impairments per source)
- [x] Implement timing offset support (asynchronous signal arrival)
- [x] Implement co-channel mixing (all sources at baseband, hardest case)
- [x] Implement adjacent-channel mixing (frequency shifting with realistic ACIR)
- [x] Implement realistic power ratio control (SIR: -20 to +20 dB for near-far)
- [x] Implement ground truth preservation (source signals, channels, metadata)
- [x] Implement comprehensive metadata output for reproducibility

**D. MIMO Spatial Mixing**
- [x] Implement MIMO mixer with spatial correlation
- [x] Implement per-antenna different mixtures (spatial diversity)
- [x] Support 2x2, 4x4, 8x8 MIMO configurations
- [x] Validate spatial correlation properties
- [x] Test MIMO mixing with time-varying channels

**E. Realistic Mixing Scenarios Definition**
- [x] Define 2-source scenarios (GSM+LTE, UMTS+5G, LTE+5G, GSM+UMTS, UMTS+LTE, GSM+5G)
- [x] Define 3-source scenarios (GSM+UMTS+LTE, UMTS+LTE+5G, GSM+LTE+5G, GSM+UMTS+5G)
- [x] Define 4-source scenario (GSM+UMTS+LTE+5G)
- [x] Define co-channel vs adjacent-channel configurations per scenario
- [x] Define power ratio distributions (equal, near-far, realistic SIR ranges)
- [x] Document all scenarios with 3GPP coexistence references

**F. Comprehensive Unit Tests (check/unit_test_mixing.py)**
- [x] Test power ratio accuracy after mixing (validate SIR)
- [x] Test frequency offset accuracy for adjacent-channel mixing
- [x] Test timing offset handling and edge cases
- [x] Test MIMO spatial correlation validation
- [x] Test ground truth preservation for all source signals
- [x] Test metadata completeness and correctness
- [x] Test 2-source basic mixing
- [x] Test 3-source realistic scenario
- [x] Test 4-source near-far scenario
- [x] Test MIMO 2x2 and 4x4 configurations
- [x] Test mixer clear and source info methods

**G. Demonstration Script and Notebook**
- [x] Create src/run_mixing.py demonstration script
- [x] Create check/demo_phase1_3.ipynb comprehensive demonstration
- [x] Visualize co-channel mixing
- [x] Visualize adjacent-channel mixing
- [x] Visualize MIMO mixing
- [x] Visualize power ratio effects
- [x] Demonstrate ground truth preservation
- [x] Show realistic 2/3/4-source mixing scenarios

### 1.4 Development Environment
- [x] Install missing dependencies (pytest, dev tools)
- [x] Fix all unit tests
- [x] Remove all emojis from codebase (66 files affected)

## Phase 2: Dataset Generation
### 2.1 Parameter Space Definition - COMPLETE
- [x] Define bandwidth options per standard (GSM: 200kHz, UMTS: 5MHz, LTE: 1.4/3/5/10/15/20MHz, 5G: flexible)
- [x] Define modulation schemes per standard (e.g., LTE: QPSK/16QAM/64QAM/256QAM)
- [x] Define SNR ranges for training (e.g., -10 to +30 dB for realistic conditions)
- [x] Define channel model distribution (TDL-A/B/C/D/E proportions: NLOS vs LOS scenarios)
- [x] Define hardware impairment parameter ranges (CFO: 0.05-5 ppm, I/Q: 0.1-3 dB, etc.)
- [x] Define Doppler frequency ranges (0-500 Hz for different mobility scenarios)
- [x] Define mixing mode distribution (co-channel vs adjacent-channel proportions)
- [x] Define MIMO configuration distribution (SISO vs 2x2 vs 4x4 proportions)
- [x] Define source count distribution (1/2/3/4-source proportions)
- [x] Define power ratio ranges for mixed scenarios (SIR: -20 to +20 dB)
- [x] Calculate total parameter space size and combinations
- [x] Document all parameter choices with 3GPP justifications in paper/dataset_parameters.md

### 2.2 Demonstration Dataset (Small Scale Validation)
- [x] Create src/generate_dataset.py integration script (ParameterSampler → generators → channels → mixer → DatasetWriter)
- [x] Generate single-standard samples — 1000/standard = 4000 total in data/rfss_single.h5; seed offset 2,000,000; verified clean (no NaN/Inf, no zero-length, all standards represented with full parameter variety)
- [x] Generate 2-source mixed samples — present in 100k; quality validated by check/quality_check.py (100 samples, all checks pass)
- [x] Generate 3-source mixed samples — present in 100k; quality validated by check/quality_check.py (100 samples, all checks pass)
- [x] Generate 4-source mixed samples — present in 100k; quality validated by check/quality_check.py (100 samples, all checks pass)
- [x] Validate MIMO config distribution in 100k — verified via coverage_analysis(): 1x1=50.2%, 2x2=30.1%, 4x4=19.7%; all within 0.3% of intended weights
- [x] Validate signal quality metrics (PAPR, power, SNR) — check/quality_check.py, 300 sampled checks, all pass
- [x] Validate ground truth preservation — source count matches metadata for all 300 checked samples; design: sources stored at native sample rates, mixed at max rate; training code must upsample before source comparison
- [x] Validate full parameter coverage across 100k — coverage_analysis() in quality_check.py scans 20k samples; all distributions within 0.003 of intended weights (num_sources, mixing_mode, MIMO all OK)
- [x] Create parameter distribution visualization — check/demo_phase2.ipynb executed (9/9 cells), 4 figures saved: fig_distributions_pie.png, fig_distributions_bar.png, fig_snr_siglen.png, fig_sample_inspect.png
- [x] Fix remaining issues in check/quality_check.py — all fixed and verified 2026-02-21:
  1. `_check_dist` extracted to module level (was nested inside `coverage_analysis()`)
  2. `collect_single_indices` now uses `SINGLE_SCAN_STEP = 10` for spread sampling across full range
- [x] Persist quality check results — saves to check/quality_check_results.json

### 2.3 Dataset Infrastructure
- [x] Test HDF5 storage format — validated by successful generation of 104k samples
- [x] Verify metadata schema implementation — schema consistent across single-source and multi-source files
- [x] Test DatasetWriter class with sample data — used in production for 104k samples; flush-before-checkpoint pattern confirmed safe
- [x] Implement PyTorch Dataset class for loading samples — RFSSDataset in src/utils_dataset.py (70/15/15 sequential split)
- [x] Implement data loader with batching and shuffling — create_dataloader + _collate_rfss_batch in src/utils_dataset.py
- [x] Implement reproducibility framework — master_seed=42; multi-source seeds 42+sample_id; single-source seeds 42+2,000,000+global_idx
- [x] Create dataset validation utilities — validate_dataset() and inspect_sample() in src/utils_dataset.py
- [x] Create dataset inspection tools — inspect_sample() + check/quality_check.py
- [x] End-to-end DataLoader test — check/unit_test_dataset.py: 15 tests (RFSSDataset, DataLoader batch shapes/dtypes, split boundaries); 48/48 total pass
- [x] Document dataset format specification — paper/dataset_spec.md: HDF5 layout, chunking, signal layout, native rates, metadata JSON schema, split logic, PyTorch interface, parameter distributions table, reproducibility seeds

### 2.4 Full Dataset Generation
- [x] Confirm final dataset size: 100k multi-source + 4k single-source — confirmed via HDF5 actual_samples attr
- [x] Implement train/val/test split: 70/15/15 at load time in RFSSDataset (sequential index split)
- [x] Implement progress tracking and checkpointing — flush-before-checkpoint, 1000-sample intervals for multi-source, 500 for single-source
- [x] Generate full dataset: data/rfss_dataset.h5 — 100,000 samples, 103 GB, completed 2026-02-21 05:06
- [x] Generate single-source dataset: data/rfss_single.h5 — 4,000 samples, 1.3 GB, completed 2026-02-21 11:09
- [x] Fix code quality issues in src/generate_dataset.py — all fixed and verified 2026-02-21 (48/48 tests pass):
  1. `import numpy as np` — removed (unused)
  2. `List` in typing imports — removed (unused)
  3. `STANDARDS` import moved to top-level (was inside `generate_single_source_dataset()`)
  4. `import argparse` moved to top-level (was inside `main()`)
  5. Dead `sample_id = global_idx % num_samples_per_standard` variable removed; config uses global_idx directly
- [x] Run quality assurance — check/quality_check.py: 300 sampled checks across 2/3/4-source groups, all pass; spot-checks around crash-zone boundary (samples 38999/39000) confirm data integrity
- [x] Full parameter coverage analysis — done via coverage_analysis() in quality_check.py; results in check/quality_check_results.json
- [x] Create dataset documentation — dataset card embedded in src/upload_huggingface.py (CC-BY-4.0, metadata YAML, usage examples)
- [x] Storage verified: 103 GB multi-source + 1.3 GB single-source, gzip compression level 6
- [ ] HuggingFace upload — deferred until paper is accepted/submitted; script ready at src/upload_huggingface.py

## Phase 3: Baseline Experiments — COMPLETE (2026-03-04)
### 3.1 Traditional Methods Implementation — COMPLETE
- [x] Fix bugs in ICA implementation (scipy.signal namespace, nested functions, scale-invariant SINR)
- [x] Fix bugs in NMF implementation (scipy.signal namespace, noverlap param, sklearn alpha_W/alpha_H)
- [x] Implement SI-SINR metric (scale-invariant, handles amplitude ambiguity)
- [x] Implement permutation-invariant matching (Hungarian algorithm)
- [x] Test baseline code on synthetic signals (7.68 MHz, 7680 samples)

### 3.2 Baseline Performance Evaluation — COMPLETE
- [x] Run ICA on 2-source mixtures — mean SI-SINR: -31.95 dB ± 10.02
- [x] Run NMF on 2-source mixtures — mean SI-SINR: -23.04 dB ± 16.82
- [x] Run ICA on 3-source mixtures — mean SI-SINR: -35.65 dB ± 8.17
- [x] Run NMF on 3-source mixtures — mean SI-SINR: -30.16 dB ± 11.35
- [x] Run ICA on 4-source mixtures — mean SI-SINR: -36.82 dB ± 10.93
- [x] Run NMF on 4-source mixtures — mean SI-SINR: -26.43 dB ± 16.59
- [x] Document actual baseline performance — paper claims (+15.2/+18.3 dB) definitively fabricated
- Results saved to check/baseline_results.json (N=30 per group, test split)

### 3.3 Baseline Analysis — COMPLETE
- [x] Analyze failure modes of ICA — Gaussian OFDM signals violate ICA non-Gaussianity assumption;
      underdetermined SISO problem; performance degrades with source count
- [x] Analyze failure modes of NMF — co-channel mixing makes spectral patterns overlap;
      NMF has higher variance (occasional adjacent-channel cases up to +3 dB)
- [x] Determine if baselines succeed or fail — FAIL; SI-SINR -23 to -37 dB (worse than raw mixture)
- [ ] Create performance visualization plots — deferred; results in baseline_results.json
- [x] Write baseline results summary — in working_log.md 2026-03-04

## Phase 4: Deep Learning Development — COMPLETE (2026-03-04)
### 4.1 Model Architecture Review and Fix — COMPLETE
- [x] Review CNN-LSTM from old_agent — fixed wrong output (per-standard labels), no PIT, nested functions
- [x] Identify training instability root causes — no permutation-invariant loss; fixed output assuming known standards; NaN from non-scale-invariant loss
- [x] Fix or redesign CNN-LSTM architecture — CNN encoder + BiLSTM + ConvTranspose1d decoder; n_sources generic output
- [x] Review Conv-TasNet from old_agent — wrong output structure; no PIT; numpy import; fixed skip_channels logic
- [x] Fix or redesign Conv-TasNet — clean TCN with GlobalLayerNorm, shared decoder, PIT SI-SINR loss
- [x] Review DPRNN from old_agent — correct structure but no PIT, wrong output labels, numpy import
- [x] Fix or redesign DPRNN — clean _DualRNNBlock, correct inter/intra LSTM sizes, PIT loss
- [x] Document all architecture choices — see working_log.md 2026-03-04

### 4.2 Training Infrastructure — COMPLETE
- [x] Implement training script with proper error handling — src/train.py
- [x] Implement evaluation script — Trainer.evaluate() in train.py
- [x] Add experiment tracking (tensorboard) — SummaryWriter in Trainer (optional)
- [x] Add checkpoint saving and loading — Trainer.save_checkpoint(), keep 3 best by val_loss
- [x] Add early stopping — ReduceLROnPlateau scheduler (patience configurable)
- [x] Add visualization tools for training progress — tensorboard logs in runs/
- [x] Test training pipeline on tiny dataset — smoke test passed: all 3 models train, loss decreases

## Phase 5: Deep Learning Experiments — COMPLETE (2026-03-05)
### 5.1 Model Training — COMPLETE
- [x] Train Conv-TasNet on 2-source separation — best val SI-SINR: -20.18 dB (epoch 18)
- [x] Train best models on 3-source separation — Conv-TasNet best val: -21.82 dB (epoch 18)
- [ ] Hyperparameter tuning — deferred; single run sufficient for paper
- [x] Train best models on 4-source separation — Conv-TasNet best val: -22.64 dB (epoch 19)

### 5.2 Model Evaluation — COMPLETE
- [x] Evaluate Conv-TasNet on test set — N=150 per source count
- [x] Compare against ICA baseline — ConvTasNet +12 to +14 dB improvement
- [x] Compare against NMF baseline — ConvTasNet +2 to +9 dB improvement
- [ ] Analyze per-standard separation performance — deferred to Phase 6
- [x] Measure actual SINR improvements — see experiment_results.md
- [x] Create result tables — in experiment_results.md
- [ ] Create performance comparison figures — deferred to Phase 6

### 5.3 Analysis and Ablation
- [ ] Analyze what models learned
- [ ] Identify failure cases
- [ ] Perform ablation studies if relevant
- [ ] Test generalization to unseen scenarios
- [ ] Document all experimental findings

## Phase 6: Paper Revision (Full Rewrite)

### Venue Strategy (decided 2026-03-07)
- **Primary target**: NeurIPS 2026 Datasets & Benchmarks track
  - No APC; maximum dataset visibility; ML community adoption
  - Requires public dataset link before submission → HuggingFace upload needed
  - Deadline: typically May–June for December conference
- **Fallback**: IEEE Transactions on Wireless Communications (TWC)
  - No APC (subscription access); IF ~10; channel modeling angle fits well
  - Better fit than TSP for a dataset+benchmark paper (TSP expects algorithmic novelty)
- **Companion**: ICASSP 2026 short paper (5 pages) for SP community visibility
- **Interim**: arXiv preprint already live (2508.12106v1)

### HuggingFace
- Use placeholder URL `https://huggingface.co/datasets/rfss/rfss-dataset` in paper for now
- User will create the actual repo and upload when ready to submit
- Upload script: `src/upload_huggingface.py` (ready)

### 6.1 Figure Generation (Python scripts → paper/figures/)
- [ ] Fig 1: STFT spectrograms of all 4 standards (run signal generators)
- [ ] Fig 2: Dataset construction pipeline diagram (matplotlib)
- [ ] Fig 3: Dataset statistics — source count distribution, mixing mode, standard combinations (scan HDF5 metadata)
- [ ] Fig 4: Signal characterization — PAPR, PSD, amplitude distribution per standard
- [ ] Fig 5: Benchmark results bar chart — all methods × source counts (from breakdown_results.json)
- [ ] Fig 6: Co-channel vs adjacent-channel breakdown chart (from breakdown_results.json)

### 6.2 Paper Rewrite (from scratch, NeurIPS D&B style)
Target: IEEEtran journal class (compatible with NeurIPS D&B extended; clean two-column)
- [ ] Abstract (200 words max; what/why/how many/key result)
- [ ] Sec 1: Introduction — motivation, gap, 4 bullet contributions
- [ ] Sec 2: Related Work — RF datasets (RadioML, GNU Radio), SP source separation datasets
- [ ] Sec 3: Dataset Construction — signal gen (equations), channel model (TDL + impairments), mixing modes, parameters
- [ ] Sec 4: Dataset Characterization — statistics, PAPR/PSD/amplitude, 3GPP compliance validation
- [ ] Sec 5: Benchmark Experiments — setup, PI-SI-SINR definition, all results tables, analysis
- [ ] Sec 6: Data Access & Format — HDF5 layout, PyTorch DataLoader API, HuggingFace link, license
- [ ] Sec 7: Limitations — adjacent-channel reference convention, no 6G waveforms, single-antenna receiver
- [ ] Sec 8: Conclusion
- [ ] References (original [1–15] + new [16–24]: TDL, Le Roux SI-SINR, Conv-TasNet, DPRNN, Jakes, Rapp)

### 6.3 Quality Checklist (before reviewer sees it)
- [ ] Every number in the paper traceable to code or experiment_results.md
- [ ] All figures generated from actual data (no placeholders)
- [ ] No claim without citation or experimental evidence
- [ ] Abstract, intro, conclusion are consistent with each other
- [ ] Proofread for grammar and flow
- [ ] Compile clean with no Overfull warnings

### 6.4 Dataset Publication
- [ ] HuggingFace upload — user creates repo at submission time; script ready at src/upload_huggingface.py
- [ ] Use placeholder URL `https://huggingface.co/datasets/rfss/rfss-dataset` until then

## Open Questions — Resolved
- [x] Correct ICA/NMF performance → actual PI-SI-SINR in experiment_results.md (fabricated claims removed)
- [x] Dataset size → 100k multi-source + 4k single-source
- [x] Train/val/test split → 70/15/15 at load time
- [x] SNR/power ranges → SNR −10 to +40 dB; SIR −20 to +20 dB
- [x] DL code → written from scratch (src/models.py, src/train.py); all 9 runs complete
- [x] Target venue → NeurIPS D&B primary; IEEE TWC fallback
- [x] Source count focus → all three (2/3/4) benchmarked
- [x] HuggingFace → placeholder URL in paper; real upload at submission time

## Notes
- All tasks should be tracked here
- Mark tasks as complete with [x] when done
- Update status sections as work progresses
- Keep this file synchronized with actual progress
