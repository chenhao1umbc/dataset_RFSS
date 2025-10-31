# RFSS Project Task Tracking

## Project Status: RESTART FROM SCRATCH
- Previous AI agent code moved to old_agent/ (reference only, not trusted)
- Paper exists (2508.12106v1.pdf) but contains unvalidated claims
- No dataset generated, no experiments run
- Starting fresh with systematic code verification and validation

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
- [ ] Create src/generate_dataset.py integration script (ParameterSampler → generators → channels → mixer → DatasetWriter)
- [ ] Generate 100 single-standard samples with full parameter variations (25 per standard)
- [ ] Generate 100 2-source mixed samples (co-channel and adjacent-channel)
- [ ] Generate 100 3-source mixed samples (realistic coexistence scenarios)
- [ ] Generate 100 4-source mixed samples (GSM+UMTS+LTE+5G full spectrum)
- [ ] Generate 100 MIMO mixed samples (2x2 and 4x4 configurations)
- [ ] Validate signal quality metrics (PAPR, bandwidth, power) for all samples
- [ ] Validate ground truth preservation (source signals, channel parameters, metadata)
- [ ] Validate parameter coverage (ensure all dimensions are represented)
- [ ] Create visualization of parameter distribution coverage
- [ ] Document demonstration dataset statistics and findings

### 2.3 Dataset Infrastructure
- [ ] Test HDF5 storage format (already designed in dataset_parameters.md)
- [ ] Verify metadata schema implementation (already defined)
- [ ] Test DatasetWriter class with sample data
- [ ] Implement PyTorch Dataset class for loading samples
- [ ] Implement data loader with batching and shuffling
- [ ] Implement reproducibility framework (random seed management, version tracking)
- [ ] Create dataset validation utilities (check integrity, coverage, statistics)
- [ ] Create dataset inspection tools (query by parameters, visualize samples)
- [ ] Test infrastructure with demonstration dataset
- [ ] Document dataset format specification

### 2.4 Full Dataset Generation
- [ ] Confirm final dataset size: 100k samples (per dataset_parameters.md)
- [ ] Implement train/val/test split: 70/15/15 random (per dataset_parameters.md)
- [ ] Implement progress tracking and checkpointing for generation
- [ ] Generate training set with balanced parameter sampling
- [ ] Generate validation set with same distribution as training
- [ ] Generate test set ensuring coverage of all scenario types
- [ ] Run comprehensive quality assurance on full dataset
- [ ] Calculate and verify final dataset statistics (size, coverage, balance)
- [ ] Create dataset documentation (README, parameter distributions, sample access)
- [ ] Estimate storage requirements and optimize compression
- [ ] Package dataset with loading utilities and documentation

## Phase 3: Baseline Experiments
### 3.1 Traditional Methods Implementation
- [ ] Fix bugs in ICA implementation (scipy.signal namespace)
- [ ] Fix bugs in NMF implementation
- [ ] Implement SINR evaluation metric correctly
- [ ] Implement permutation-invariant matching
- [ ] Test baseline code on simple synthetic signals

### 3.2 Baseline Performance Evaluation
- [ ] Run ICA on 2-source mixtures
- [ ] Run NMF on 2-source mixtures
- [ ] Run ICA on 3-source mixtures
- [ ] Run NMF on 3-source mixtures
- [ ] Run ICA on 4-source mixtures
- [ ] Run NMF on 4-source mixtures
- [ ] Document actual baseline performance (resolve +15 dB vs -20 dB discrepancy)

### 3.3 Baseline Analysis
- [ ] Analyze failure modes of ICA
- [ ] Analyze failure modes of NMF
- [ ] Determine if baselines succeed or fail
- [ ] Create performance visualization plots
- [ ] Write baseline results summary

## Phase 4: Deep Learning Development
### 4.1 Model Architecture Review and Fix
- [ ] Review CNN-LSTM from old_agent
- [ ] Identify training instability root causes
- [ ] Fix or redesign CNN-LSTM architecture
- [ ] Review Conv-TasNet from old_agent
- [ ] Fix or redesign Conv-TasNet
- [ ] Review DPRNN from old_agent
- [ ] Fix or redesign DPRNN
- [ ] Document all architecture choices

### 4.2 Training Infrastructure
- [ ] Implement training script with proper error handling
- [ ] Implement evaluation script
- [ ] Add experiment tracking (tensorboard/wandb)
- [ ] Add checkpoint saving and loading
- [ ] Add early stopping
- [ ] Add visualization tools for training progress
- [ ] Test training pipeline on tiny dataset

## Phase 5: Deep Learning Experiments
### 5.1 Model Training
- [ ] Train CNN-LSTM on 2-source separation
- [ ] Train Conv-TasNet on 2-source separation
- [ ] Train DPRNN on 2-source separation
- [ ] Perform hyperparameter tuning
- [ ] Train best models on 3-source separation
- [ ] Train best models on 4-source separation

### 5.2 Model Evaluation
- [ ] Evaluate all models on test set
- [ ] Compare against ICA baseline
- [ ] Compare against NMF baseline
- [ ] Analyze per-standard separation performance
- [ ] Measure actual SINR improvements
- [ ] Create result tables
- [ ] Create performance comparison figures

### 5.3 Analysis and Ablation
- [ ] Analyze what models learned
- [ ] Identify failure cases
- [ ] Perform ablation studies if relevant
- [ ] Test generalization to unseen scenarios
- [ ] Document all experimental findings

## Phase 6: Paper Revision
### 6.1 Results Validation
- [ ] Replace all fabricated results with actual experimental results
- [ ] Fix ICA/NMF performance claims (verify correct SINR sign)
- [ ] Update CNN-LSTM performance with actual numbers
- [ ] Update Conv-TasNet performance with actual numbers
- [ ] Update DPRNN performance with actual numbers
- [ ] Verify dataset size claims match reality
- [ ] Verify 3GPP compliance claims with actual measurements

### 6.2 Theoretical Improvements
- [ ] Add detailed SINR evaluation metric definition
- [ ] Add complete source separation problem formulation
- [ ] Add CNN-LSTM architecture description with diagrams
- [ ] Add Conv-TasNet architecture description with diagrams
- [ ] Add DPRNN architecture description with diagrams
- [ ] Add loss function definitions for all models
- [ ] Add training procedure details (optimizer, learning rate, epochs, batch size)
- [ ] Explain permutation problem and how it was solved
- [ ] Specify dataset generation parameters (SNR ranges, mixing ratios, scenarios)
- [ ] Define train/val/test split ratios and exact sizes
- [ ] Clarify 3GPP compliance metric definition
- [ ] Add evaluation methodology section
- [ ] Add ablation study results if conducted

### 6.3 Paper Writing
- [ ] Rewrite experimental results section with actual data
- [ ] Add detailed methodology section for deep learning models
- [ ] Create publication-quality figures from real experiments
- [ ] Add discussion of results and limitations
- [ ] Add comparison with state-of-the-art methods
- [ ] Update abstract with actual contributions
- [ ] Proofread entire paper
- [ ] Format for target journal/conference
- [ ] Prepare supplementary materials if needed

## Open Questions to Resolve
- [ ] What is the correct ICA/NMF baseline performance? (+15 dB or -20 dB SINR?)
- [ ] What dataset size do we actually need? (52,847 seems arbitrary)
- [ ] What should the train/val/test split be?
- [ ] What are realistic SNR/power ranges for mixing scenarios?
- [ ] Can we reuse any deep learning code or start completely fresh?
- [ ] What is the target journal/conference for submission?
- [ ] Should we focus on 2-source separation first or multi-source?
- [ ] Do we need all three DL architectures or focus on best one?

## Notes
- All tasks should be tracked here
- Mark tasks as complete with [x] when done
- Update status sections as work progresses
- Keep this file synchronized with actual progress
