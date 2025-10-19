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

### 1.3 Signal Mixing - NOT VERIFIED
- [ ] Review signal mixer code
- [ ] Test frequency shifting
- [ ] Test power normalization
- [ ] Validate mixed signals preserve source characteristics

### 1.4 Development Environment
- [ ] Install missing dependencies (pytest, dev tools)
- [ ] Fix all unit tests
- [ ] Set up code quality tools (black, flake8, mypy)
- [ ] Remove all emojis from codebase (66 files affected)

## Phase 2: Dataset Generation
### 2.1 Single Standard Signals (Demonstration)
- [ ] Generate 100 GSM test samples
- [ ] Generate 100 UMTS test samples
- [ ] Generate 100 LTE test samples
- [ ] Generate 100 5G NR test samples
- [ ] Validate signal quality metrics (PAPR, bandwidth, EVM)

### 2.2 Mixed Signal Scenarios (Demonstration)
- [ ] Define mixing scenarios (2-source, 3-source, 4-source)
- [ ] Define SNR/power ranges for realistic mixing
- [ ] Generate 100 2-source mixed samples
- [ ] Generate 100 3-source mixed samples
- [ ] Validate ground truth preservation

### 2.3 Full Dataset Generation
- [ ] Determine final dataset size based on experimental needs
- [ ] Define train/val/test split ratios
- [ ] Generate training set
- [ ] Generate validation set
- [ ] Generate test set
- [ ] Document dataset statistics
- [ ] Create dataset loading utilities

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
