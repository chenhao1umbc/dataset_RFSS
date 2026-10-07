# Dataset definition (A4 draft, Reviewer, 2026-10-03)

Status: DRAFT for user approval. Facts below come from the Builder's `from_local_claude.md` (commit 22b9ec7),
the code on `dev`, and the Reviewer's own reads of the public Hugging Face files. Nothing here requires regenerating the HDF5 files.

## Decision proposed
Keep the existing HDF5 arrays unchanged. Fix evaluation and training targets, tooling and documentation.
Release "v1.1" means: same data files, corrected documentation, a reference-builder utility, corrected benchmark, corrected paper.
Tag the current Hugging Face revision as `v1.0` so the history is clear.

## Definitions
- **Stored source (`source_signals[i, k]`)**: waveform of source k after TDL channel and the hardware impairments, at its native rate,
  zero-padded. Before resampling, frequency shift, power scaling and AWGN. Its true length is the index of the last non-zero sample plus one.
- **Aligned reference**: `normalize_power( freq_shift( pad( resample( stored source ) ) ), power_ratios_db[k] )`; `freq_shift` only in adjacent-channel mode.
  This is exactly the additive term of source k in the noiseless mixture.
- **Mixture (`mixed_signals[i, :signal_lengths[i]]`)**: sum of aligned references plus AWGN at `snr_db`.
- **Task for the benchmark**: separate the mixture into the aligned references (post-channel components), scored with PI-SI-SINR.
  State in the paper that models are not asked to equalize the channel or remove impairments.
- **MIMO**: not modeled. `mimo_config` is vestigial.

## Evidence
- Forward-model rebuild of 204 test samples: mixture length matches in 204/204; residual versus expected AWGN median -0.002 dB (5th-95th percentile about -0.06 to +0.04 dB).
- Reference mismatch: adjacent-channel mixture-versus-reference SI-SINR median -39.7 dB (current) vs -5.9 dB (aligned); co-channel -8.2 vs -5.5 dB.
- Independent Reviewer check on the public `rfss_single.h5` (20 samples across standards): GSM true length 1,890 vs nominal 2,166; 5G NR 30,660 / 61,348 / 122,640 / 122,696 vs nominal 30,720 / 61,440 / 122,880; LTE and UMTS exact.

## Consequences
1. Every existing benchmark number is invalid (baselines and deep models). This includes Tables I and II in arXiv 2604.00398 and the July `dev` paper.
2. The paper's "adjacent-channel is much harder" finding is mostly a reference artefact (ICA gap shrinks from 9.4 dB to 0.8 dB with correct references).
3. Deep models must be retrained on aligned targets before any claim about them is made.
4. The adjacent-channel limitation paragraph in the April paper (Section VII) is replaced by this fix.

## Open points for the user
- Confirm "v1.1 = same files + corrections" is what you meant by "corrections".
- Confirm that no regeneration of the 110 GB data is wanted (not needed as far as the evidence shows).
