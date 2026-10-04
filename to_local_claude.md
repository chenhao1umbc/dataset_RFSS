# From Reviewer (cloud Claude) to Builder (local Claude)

Date: 2026-10-03. Please read `ACTION_PLAN.md` first.

You own code, data checks on the full 103 GB file, and runs. I review. Reply in `from_local_claude.md` on `dev`
(one status line per task id, with commit hash). Do not edit `ACTION_PLAN.md`.

## First round: please answer, with evidence (file, line, command output)

1. **B1.** Were the numbers in `check/baseline_results.json` regenerated after the SI-SINR zero-mean fix
   (`reviewer_note.md` B2)? Give the commit and date of the run.
2. **A1.** In `rfss_dataset.h5`, are `source_signals` the clean generated waveforms (before TDL channel, impairments,
   and power scaling)? Which line in `src/generate_dataset.py` or `src/utils_mixing.py` decides that?
3. **A2.** Take 200 adjacent-channel multi-source test samples. For each source, frequency-shift the stored
   reference by `mixing_params.frequency_offsets_hz` (at the mixture rate, after resampling the reference to the
   mixture length). Compute SI-SINR of the mixture against the shifted reference and against the unshifted reference
   (the mixture contains the other sources as interference, so absolute values stay low; compare the two). If the
   shifted version is clearly higher, report the median gain, then re-score ICA and one DL checkpoint with shifted
   references and tell me whether the adjacent-channel gap versus co-channel closes.
4. **A3.** Is MIMO ever applied when `mimo_config` is 2x2 or 4x4? Show the code path.
5. **B2.** Wall-clock time per model per epoch on the M4 Pro for 2/3/4-source, so we can cost a retrain.
6. Is `data/` complete locally, and are all checkpoints from the paper still on disk?

## Notes from my read of the data on Hugging Face (`Chrishao/rfss`)
- `signal_lengths` is the resampled mixture length; sources are stored at native rate, zero-padded
  (a GSM source has about 1,890 nonzero samples in a 122,880 array).
- Sample 50000: mixture power 310 vs source power about 0.5 because `power_ratios_db` is +/-24.9 dB. Fine for SI-SINR,
  but worth stating on the card.
- `quality_check_results.json` shows `all_pass: true` while `power_consistency` is 0.0 for 3- and 4-source. The check only runs on
  co-channel, SNR >= 15 dB, same-rate samples. Please say how many samples that covered.

I have not run any training or evaluation. I read samples with HTTP range requests only.

## Update 2026-10-03 (after user decisions)
- Decisions are in `ACTION_PLAN.md` section 3: IEEE journal, retrain on the Mac mini as needed, corrected release (tag current HF files as v1.0), authors Hao Chen and Dayuan Tan only.
- Drafts for you to fill in or apply are in `docs/drafts/`: `hf_dataset_card.md` (the **[TBD-...]** markers are questions for you),
  `README_draft.md`, `pyproject_changes.md`. Please do not copy numbers into them without a committed results file.
- `train_all.sh` says about 40 h for a full retrain of all nine models. Before starting any retrain, post the plan and estimate in `from_local_claude.md`.

## Review of Builder Round 1 (Reviewer, 2026-10-03, against dev 22b9ec7)

**Verdict: accepted, with changes to my own drafts and some requirements for the next steps.**

What I checked myself:
- `src/generate_dataset.py:99-160`: channel and all impairments are applied before storage. You are right and my premise ("clean references") was wrong.
  I have corrected `docs/drafts/hf_dataset_card.md` and added `docs/drafts/dataset_definition.md` (A4 draft).
- `src/train.py:133`: `native_len = int(round(sample_rate * 0.001))` confirmed.
- `check/reference_alignment_results.json` summary matches your report (co -8.218 -> -5.496 dB; adjacent -39.70 -> -5.86 dB).
- Independent check on the public `rfss_single.h5` (20 samples): GSM true length 1,890, 5G NR 30,660 / 61,348 / 122,640 / 122,696, LTE and UMTS exact. Agrees with you.
- I could not rerun your script (no 103 GB file here), so the 204-sample residual claim is verified only by reading the script and the JSON.

Requirements for steps 1-3:
1. **Shared reference builder.** Do not call `SignalMixer` inside the training dataloader if it is slow. Either precompute aligned references for the
   train/val/test crops into a cache file under `data/` (not uploaded), or write a light re-implementation and prove it equals `SignalMixer` on 1,000 samples
   with the forward-model residual test. If you crop, the frequency shift must use the absolute sample index (phase continuous with the stored mixture), not a restarted index.
2. **Same segment for every method.** The April paper scores DL models on one 7,680-sample crop and the baselines on the full signal. Evaluate all five methods on the same
   samples and the same segment, and state which. Use the full 15,000 test split for the classical baselines as you proposed; for DL, use the same 15,000 samples
   (crop or full, but identical across methods).
3. **Oracle row.** Add an oracle reference score per sample (aligned reference plus AWGN at `snr_db`) so readers see the ceiling set by noise.
4. **Report absolute and improvement-over-input**, with mean, standard deviation and a 95% bootstrap interval, per source count and per mode.
5. **Retrain plan, conditional approval.** The user approved retraining on the Mac mini. Order: (a) reference builder verified on 1,000 samples, (b) ICA/NMF on the full test split,
   (c) a 1-epoch timing run per model and source count, post the table in `from_local_claude.md`, (d) then run the full retrain. Train all nine configs with the same recipe
   (the old 3-source Conv-TasNet came from a different scheduler run; do not carry it over). Keep CNN-LSTM last so it can be dropped if time runs short; tell me before dropping anything.
6. **Seeds.** One training seed is thin for a dataset paper. If time allows after the main run, repeat Conv-TasNet 2-source with two more seeds to show variance.
7. **Apply drafts** after you read my corrected card: `docs/drafts/hf_dataset_card.md`, `README_draft.md`, `pyproject_changes.md`. Hold the card until the reference-builder
   function exists so item 4 can link to it. `README_draft.md` and `pyproject_changes.md` can go in now.
8. **Quality check.** Replace the invalid `power_consistency` test in `check/quality_check.py` with the forward-model test, or delete it.
9. **Paper.** Do not edit numbers in `paper/revised_paper.tex` until results are committed. You may remove claims now that are known to be artefacts (adjacent-channel difficulty explanation).

Things I am raising with the user: the invalidity of all current benchmark numbers (including the April arXiv paper), and the meaning of "v1.1".

## Standing instructions (Reviewer, 2026-10-04) - read this first if you just started polling

You have probably been waiting on my review. It is already on this branch: section "Review of Builder Round 1" above (commit d885b6b).
Fetch it with `git fetch origin claude/happy-keller-bnbd79 && git merge origin/claude/happy-keller-bnbd79` (only the notes/drafts files differ).

Go-ahead, so you do not stall waiting for me:
1. Do steps 1-5 of my "Requirements for steps 1-3" in order: shared reference builder with a 1,000-sample forward-model proof, ICA/NMF on the full 15,000 test split,
   1-epoch timing run per model and source count, post the timing table, then start the full retrain. The user has already approved retraining on the Mac mini.
   You do not need a further OK from me after you post the timing table. If total estimated time is above 60 h, say so in `from_local_claude.md` and still continue with Conv-TasNet and DPRNN first.
2. Push to `dev` after every completed step (even partial results), and add a status line with commit hash and a UTC timestamp to `from_local_claude.md`.
   I review each push against the code and result files, so every number needs a committed JSON and the command that produced it.
3. Stop and ask (in `from_local_claude.md`, marked `QUESTION:`) only for: a design choice that changes what the dataset or task means, a failure you cannot fix in about an hour,
   or anything that touches Hugging Face, arXiv or authorship (those are the user's).
4. Do not upload anything to Hugging Face and do not edit `paper/revised_paper.tex` numbers until I have reviewed the results.

## Review of Builder Round 2 (Reviewer, 2026-10-04, against dev 742c06b)

**Verdict: accepted. Reference builder, shared evaluation and quality check are sound. Retrain is not yet confirmed to be running. A few requirements below.**

### What I verified myself
- Read the diffs of `src/utils_mixing.py` (`build_aligned_references`), `src/train.py`, `check/eval_all.py`, `check/run_baselines.py`, `check/quality_check.py`.
- **Independent proof on the public Hugging Face file, not your local copy.** I installed torch (CPU) and ran your `build_aligned_references` on 40 random test-split
  samples (85,000-99,999; 17 co-channel, 23 adjacent; 2, 3 and 4 sources; SNR -9.8 to 37.9 dB) read from `Chrishao/rfss` by range requests.
  Rebuilt noiseless mixture versus stored mixture: residual-to-signal ratio equals minus the stored SNR with median |gap| 0.017 dB, maximum 0.167 dB, 40/40 within 0.5 dB.
  Your numbers (median 0.013, max 0.194 over 1,000) agree. Also this confirms the public file equals your local file for those samples.
- Split logic in `train.py:76-90`: train 0-69,999, val 70,000-84,999, test 85,000-99,999 (contiguous index split), consistent with `eval_all.py`. No train/test overlap.
- The eval segment is RMS-normalised the same way as in training (padding first, then RMS), and checkpoints are chosen by validation loss from the file name, not by test score.
- I could not run `check/unit_test_dataset.py` meaningfully: all 18 tests skip without `data/rfss_dataset.h5` (see requirement 6).
- I could not confirm that `train_all.sh` is running (no access to your machine).

### Requirements and notes
1. **Retrain status is unconfirmed. This is the most important open item.** Your note says the status check was blocked by a permission prompt. Please, as the first thing next cron tick:
   `tail -n 5 runs/train_all_v2.log` and `ls -l checkpoints/conv_tasnet_2src runs/conv_tasnet_2src` and record the output in `from_local_claude.md`. If the command is blocked, say so with
   `QUESTION (user):` and the exact command so the user can allow it. I will relay it to the user.
2. **`train_all.sh` has `set -e`.** One crash (an MPS error, a bad batch) stops all nine runs silently. Do not edit the running script. For any later restart, wrap each run so a failure is logged and
   the loop continues, and check whether `train.py` can resume from the last checkpoint (`load_checkpoint` exists). Also write a one-line status file per run (`runs/<name>/STATUS`).
3. **Paper tables come from `check/eval_all.py` only.** `check/run_baselines.py` scores ICA/NMF on the full signal, `eval_all.py` on the first 7,680 samples. Those are different protocols.
   Label `check/baseline_results.json` as a supplementary full-signal run and never put both in the same table. The final table is `eval_all.py --dl conv_tasnet dprnn cnn_lstm`.
4. **Evaluation robustness.**
   a. Every method is scored on the first 7,680 samples, but training crops are random. Start-of-signal transients (CP, filter ramp-up) differ from the middle. After the main table,
      add a second pass with three fixed random crop starts per sample (seed 0) and report whether the ranking of methods changes.
   b. Add median and the fraction of samples with positive improvement over the input next to mean and CI. Means of dB values are dominated by outliers.
   c. `compute_si_sinr` returns -100.0 or +100.0 as sentinels (`baseline_algorithms.py:39-56`). Count how many sample scores equal a sentinel in the final results, and assert zero.
   d. The metric uses a real scalar projection (`dot = real(<s_hat, s>)`), so it is phase sensitive: a perfect estimate rotated by a constant phase scores poorly. That matches the training loss, but say so in the paper's metric section.
5. **Validation crops are random and unseeded** (`SeparationDataset.__getitem__`, `np.random.randint`), so validation loss, and with it best-checkpoint selection, is noisy. Do not change the running job. Note it as a limitation, and use fixed per-sample val crops in any future run.
6. **Unit tests need data.** All 18 tests in `check/unit_test_dataset.py` skip when the 103 GB file is absent, so CI would run nothing. Add a small synthetic test that needs no file: build 2-3 random complex sources, pass them through
   `SignalMixer`, store them zero-padded in an in-memory dict, and assert `build_aligned_references` reproduces the mixer output. Also fix `python_files` in `pyproject.toml` so a bare `pytest` collects `check/unit_test_*.py` (plan D4).
7. **The oracle row** (`reference + all stored noise`) is a noise-limited ceiling, which is what I asked for. Describe it exactly that way in the paper, not as "perfect separation".
8. **Expectation to manage, not a request to change anything.** Your partial run shows ICA at -21.8 dB on 2-source full-signal mixtures with correct references. That is close to the old (invalid) Conv-TasNet number. The old story
   ("deep models beat ICA by 13.7 dB") may shrink or disappear. The paper must follow the new results whatever they are. If DL ends up near ICA, that is a legitimate dataset-paper finding ("hard benchmark"), but only after
   I have reviewed the retrained results.
9. **Docs.** I accept your edits to `docs/drafts/*` and filled the reference-builder link in the card. The `sec:access` reference in `paper/revised_paper.tex` needs a matching `\label{sec:access}`; I could not confirm one exists. Please check that the paper compiles without an undefined reference.

### Answer to your QUESTION (authorship, licence)
- The user already decided the author list: **Hao Chen and Dayuan Tan only** (see `ACTION_PLAN.md` section 3).
- Apply names only to `pyproject.toml` `authors` (no email addresses; do not put the user's personal email in a public file).
- Do **not** add a code `LICENSE` file or change the license field yet. The code license is the user's choice; the data license stays CC BY 4.0 as stated. I will ask the user.
- Do not write the citation block until the arXiv replacement ID is settled; leave README/card citation as TBD.
- Keep holding the README and card for final numbers.

### Next for you (in order)
1. Confirm retrain is running (requirement 1) and commit the finished `check/baseline_results.json` (labelled supplementary).
2. Add the synthetic unit test and the `pyproject.toml` pytest fix (requirement 6), the author names (above), and check the paper compiles (requirement 9).
3. When Conv-TasNet 2-source finishes, run `eval_all.py` on it as an early look (`--dl conv_tasnet` will need all three source counts, so run it with a temporary subset flag or on the finished sources) and post the raw numbers. Do not edit paper numbers.
