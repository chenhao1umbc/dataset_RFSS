> **OWNERSHIP:** This file is maintained by the cloud Claude session ("Reviewer").
> Local Claude ("Builder"): do not edit it. Report progress in `from_local_claude.md`.
> The older `plan.md` / `to_paper_*.md` files are the previous paper-phase workflow and stay as history.

# RFSS Action Plan (written 2026-10-03)

Goal: ship one correct, reproducible RFSS release: dataset on Hugging Face, code on GitHub,
and a corrected paper that replaces the unsupported 2025 arXiv paper.

## 0. Situation in one paragraph

| Item | State |
|---|---|
| arXiv 2508.12106 (Aug 2025, 1 citation) | Results unsupported (26.7 dB, 52,847 samples). Repo's own results say ICA/NMF fail. |
| arXiv 2604.00398 (Apr 2026) | Honest benchmark, but several dataset descriptions are wrong (see section 2, item P). |
| `paper/revised_paper.tex` on `dev` (Jul 2026) | Fixes most of those errors. Not on arXiv. |
| HF `Chrishao/rfss` | Public. Two HDF5 files, no dataset card, no license, not linked from any paper. |
| GitHub `main` | Old 2025 code with placeholders. The real code is only on `dev`. |

## 1. Roles and sync protocol

| Who | Does | Does not |
|---|---|---|
| **User** | Decisions in section 3, co-author sign-off, arXiv and HF account actions | |
| **Local Claude (Builder)** | Code, data checks that need the 103 GB file, training/eval runs, paper edits | Edit this file |
| **Cloud Claude (Reviewer)** | Review of every claim against code and data, dataset card, README/docs, repo cleanup, drafts of arXiv text | Run training; push to `dev` |

Git protocol (simple, no merge conflicts):
1. Builder works on `dev`. Reviewer works on `claude/happy-keller-bnbd79` and never pushes to `dev`.
2. Reviewer's messages and drafts live in files on the reviewer branch: this file and `to_local_claude.md`.
   Builder pulls them with `git fetch origin claude/happy-keller-bnbd79 && git merge origin/claude/happy-keller-bnbd79`
   (only those files change, so merges are clean).
3. Builder's messages live on `dev` in `from_local_claude.md`. Reviewer reads them with `git fetch origin dev`.
4. Every task below gets a status line in `from_local_claude.md` with a commit hash. The Reviewer verifies against
   that commit, not against prose.
5. Reviewer prepares docs/dataset-card/README changes as commits on the reviewer branch; the user (or Builder)
   merges them into `dev`.
6. No numbers go into the paper unless they come from a results file committed to the repo with the command that made it.

## 2. Workstreams

Legend: **B** = Builder, **R** = Reviewer, **U** = User. Gate = must finish before the next step.

### P. Paper problems already known (April arXiv vs July `dev` paper)
April arXiv says: common 30.72 MHz rate, about 4 ms signals, uniform channel/Doppler (1-300 Hz), SNR 0-30 dB, balanced mixing modes,
GSM PAPR 1-2 dB. Data says: native rates resampled to the maximum source rate in each sample (up to 122.88 MHz), 1 ms, weighted channel
choice, Doppler up to 700 Hz, SNR -10 to 40 dB, 40/60 co/adjacent, GSM PAPR about 5 dB (resampling artifact). `dev` paper already fixes these.

### A. Settle the dataset's definition (GATE for everything below)
| # | Task | Owner |
|---|---|---|
| A1 | Confirm what `source_signals` holds: clean transmitted waveform (pre-channel, pre-impairment), at native rate, zero-padded. Evidence so far: `generate_dataset.py` stores the clean signal. State it in one paragraph. | B, R verifies |
| A2 | Adjacent-channel references are pre-frequency-shift (April paper admits it). `mixing_params.frequency_offsets_hz` is already in the metadata. Test whether shifting the reference by that offset (at the mixture rate) makes the adjacent-channel score comparable to co-channel. If yes, this is an evaluation-side fix and the HDF5 need not be regenerated. | B |
| A3 | `mimo_config` says 2x2/4x4 for about 50% of samples but mixtures are single-stream and the paper says SISO only. Find out whether MIMO is applied anywhere. If not, treat the field as vestigial: document it on the dataset card now; fix in a later data version. | B, R verifies |
| A4 | Write down the decision as `docs/dataset_definition.md` (reference definition, adjacent-channel convention, MIMO status, padding and `signal_lengths` meaning). | R drafts, U approves |

### B. Benchmark re-validation (after A)
| # | Task | Owner |
|---|---|---|
| B1 | Confirm whether `baseline_results.json` was regenerated after the SI-SINR zero-mean fix (reviewer_note.md B2). Record the commit hash and date of the run that produced the numbers in the paper. | B |
| B2 | If A2 changes the adjacent-channel target: the DL models were trained on the old target for 60% of the data, so retraining is likely needed (approved, Mac mini). Estimate hours per model and post it in `from_local_claude.md` before starting. | B |
| B3 | Evaluate baselines and DL models on the same test samples (paired), on far more than N=150/300 (full 15,000 test split if feasible), report mean with confidence interval. | B |
| B4 | Report SI-SINR improvement over the mixture input as well as absolute output, so the headline number is not only "less negative than ICA". | B |
| B5 | Keep the 3-source Conv-TasNet checkpoint issue out of the final table: retrain that config with the same recipe as the others. | B |
| B6 | Every published number maps to a committed JSON plus the exact command. | B, R verifies |

### C. Hugging Face release (parallel with B once A is decided)
| # | Task | Owner |
|---|---|---|
| C1 | Dataset card `README.md` for `Chrishao/rfss`: license (CC BY 4.0 per repo), file layout, array shapes, `signal_lengths` meaning, reference definition, split indices (0-69,999 / 70,000-84,999 / 85,000-99,999), metadata schema, known limitations (A1-A3), loading snippet, citation. | R drafts |
| C2 | Add a small preview file (about 1,000 samples, under 1 GB) so people can try the data without 103 GB. | B |
| C3 | Upload card and preview with a token the user holds. Rotate nothing into the repo. | U or B |
| C4 | Remove hardcoded local paths and token path from `upload_hf.py`; use `HF_TOKEN` env var. | B |

### D. Repo release on GitHub
| # | Task | Owner |
|---|---|---|
| D1 | Make `dev` the new `main` content (merge dev into main). Move `old_agent/` out of the tree or delete it (it is in git history). | B, U approves |
| D2 | Rewrite `README.md`: what RFSS is, HF link, install, quick start with the HDF5 loader, reproduce-the-benchmark commands, citation. | R drafts |
| D3 | Fix `pyproject.toml`: author, URL, license classifier mismatch, Python version, `pytest-cov` requirement. Add `LICENSE` and `CITATION.cff`. | R drafts, B applies |
| D4 | Run `check/unit_test_*.py` in a clean environment and make them pass. Add a CI job. | B |
| D5 | Remove working-log noise from the release branch (`work_log.md`, `working_log.md`, `pm_init_*`, `pains.html`) or move to a `notes/` folder. | B, U decides |

### E. Paper v2 (after B and A are final)
| # | Task | Owner |
|---|---|---|
| E1 | Base on `paper/revised_paper.tex`. Update every number from B6 results. Describe the dataset exactly as in `docs/dataset_definition.md`. | B edits, R reviews |
| E2 | Add HF and GitHub links and a data-availability statement. | B |
| E3 | Add a short, plain correction note: v1 results were not reproducible and are replaced. | R drafts |
| E4 | Final audit: claims vs code vs data vs HF card. | R |
| E5 | Venue is IEEE journal (decided). Remove NeurIPS wording from `tasks.md`/`plan.md`. Update author list to Hao Chen and Dayuan Tan. | B |

### F. arXiv actions (user only, in this order)
1. Author agreement. Dayuan Tan must agree to the plan (replace 2508.12106 with v2, then withdraw 2604.00398). Rui Jin is on both existing
   arXiv entries; removing an author should be done with that person's knowledge and consent, and arXiv may ask about an author change
   in a replacement. Suggested route: tell Rui Jin, get a written OK, and move the contribution to the acknowledgments if applicable.
2. Ask arXiv help whether a substantially rewritten v2 is acceptable, and mention the duplicate 2604.00398.
3. Submit v2 to 2508.12106. Comments field: "Substantially revised; v1 results were not reproducible and have been corrected. Supersedes arXiv:2604.00398."
4. After v2 is live: withdraw 2604.00398 with the comment "Superseded by arXiv:2508.12106v2."
5. Notify the author of the citing paper (the one citation) that the numbers changed.

## 3. Decisions (made by the user, 2026-10-03)
1. **Venue: IEEE journal.** `paper/revised_paper.tex` already uses IEEEtran. Drop the NeurIPS D&B references in `tasks.md` and `plan.md`.
2. **Retraining: approved on the local Mac mini.** Builder still reports time estimates before each long run (B2).
3. **Release scope: corrected release (v1.1), not v1.0 as is.** Data fixes from workstream A land before the paper goes to arXiv.
   Note: v1.0 is already public on Hugging Face (32 downloads at last check). Plan: tag the current files as `v1.0`,
   put corrected files on `main`, and say so on the card. Do not silently overwrite v1.0.
4. **Authors: Hao Chen and Dayuan Tan only; Rui Jin removed.** See F1 for what this requires.

## 4. Order of work
1. Now, in parallel: A1-A3 (Builder), C1 and D2/D3 drafts (Reviewer), B1 (Builder).
2. A4 sign-off, then B2-B6 and C2-C4.
3. D1, D4, D5 once numbers are stable.
4. E1-E4, then F.

## 5. Definition of done
- HF card, README, paper, and code agree on every dataset fact.
- Every paper number reproduces from a committed command.
- arXiv has one paper (2508.12106 v2); 2604.00398 is withdrawn with a pointer.
