# Reviewer Notes: revised_paper.tex / revised_paper.pdf

Reviewer: Claude
Last verified: 2026-03-08 (third pass)

---

## Status: ALL TECHNICAL ISSUES RESOLVED

Every issue raised across two review rounds has been addressed. The paper is technically clean.

---

## Complete Issue Tracker

### Round 1 Critical — All Fixed

| # | Issue | Resolution |
|---|-------|-----------|
| 1 | Epochs stated as 100, actual 30 | Fixed: "cosine annealing over 30 epochs" (line 532) |
| 2 | Crop duration: "250 ms" should be 250 µs | Fixed: "7,680 samples (250~$\mu$s)" (line 533) |
| 3 | Optimizer stated as AdamW, actual Adam | Fixed: "Optimization uses Adam" (line 531) |
| 4 | ICA cited Cabric 2004 (spectrum sensing) | Fixed: `\cite{hyvarinen2000ica}`; new bib entry added |
| 5 | "DeepSig's open releases" misattributed Rajendran 2018 | Fixed: neutral phrasing, no DeepSig mention (lines 155-159) |
| 6 | Adjacent-channel floor "-28 dB for all methods" wrong for ICA | Fixed: "deep learning methods"; ICA -38 to -42 dB stated explicitly (lines 626-629) |

### Round 1 Significant — All Fixed

| # | Issue | Resolution |
|---|-------|-----------|
| 7 | GSM GMSK equation had amplitude modulation term | Fixed: correct constant-envelope form with cumulative phase integral (lines 228-238) |
| 8 | NMF improvement in conclusion "1--4 dB" understated | Fixed: "2--4~dB" (line 731) |
| 9 | DPRNN 4-source early stopping (epoch 4 of 30) undisclosed | Fixed: footnote added (lines 572-576) |
| 10 | Abstract "1 to 4 sources" vs actual 2-4 in main file | Fixed: "2 to 4 simultaneous sources per sample (plus 4,000 single-source reference samples in a companion file)" (lines 36-37) |
| 13 | SI-SINR vs SI-SNR naming not explained | Fixed: clarification paragraph added (lines 170-174) |
| 19 | SISO-only benchmark not explicitly stated | Fixed: "All experiments use single-channel (SISO) signals; spatial diversity is not exploited." (lines 536-537) |

### Round 1 Minor — All Fixed

| # | Issue | Resolution |
|---|-------|-----------|
| 12 | Missing pages for ICASSP 2019 (leroux2019sdr) | Fixed: pp. 626--630 added to bib |
| 12 | Missing pages for ICASSP 2020 (luo2020dual) | Fixed: pp. 8501--8505 added to bib |
| 17 | No NMF citation in Sec II-C | Fixed: `\cite{fevotte2011nmf}`; new bib entry added |

### Round 2 Remaining — All Fixed

| # | Issue | Resolution |
|---|-------|-----------|
| R1 | Pipeline said "1--4 source standards" (Sec III intro) | Fixed: "2--4 source standards" (line 203) |
| R2 | Mixing scenarios said "$N_{\rm src}$ drawn from {1,2,3,4}" | Fixed: "{2, 3, 4}" (line 350) |
| R3 | Conv-TasNet 3-src from ReduceLROnPlateau run, undisclosed | Fixed: footnote added alongside DPRNN footnote (lines 568-572) |

---

## Reference Audit — Final

All references verified as real, correctly attributed, and properly formatted.

| Key | Verdict |
|-----|---------|
| `hyvarinen2000ica` | Correct — Hyvärinen & Oja 2000, *Neural Networks* 13(4-5):411-430 |
| `fevotte2011nmf` | Correct — Févotte & Idier 2011, *Neural Computation* 23(9):2421-2456 |
| `leroux2019sdr` | Correct — ICASSP 2019, pp. 626-630 |
| `luo2019conv` | Correct — IEEE/ACM TASLP 27(8):1256-1266, 2019 |
| `luo2020dual` | Correct — ICASSP 2020, pp. 8501-8505 |
| `rapp1991effects` | Correct — ESA-SP 332:179-184, 1991 (standard Rapp model citation) |
| `jakes1994microwave` | Correct — Wiley-IEEE Press, 1994 |
| All 3GPP tech reports | Correct format and content |
| `cabric2004implementation` | Correct use — cited in Intro for coexistence motivation (appropriate) |
| `rajendran2018deep` | Correct — KU Leuven/imec paper; misattribution to DeepSig removed |
| `oshea2016radio` | Acceptable — GNU Radio Conference 2016; no formal page numbers available |

---

## Data Consistency Check — Final

| Claim | Paper | Ground Truth | Status |
|-------|-------|-------------|--------|
| All numerical results (Tables I and II) | — | experiment_results.md | VERIFIED OK |
| Epochs | 30 | 30 (train.py default) | OK |
| Optimizer | Adam | Adam (train.py line 380) | OK |
| Crop duration | 250 µs | 7680 / 30.72e6 = 250 µs | OK |
| NMF improvement (conclusion) | 2--4 dB | 2.2--4.4 dB actual | OK |
| Adjacent-channel floor | DL ~−28 dB; ICA −38 to −42 dB | Matches experiment_results.md | OK |
| Conv-TasNet 3-src scheduler | ReduceLROnPlateau (footnoted) | ReduceLROnPlateau | OK |
| Source count in dataset | {2,3,4} | 2/3/4 source only in rfss_dataset.h5 | OK |

---

## Pre-Submission Checklist (Non-Technical)

- [ ] **Author affiliation incomplete** — "Department of Electrical Engineering" with no institution, city, country, or email. Must be completed before submission.
- [ ] **HuggingFace URL is a placeholder** — `https://huggingface.co/datasets/rfss/rfss-dataset` must be live before submitting. Upload script is ready at `src/upload_huggingface.py`.
- [ ] **Figures must be generated** — All `\includegraphics` targets (fig_pipeline.pdf, fig_dataset_stats.pdf, fig_spectrograms.pdf, fig_signal_quality.pdf, fig_benchmark.pdf) must exist in `paper/figures/` before compiling final PDF.
- [ ] **Compile clean** — Run `pdflatex` + `bibtex` + `pdflatex` × 2 and verify no undefined references or Overfull \hbox warnings.
