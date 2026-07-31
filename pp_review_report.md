# Post-Paper Audit Report

**Paper**: RFSS: A Multi-Standard RF Signal Source Separation Dataset with 3GPP-Standardized Channel and Hardware Impairments
**Venue**: IEEE journal (IEEEtran `journal` class — target not explicitly stated, compatible with IEEE Access / IEEE Transactions)
**Date**: 2026-03-26
**Pages**: 10
**References**: 40 cited / 40 in .bib

---

## Severity Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2     |
| MAJOR    | 1     |
| MINOR    | 3     |

**Verdict: FAIL**
Reason: 2 CRITICAL AI-vocabulary items. Both must be replaced before submission.

---

## 1. Reference Audit

### 1a. Pipeline Results
- **Total references**: 40
- **Orphaned .bib entries**: 0
- **Undefined citations**: 0
- **Compilation**: clean — no undefined citation warnings, no overfull \\hbox

### 1b. WebSearch Verification (8 sampled: all suspicious + random 20% of REAL)

All 8 sampled references verified REAL with correct titles, authors, venues, and years. No fabricated references found. Specific findings:

| BibKey | Status | Notes |
|--------|--------|-------|
| lancho2024rfchallenge | REAL | Published IEEE Open J. Commun. Soc. 2025, vol.6 pp.4083–4100. arXiv:2409.08839 confirmed. |
| rapp1991effects | REAL | ESA Special Publication vol.332 pp.179–184, 1991. ADS record confirmed. |
| mennes2020sc2 | REAL | IEEE TNSM vol.17 no.4 pp.2024–2038, DOI:10.1109/TNSM.2020.3031078. |
| jaeckel2014quadriga | REAL | IEEE TAP vol.62 pp.3242–3256, 2014. IEEExplore confirmed. |
| hadani2017otfs | REAL | IEEE WCNC 2017 pp.1–6. IEEExplore confirmed. |
| abdoli2015fofdm | REAL | IEEE SPAWC 2015 pp.66–70. IEEExplore confirmed. |
| wichern2019wham | REAL | Interspeech 2019. ISCA archive confirmed. |
| subakan2021sepformer | REAL | ICASSP 2021. Semantic Scholar confirmed. |

### 1c. BibTeX Note Field Audit

3 note fields present:
- `lancho2024rfchallenge`: `note = {arXiv:2409.08839}` — acceptable (arXiv ID annotation on published paper)
- `3gpp45004`: `note = {Release 15}` — acceptable (informative release tag)
- `rafii2017musdb18`: `note = {10.5281/zenodo.1117372}` — acceptable (Zenodo DOI for dataset citation)

No internal commentary contamination found.

### 1d. arXiv Citation Check

`lancho2024rfchallenge` carries an arXiv note but is correctly cited as an IEEE Open Journal paper with `year = {2025}`. No arXiv-only citations detected.

### 1e. Reference Density

40 references. Appropriate for IEEE journal paper of this type and scope.

---

## 2. Writing Style Audit

### 2a. AI Writing Verdict: **YES — AI DETECTED (CRITICAL)**

Two blacklisted AI-vocabulary words found. Both are CRITICAL. Paper FAILS this gate.

### 2b. AI Vocabulary Markers Found (CRITICAL: each must be replaced)

| Word | Location | Context | Replacement |
|------|----------|---------|-------------|
| `comprehensive` | revised_paper.tex:32 (abstract) | "a comprehensive open-source dataset of 100,000 multi-source RF signal samples" | Remove the adjective entirely — the scale (100,000 samples, 103 GB) speaks for itself. Rewrite: "an open-source dataset of 100,000 multi-source RF signal samples" |
| `facilitated` | revised_paper.tex:102 | "WSJ0-2mix and MUSDB facilitated the rapid development of Conv-TasNet, DPRNN, and related architectures" | Replace with "enabled": "WSJ0-2mix and MUSDB enabled the rapid development of..." |

### 2c. AI-Telltale Phrases: CLEAN

No matches for any banned phrase patterns. No "Moreover,", "Furthermore,", "Additionally," paragraph openers. No em dashes, en dashes, or prose hyphens.

### 2d. Structural Patterns: CLEAN

- No rigid subsection template repetition detected. Each subsection has natural variation in structure, length, and entry angle.
- Paragraph lengths vary naturally. Short declarative sentences ("This data has not previously existed.") coexist with long compound-complex sentences for algorithmic descriptions.
- Sentence length standard deviation appears healthy — no narrow-band uniformity.

### 2e. Style Profile Match (Hao Chen / IEEE venues)

Overall strong match. Key positive signals:

| Dimension | Expected (Author) | Found | Assessment |
|-----------|-------------------|-------|------------|
| Voice | Active dominant | "We present", "We generate", "We adopt", "We evaluate", "We benchmark" | PASS |
| Prior work critique | Direct, no excessive praise | "RadioML contains only single-signal samples: there is no mixture of two or more simultaneous transmissions, and therefore no ground truth for separation." | PASS |
| Claim support | Quantified, table-referenced | All dB claims reference specific table rows; improvement values computed from reported numbers | PASS |
| Hedging | Moderate, specific | "These assumptions hold approximately for cellular signals but become weaker when..." | PASS |
| Transitions | Functional connectors | "However", "In contrast to", "More critically", "Neither approach", "The most closely related" | PASS |
| Acronym order | Full term first, abbrev. in parens | GSM, UMTS, LTE, NR, AMC, PIT, STFT, BiLSTM, TCN, CFO — all correctly introduced | PASS |
| Paper road-map | Explicit in introduction | "Section II reviews... Section III describes... Section V presents..." | PASS |
| Vocabulary choice | Domain-specific, precise | "spectral occupancy", "Hankel embedding", "Wiener-ratio masking", "permutation ambiguity" | PASS |

The two blacklisted words ("comprehensive", "facilitated") are isolated vocabulary substitutions. The underlying sentence construction and argumentation style are author-consistent throughout.

---

## 3. Venue Format Compliance

### 3a. Compilation Health

- Undefined citations: **none**
- Overfull \\hbox: **none**
- LaTeX warnings: nominal
- PDF pages: **10** (appropriate for IEEE journal submission)

### 3b. Figures (5 figures, all verified via PDF rendering)

All 5 figure files exist (`fig_pipeline.pdf`, `fig_dataset_stats.pdf`, `fig_spectrograms.pdf`, `fig_signal_quality.pdf`, `fig_benchmark.pdf`). Visual inspection via Ghostscript:

| Figure | Caption | Issues |
|--------|---------|--------|
| Fig. 1 (pipeline) | Self-contained, describes pipeline stages | OK — renders as full-width diagram |
| Fig. 2 (dataset stats) | Describes three sub-panels (source count, mixing mode, standard combinations) | OK |
| Fig. 3 (spectrograms) | Describes axes, color encoding, interpretive role | OK |
| Fig. 4 (signal quality) | Describes three sub-panels (PAPR, PSD, amplitude) | OK |
| Fig. 5 (benchmark) | Describes overall and co-channel panels | OK |

Figures use `\textwidth` placement with `figure*` environment (full-width in two-column layout) — appropriate.

### 3c. Tables

| Table | Formatting | Issues |
|-------|-----------|--------|
| Table I (overall PI-SI-SINR) | `booktabs`, `\resizebox{\columnwidth}` | OK |
| Table II (co-channel PI-SI-SINR) | `booktabs`, `\resizebox{\columnwidth}` | OK |
| Table III (SNR stratification) | `booktabs`, `\resizebox{\columnwidth}` | OK |

No `\hline` usage. No `\scriptsize` or `\tiny`. All tables have captions and `\label`. Numeric precision consistent (2 decimal places in dB). Bold used correctly for best-per-row values.

### 3d. Abstract

- **Word count**: ~249 words (LaTeX-stripped). Right at the IEEE 250-word boundary — **borderline MAJOR**.
- Acronyms: all defined inline (TDL, PI-SI-SINR, NMF, ICA used in abstract; ICA and NMF are algorithm names cited with references, acceptable in IEEE style)
- All abstract claims substantiated in body: Conv-TasNet 2-source −21.18 dB (Table I), co-channel −12.34 dB (Table II), ICA −28.04 dB (Table II), NMF −16.19 dB (Table II) — all verified against experiment_results.md
- No citation numbers in abstract

### 3e. Structure Check

- Introduction: problem → importance → gaps → contributions (6 items) → roadmap — **complete and correct**
- Each contribution maps to a specific body section
- Conclusion introduces no new claims; restates key results with forward-looking extensions only
- No orphan claims found

### 3f. Numbers Verified Against experiment_results.md

| Claim in paper | experiment_results.md value | Match |
|----------------|-----------------------------|-------|
| Conv-TasNet 2-src: −21.18 dB | −21.18 dB | YES |
| Conv-TasNet 3-src: −21.08 dB | −21.08 dB | YES |
| Conv-TasNet 4-src: −22.13 dB | −22.13 dB | YES |
| ICA 2-src: −34.91 dB | −34.91 dB | YES |
| Co-channel Conv-TasNet 2-src: −12.34 dB | −12.34 dB | YES |
| Co-channel ICA 2-src: −28.04 dB | −28.04 dB | YES |
| Conv-TasNet improvement over ICA: 13.7 dB (2-src) | 13.73 dB (rounds to 13.7) | YES |

All reported numbers are internally consistent and match ground-truth results files.

---

## 4. Additional Issues Found

### British Spelling (MINOR × 2)

| Word | Line | Fix |
|------|------|-----|
| `generalise` | 232 | → `generalize` |
| `artefacts` | 245 | → `artifacts` |

IEEE uses American English. These are the only two British spellings found.

### "robust" / "robustness" (MINOR — borderline)

Line 731: "The relative robustness here can be attributed to the spectral diversity of the four cellular standards"

Per the blacklist, "robust" is flagged without a robustness evaluation. However, the immediately preceding sentence provides quantitative backing: "The modest degradation from 2-source to 4-source configurations ($\leq 1.1$ dB for Conv-TasNet)". This is sufficient evidence. **Not blocking**, but consider replacing with "The modest degradation ($\leq 1.1$ dB) can be attributed to..." to make the quantitative anchor more direct.

### Contribution Ordering (NON-ISSUE — verified)

The OCR output appeared to show items 5 and 6 out of order. Confirmed via Ghostscript page rendering: items are in correct sequence (1–6) in the actual PDF. OCR column-flow confusion; no action needed.

---

## 5. Recommended Actions (Prioritized)

### CRITICAL (must fix before submission)

1. **Replace "comprehensive" in abstract (line 32)**: Remove the word entirely. "a comprehensive open-source dataset of 100,000 multi-source RF signal samples" → "an open-source dataset of 100,000 multi-source RF signal samples generated with full 3GPP standards compliance"

2. **Replace "facilitated" (line 102)**: "WSJ0-2mix and MUSDB facilitated the rapid development of Conv-TasNet, DPRNN, and related architectures" → "WSJ0-2mix and MUSDB enabled the rapid development of Conv-TasNet, DPRNN, and related architectures"

### MAJOR (strongly recommended)

3. **Abstract word count (~249 words)**: Trim 3–5 words to stay comfortably under 250. Candidate: remove "generated with full 3GPP standards compliance" from the abstract sentence (it is immediately elaborated in the next sentence about what standards are covered).

### MINOR (fix before submission)

4. **Fix British spelling "generalise" (line 232)** → "generalize"
5. **Fix British spelling "artefacts" (line 245)** → "artifacts"
6. **Strengthen "robustness" anchor (line 731)**: Rewrite to lead with the number: "The $\leq 1.1$ dB degradation from 2-source to 4-source (Conv-TasNet) is encouraging..." removing the word "robustness" entirely.
