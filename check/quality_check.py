"""
Dataset quality check and coverage analysis.

Section 1: Signal quality — sample 100 each of 2/3/4-source mixtures from
  rfss_dataset.h5 and 100 each of 4 standards from rfss_single.h5.
  Checks: NaN/Inf, signal length, power, PAPR, source count, standard labels,
  SNR range, power consistency (same-rate co-channel, SNR >= 15 dB).

Section 2: Coverage analysis — scan every 5th record of rfss_dataset.h5
  (20k samples) and verify num_sources / standards / mixing_modes / MIMO
  distributions are within 5% absolute of the intended weights.

Results saved to check/quality_check_results.json.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

import h5py
import torch

MULTI_H5 = Path('data/rfss_dataset.h5')
SINGLE_H5 = Path('data/rfss_single.h5')
RESULTS_PATH = Path('check/quality_check_results.json')
SAMPLES_PER_GROUP = 100
VALID_STANDARDS = {'GSM', 'UMTS', 'LTE', '5G_NR'}
SCAN_STEP = 5        # scan every 5th record → 20k samples for coverage
SINGLE_SCAN_STEP = 10  # scan every 10th record → 100 evenly spaced per standard

# Intended distribution weights (from utils_dataset.py)
INTENDED = {
    'num_sources': {2: 0.50, 3: 0.35, 4: 0.15},
    'mixing_mode': {'co-channel': 0.40, 'adjacent-channel': 0.60},
    'mimo': {'1x1': 0.50, '2x2': 0.30, '4x4': 0.20},
}
TOLERANCE = 0.05  # 5% absolute deviation allowed


# ---------------------------------------------------------------------------
# Torch-based signal metrics
# ---------------------------------------------------------------------------

def _to_tensor(arr) -> torch.Tensor:
    """Convert HDF5-returned numpy array to torch complex64 tensor."""
    return torch.from_numpy(arr)


def power_dbfs(t: torch.Tensor) -> float:
    """RMS power in dBFS."""
    rms = t.abs().pow(2).mean().sqrt().item()
    return 20 * math.log10(rms) if rms > 0 else -999.0


def papr_db(t: torch.Tensor) -> float:
    """Peak-to-average power ratio in dB."""
    peak = t.abs().max().item()
    rms = t.abs().pow(2).mean().sqrt().item()
    return 20 * math.log10(peak / rms) if rms > 0 else 0.0


def count_nonzero_sources(src_block, sig_len: int) -> int:
    """Count non-zero source slots in the (4, max_len) HDF5 block."""
    count = 0
    for i in range(4):
        chunk = _to_tensor(src_block[i, :sig_len])
        if chunk.any():
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-sample quality check (multi-source)
# ---------------------------------------------------------------------------

def check_sample(h5, idx: int) -> dict:
    sig_len = int(h5['signal_lengths'][idx])
    mixed = _to_tensor(h5['mixed_signals'][idx, :sig_len])
    src_block = h5['source_signals'][idx, :, :]
    meta = json.loads(h5['metadata'][idx])

    result = {
        'idx': idx,
        'num_sources': meta.get('num_sources', -1),
        'snr_db': meta.get('snr_db', None),
        'mixing_mode': meta.get('mixing_params', {}).get('mixing_mode', 'unknown'),
        'checks': {},
    }
    c = result['checks']

    c['mixed_no_nan_inf'] = bool(mixed.isfinite().all())

    sources_ok = True
    for i in range(4):
        chunk = _to_tensor(src_block[i, :sig_len])
        if chunk.any() and not chunk.isfinite().all():
            sources_ok = False
            break
    c['sources_no_nan_inf'] = sources_ok

    c['signal_length_positive'] = sig_len > 0

    pwr = power_dbfs(mixed)
    num_src = meta.get('num_sources', 2)
    power_upper = 30.0 + 10 * (num_src - 2)
    c['mixed_power_in_range'] = -40.0 <= pwr <= power_upper
    result['mixed_power_dbfs'] = round(pwr, 2)

    pr = papr_db(mixed)
    c['papr_in_range'] = 0.0 <= pr <= 25.0
    result['papr_db'] = round(pr, 2)

    actual = count_nonzero_sources(src_block, sig_len)
    c['source_count_consistent'] = actual == meta.get('num_sources', -1)
    result['actual_nonzero_sources'] = actual

    c['valid_standards'] = all(
        s.get('standard') in VALID_STANDARDS for s in meta.get('sources', [])
    )

    snr = meta.get('snr_db', None)
    c['snr_in_range'] = snr is not None and -10.0 <= snr <= 40.0

    mixing_mode = meta.get('mixing_params', {}).get('mixing_mode', '')
    source_rates = [s.get('signal_params', {}).get('sample_rate', 0)
                    for s in meta.get('sources', [])]
    same_rate = len(set(source_rates)) == 1
    if mixing_mode == 'co-channel' and snr is not None and snr >= 15.0 and same_rate:
        power_ratios = meta.get('mixing_params', {}).get('power_ratios_db', [])
        reconstructed = torch.zeros(sig_len, dtype=torch.complex64)
        for i, pr_db_val in enumerate(power_ratios):
            if i < 4:
                src = _to_tensor(src_block[i, :sig_len])
                if src.any():
                    reconstructed = reconstructed + (10 ** (pr_db_val / 20.0)) * src
        mixed_pwr = mixed.abs().pow(2).mean().item()
        recon_pwr = reconstructed.abs().pow(2).mean().item()
        if recon_pwr > 0 and mixed_pwr > 0:
            ratio_db = 10 * math.log10(mixed_pwr / recon_pwr)
            c['power_consistency'] = abs(ratio_db) <= 5.0
            result['power_ratio_db'] = round(ratio_db, 2)
        else:
            c['power_consistency'] = False
    else:
        c['power_consistency'] = None

    return result


# ---------------------------------------------------------------------------
# Per-sample quality check (single-source)
# ---------------------------------------------------------------------------

def check_single_sample(h5, idx: int) -> dict:
    sig_len = int(h5['signal_lengths'][idx])
    mixed = _to_tensor(h5['mixed_signals'][idx, :sig_len])
    src = _to_tensor(h5['source_signals'][idx, 0, :sig_len])
    meta = json.loads(h5['metadata'][idx])

    result = {
        'idx': idx,
        'standard': meta.get('sources', [{}])[0].get('standard', 'unknown'),
        'snr_db': meta.get('snr_db', None),
        'checks': {},
    }
    c = result['checks']

    c['mixed_no_nan_inf'] = bool(mixed.isfinite().all())
    c['source_no_nan_inf'] = bool(src.isfinite().all())
    c['signal_length_positive'] = sig_len > 0

    pwr = power_dbfs(mixed)
    c['mixed_power_in_range'] = -40.0 <= pwr <= 30.0
    result['mixed_power_dbfs'] = round(pwr, 2)

    pr = papr_db(mixed)
    c['papr_in_range'] = 0.0 <= pr <= 25.0
    result['papr_db'] = round(pr, 2)

    c['num_sources_is_one'] = meta.get('num_sources', -1) == 1
    c['valid_standard'] = result['standard'] in VALID_STANDARDS

    snr = meta.get('snr_db', None)
    c['snr_in_range'] = snr is not None and -10.0 <= snr <= 40.0

    return result


# ---------------------------------------------------------------------------
# Collect indices by group
# ---------------------------------------------------------------------------

def collect_multi_indices(h5, target_counts: dict) -> dict:
    total = int(h5.attrs['actual_samples'])
    collected = {k: [] for k in target_counts}
    for idx in range(0, total, SCAN_STEP):
        if all(len(collected[k]) >= target_counts[k] for k in target_counts):
            break
        try:
            meta = json.loads(h5['metadata'][idx])
            ns = meta.get('num_sources', -1)
            if ns in collected and len(collected[ns]) < target_counts[ns]:
                collected[ns].append(idx)
        except Exception:
            continue
    return collected


def collect_single_indices(h5, target_per_standard: int) -> dict:
    total = int(h5.attrs['actual_samples'])
    collected = {s: [] for s in VALID_STANDARDS}
    for idx in range(0, total, SINGLE_SCAN_STEP):
        if all(len(collected[s]) >= target_per_standard for s in VALID_STANDARDS):
            break
        try:
            meta = json.loads(h5['metadata'][idx])
            std = meta.get('sources', [{}])[0].get('standard', '')
            if std in collected and len(collected[std]) < target_per_standard:
                collected[std].append(idx)
        except Exception:
            continue
    return collected


# ---------------------------------------------------------------------------
# Summarize quality results
# ---------------------------------------------------------------------------

def summarize(group_results: list, group_name: str) -> bool:
    n = len(group_results)
    if n == 0:
        print(f"\n{group_name}: no samples found")
        return False

    check_names = list(group_results[0]['checks'].keys())
    print(f"\n{'='*60}")
    print(f"{group_name}  (n={n})")
    print(f"{'='*60}")

    all_pass = True
    for cn in check_names:
        values = [r['checks'][cn] for r in group_results]
        applicable = [v for v in values if v is not None]
        if not applicable:
            print(f"  {cn:<35} N/A")
            continue
        passed = sum(1 for v in applicable if v)
        pct = 100 * passed / len(applicable)
        status = 'PASS' if pct == 100 else ('WARN' if pct >= 90 else 'FAIL')
        if status != 'PASS':
            all_pass = False
        print(f"  {cn:<35} {passed:3d}/{len(applicable)}  {pct:5.1f}%  [{status}]")

    powers = [r['mixed_power_dbfs'] for r in group_results]
    paprs = [r['papr_db'] for r in group_results]
    snrs = [r['snr_db'] for r in group_results if r['snr_db'] is not None]
    print(f"\n  Mixed power (dBFS): min={min(powers):.1f}  max={max(powers):.1f}"
          f"  mean={sum(powers)/len(powers):.1f}")
    print(f"  PAPR (dB):         min={min(paprs):.1f}  max={max(paprs):.1f}"
          f"  mean={sum(paprs)/len(paprs):.1f}")
    if snrs:
        print(f"  SNR (dB):          min={min(snrs):.1f}  max={max(snrs):.1f}"
              f"  mean={sum(snrs)/len(snrs):.1f}")
    consistency = [r.get('power_ratio_db') for r in group_results
                   if r.get('power_ratio_db') is not None]
    if consistency:
        print(f"  Power consistency: {len(consistency)} checked"
              f"  mean ratio={sum(consistency)/len(consistency):.2f} dB")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return all_pass


# ---------------------------------------------------------------------------
# Coverage analysis helper
# ---------------------------------------------------------------------------

def _check_dist(name: str, counts: Counter, intended: dict) -> tuple:
    """Check one distribution against intended proportions.

    Returns (ok: bool, result: dict) where result maps key → stats dict.
    """
    total_c = sum(counts.values())
    print(f"\n  {name}:")
    ok = True
    result = {}
    for k, intended_p in intended.items():
        actual_p = counts.get(k, 0) / total_c if total_c > 0 else 0
        dev = abs(actual_p - intended_p)
        status = 'OK' if dev <= TOLERANCE else 'WARN'
        if status != 'OK':
            ok = False
        print(f"    {str(k):<20} actual={actual_p:.3f}  intended={intended_p:.3f}"
              f"  dev={dev:.3f}  [{status}]")
        result[str(k)] = {'actual': round(actual_p, 4), 'intended': intended_p,
                          'deviation': round(dev, 4), 'ok': status == 'OK'}
    return ok, result


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def coverage_analysis(h5) -> dict:
    """Scan every SCAN_STEP-th record and check distribution proportions."""
    total = int(h5.attrs['actual_samples'])
    num_sources_cnt: Counter = Counter()
    standards_cnt: Counter = Counter()
    mixing_mode_cnt: Counter = Counter()
    mimo_cnt: Counter = Counter()
    snr_values = []
    scanned = 0

    for idx in range(0, total, SCAN_STEP):
        try:
            meta = json.loads(h5['metadata'][idx])
        except Exception:
            continue
        scanned += 1
        num_sources_cnt[meta.get('num_sources', -1)] += 1
        for src in meta.get('sources', []):
            standards_cnt[src.get('standard', 'unknown')] += 1
        mode = meta.get('mixing_params', {}).get('mixing_mode', 'unknown')
        mixing_mode_cnt[mode] += 1
        mc = meta.get('mimo_config', {})
        mimo_key = f"{mc.get('num_tx', 1)}x{mc.get('num_rx', 1)}"
        mimo_cnt[mimo_key] += 1
        snr = meta.get('snr_db')
        if snr is not None:
            snr_values.append(snr)

    print(f"\n{'='*60}")
    print(f"Coverage analysis  (scanned {scanned:,} / {total:,} samples)")
    print(f"{'='*60}")

    deviations = {}

    all_ok = True
    ok, result = _check_dist('num_sources', num_sources_cnt, INTENDED['num_sources'])
    all_ok &= ok
    deviations['num_sources'] = result
    ok, result = _check_dist('mixing_mode', mixing_mode_cnt, INTENDED['mixing_mode'])
    all_ok &= ok
    deviations['mixing_mode'] = result
    ok, result = _check_dist('mimo_config', mimo_cnt, INTENDED['mimo'])
    all_ok &= ok
    deviations['mimo_config'] = result

    print(f"\n  standards (per-source occurrence):")
    std_total = sum(standards_cnt.values())
    std_result = {}
    for s in sorted(VALID_STANDARDS):
        p = standards_cnt.get(s, 0) / std_total if std_total > 0 else 0
        print(f"    {s:<20} {p:.3f}  (count={standards_cnt.get(s, 0):,})")
        std_result[s] = round(p, 4)
    deviations['standards'] = std_result

    if snr_values:
        print(f"\n  SNR: min={min(snr_values):.1f}  max={max(snr_values):.1f}"
              f"  mean={sum(snr_values)/len(snr_values):.1f} dB")

    print(f"\n  Overall: {'ALL WITHIN TOLERANCE' if all_ok else 'DEVIATIONS DETECTED'}")
    return {'scanned': scanned, 'total': total, 'distributions': deviations,
            'snr': {'min': round(min(snr_values), 2), 'max': round(max(snr_values), 2),
                    'mean': round(sum(snr_values)/len(snr_values), 2)} if snr_values else {}}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MULTI_H5.exists():
        sys.exit(f"HDF5 file not found: {MULTI_H5}")

    report = {}

    # ---- Section 1a: multi-source quality check ----
    print(f"\nQuality check (multi-source): {MULTI_H5}")
    with h5py.File(MULTI_H5, 'r') as h5:
        total = int(h5.attrs['actual_samples'])
        print(f"Total samples: {total:,}")
        target = {2: SAMPLES_PER_GROUP, 3: SAMPLES_PER_GROUP, 4: SAMPLES_PER_GROUP}
        indices = collect_multi_indices(h5, target)
        for ns, idx_list in indices.items():
            print(f"  {ns}-source: {len(idx_list)} indices found")

        multi_results = {}
        all_multi_ok = True
        for ns in [2, 3, 4]:
            results = []
            for idx in indices[ns]:
                try:
                    results.append(check_sample(h5, idx))
                except Exception as e:
                    print(f"  ERROR on sample {idx}: {e}")
            ok = summarize(results, f"{ns}-source mixtures")
            if not ok:
                all_multi_ok = False
            multi_results[f'{ns}_source'] = {
                'n': len(results),
                'all_pass': ok,
                'check_pass_rates': {
                    cn: sum(1 for r in results if r['checks'].get(cn) is True) /
                        max(1, sum(1 for r in results if r['checks'].get(cn) is not None))
                    for cn in (results[0]['checks'] if results else {})
                }
            }

        print(f"\n{'='*60}")
        print(f"Multi-source: {'ALL PASSED' if all_multi_ok else 'FAILURES DETECTED'}")
        print(f"{'='*60}")
        report['multi_source_quality'] = multi_results

        # ---- Section 1b: coverage analysis ----
        coverage = coverage_analysis(h5)
        report['coverage'] = coverage

    # ---- Section 1c: single-source quality check ----
    if SINGLE_H5.exists():
        print(f"\nQuality check (single-source): {SINGLE_H5}")
        with h5py.File(SINGLE_H5, 'r') as h5s:
            stotal = int(h5s.attrs['actual_samples'])
            print(f"Total samples: {stotal:,}")
            indices_s = collect_single_indices(h5s, SAMPLES_PER_GROUP)
            for std, idx_list in indices_s.items():
                print(f"  {std}: {len(idx_list)} indices found")

            single_results = {}
            all_single_ok = True
            for std in sorted(VALID_STANDARDS):
                results = []
                for idx in indices_s.get(std, []):
                    try:
                        results.append(check_single_sample(h5s, idx))
                    except Exception as e:
                        print(f"  ERROR on sample {idx}: {e}")
                ok = summarize(results, f"{std} single-source")
                if not ok:
                    all_single_ok = False
                single_results[std] = {
                    'n': len(results),
                    'all_pass': ok,
                }

            print(f"\n{'='*60}")
            print(f"Single-source: {'ALL PASSED' if all_single_ok else 'FAILURES DETECTED'}")
            print(f"{'='*60}")
            report['single_source_quality'] = single_results
    else:
        print(f"\nSkipping single-source check: {SINGLE_H5} not found")

    # ---- Save results ----
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    overall = all_multi_ok and (all_single_ok if SINGLE_H5.exists() else True)
    print(f"\n{'='*60}")
    print(f"FINAL: {'ALL CHECKS PASSED' if overall else 'FAILURES DETECTED — review above'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
