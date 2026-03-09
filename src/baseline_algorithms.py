"""Baseline source separation algorithms for the RFSS dataset.

Implements ICA (time-delay embedding) and NMF (spectrogram masking) as
classical baselines for RF source separation. Both operate on a single
mixed observation (SISO).

References:
- ICA via Hankel embedding: Hyvarinen et al., "Independent Component Analysis" (2001)
- NMF source separation: Lee & Seung, "Learning the parts of objects by NMF" (1999)
- SI-SINR: Le Roux et al., "SDR - Half-baked or Well Done?" ICASSP 2019
- Permutation-invariant evaluation: Yu et al., "Permutation invariant training" ICASSP 2017
"""

import warnings

import numpy as np
import scipy.signal as sp_signal
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF as SklearnNMF
from typing import List, Tuple


def compute_si_sinr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Interference-plus-Noise Ratio in dB.

    Projects the estimate onto the reference (removing scale ambiguity), then
    computes the power ratio of the projected signal vs the residual error.

    Args:
        estimate: Separated signal estimate (complex or real).
        reference: Ground-truth source signal (complex or real).

    Returns:
        SI-SINR in dB. Returns -100.0 if reference has zero power,
        +100.0 if residual has zero power.
    """
    length = min(len(estimate), len(reference))
    s_hat = estimate[:length].astype(np.complex128)
    s = reference[:length].astype(np.complex128)

    # Zero-mean centering per Le Roux et al. 2019 (standard SI-SINR definition)
    s_hat = s_hat - s_hat.mean()
    s = s - s.mean()

    ref_power = float(np.real(np.dot(s.conj(), s)))
    if ref_power < 1e-12:
        return -100.0

    dot = float(np.real(np.dot(s_hat.conj(), s)))
    s_target = (dot / ref_power) * s
    e_noise = s_hat - s_target

    target_power = float(np.real(np.dot(s_target.conj(), s_target)))
    noise_power = float(np.real(np.dot(e_noise.conj(), e_noise)))
    if noise_power < 1e-12:
        return 100.0

    return 10.0 * np.log10(target_power / noise_power)


def permutation_invariant_si_sinr(
    estimates: List[np.ndarray],
    references: List[np.ndarray],
) -> Tuple[float, List[int]]:
    """Find the best permutation of estimates matching references via Hungarian algorithm.

    Args:
        estimates: List of separated signal arrays.
        references: List of ground-truth source arrays.

    Returns:
        Tuple of (mean SI-SINR in dB, optimal permutation mapping estimate i → reference perm[i]).
    """
    n = min(len(estimates), len(references))
    sinr_matrix = np.full((n, n), -100.0)
    for i in range(n):
        for j in range(n):
            sinr_matrix[i, j] = compute_si_sinr(estimates[i], references[j])

    row_ind, col_ind = linear_sum_assignment(-sinr_matrix)
    mean_sinr = float(sinr_matrix[row_ind, col_ind].mean())
    perm = [int(col_ind[i]) for i in range(len(row_ind))]
    return mean_sinr, perm


def resample_to_length(source: np.ndarray, target_len: int) -> np.ndarray:
    """Resample source to target_len via Fourier resampling.

    Handles complex signals by resampling real and imaginary parts separately.

    Args:
        source: Input signal (complex or real).
        target_len: Desired output length.

    Returns:
        Resampled signal of length target_len.
    """
    if len(source) == target_len:
        return source.astype(np.complex128)
    if np.iscomplexobj(source):
        r = sp_signal.resample(source.real, target_len)
        i = sp_signal.resample(source.imag, target_len)
        return r + 1j * i
    return sp_signal.resample(source.real, target_len).astype(np.complex128)


class ICASourceSeparation:
    """Single-channel ICA via time-delay (Hankel) embedding.

    Constructs a sliding-window feature matrix from the mixed signal
    (stacking real and imaginary parts as independent features), applies
    FastICA to identify statistically independent temporal components, then
    reconstructs each component via overlap-add of its contribution to every
    window.

    This approach is underdetermined when the number of sources exceeds
    the number of statistically distinguishable temporal patterns. Performance
    degrades for approximately Gaussian sources (e.g., OFDM signals).

    Args:
        n_components: Number of sources to separate.
        max_iter: Maximum FastICA iterations.
        tol: FastICA convergence tolerance.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 500,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _window_params(self, signal_len: int) -> Tuple[int, int]:
        """Choose window/hop so that n_windows >= 10 * n_components."""
        # We need (signal_len - window) // hop + 1 >> n_components
        # Use window = signal_len // (n_components * 12) as rule of thumb
        window = max(16, min(256, signal_len // (self.n_components * 12)))
        hop = max(1, window // 2)
        return window, hop

    def _build_hankel(
        self, signal: np.ndarray, window: int, hop: int
    ) -> np.ndarray:
        """Build (n_windows, 2*window) feature matrix from complex signal."""
        n = len(signal)
        n_windows = max(1, (n - window) // hop + 1)
        X = np.zeros((n_windows, 2 * window), dtype=np.float64)
        for j in range(n_windows):
            start = j * hop
            end = min(start + window, n)
            length = end - start
            X[j, :length] = signal.real[start:end]
            X[j, window : window + length] = signal.imag[start:end]
        return X

    def separate(self, mixed_signal: np.ndarray) -> List[np.ndarray]:
        """Separate mixed_signal into n_components source estimates.

        Args:
            mixed_signal: 1-D complex mixed signal.

        Returns:
            List of n_components complex arrays, each of length len(mixed_signal).
            Returns copies of the mixture if ICA cannot converge.
        """
        n = len(mixed_signal)
        window, hop = self._window_params(n)
        X = self._build_hankel(mixed_signal, window, hop)
        n_windows = X.shape[0]

        # Require at least 2× more windows than components
        n_comp = min(self.n_components, n_windows // 2)
        if n_comp < self.n_components:
            return [mixed_signal.copy().astype(np.complex128) for _ in range(self.n_components)]

        ica = FastICA(
            n_components=n_comp,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            whiten="unit-variance",
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                S = ica.fit_transform(X)  # (n_windows, n_comp)
                A = ica.mixing_           # (2*window, n_comp)
            except Exception:
                return [mixed_signal.copy().astype(np.complex128) for _ in range(self.n_components)]

        results = []
        for k in range(n_comp):
            X_k = np.outer(S[:, k], A[:, k])  # (n_windows, 2*window)
            real_out = np.zeros(n, dtype=np.float64)
            imag_out = np.zeros(n, dtype=np.float64)
            count = np.zeros(n, dtype=np.float64)

            for j in range(n_windows):
                start = j * hop
                end = min(start + window, n)
                length = end - start
                real_out[start:end] += X_k[j, :length]
                imag_out[start:end] += X_k[j, window : window + length]
                count[start:end] += 1.0

            count = np.maximum(count, 1.0)
            results.append((real_out / count) + 1j * (imag_out / count))

        while len(results) < self.n_components:
            results.append(np.zeros(n, dtype=np.complex128))

        return results[: self.n_components]


class NMFSourceSeparation:
    """Spectrogram-based NMF source separation with ratio masking.

    Decomposes the magnitude spectrogram of the mixed signal into K
    non-negative components using NMF, then reconstructs each source by
    applying a soft ratio mask (Wiener-like) to the complex STFT.

    The STFT is computed separately for the real and imaginary parts of
    the complex mixed signal, and the ratio mask is applied to both to
    reconstruct complex source estimates.

    Args:
        n_components: Number of sources to separate.
        max_iter: Maximum NMF iterations.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state

    def _stft_params(self, signal_len: int) -> Tuple[int, int]:
        """Choose STFT nperseg/noverlap for the signal length.

        Ensures nperseg < signal_len and noverlap < nperseg.
        """
        # At least 8 time frames; nperseg must be strictly less than signal_len
        max_nperseg = max(4, signal_len - 1)
        nperseg = min(256, max(4, signal_len // 8), max_nperseg)
        noverlap = nperseg // 2
        return nperseg, noverlap

    def separate(self, mixed_signal: np.ndarray) -> List[np.ndarray]:
        """Separate mixed_signal into n_components source estimates.

        Args:
            mixed_signal: 1-D complex mixed signal.

        Returns:
            List of n_components complex arrays, each of length len(mixed_signal).
            Returns copies of the mixture if NMF cannot decompose.
        """
        n = len(mixed_signal)
        nperseg, noverlap = self._stft_params(n)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _, _, Zr = sp_signal.stft(mixed_signal.real, nperseg=nperseg, noverlap=noverlap)
            _, _, Zi = sp_signal.stft(mixed_signal.imag, nperseg=nperseg, noverlap=noverlap)

        n_freqs, n_times = Zr.shape
        if n_times < self.n_components:
            return [mixed_signal.copy().astype(np.complex128) for _ in range(self.n_components)]

        # Combined magnitude spectrogram (real + imag power summed)
        V = np.sqrt(np.abs(Zr) ** 2 + np.abs(Zi) ** 2) + 1e-10  # (n_freqs, n_times)

        nmf = SklearnNMF(
            n_components=self.n_components,
            init="nndsvda",
            random_state=self.random_state,
            max_iter=self.max_iter,
            alpha_W=0.0,
            alpha_H=0.0,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                W = nmf.fit_transform(V)   # (n_freqs, K)
                H = nmf.components_        # (K, n_times)
            except Exception:
                return [mixed_signal.copy().astype(np.complex128) for _ in range(self.n_components)]

        V_approx = W @ H + 1e-10  # (n_freqs, n_times)

        results = []
        for k in range(self.n_components):
            mask = np.outer(W[:, k], H[k, :]) / V_approx  # ratio mask in [0, 1]

            masked_Zr = mask * Zr
            masked_Zi = mask * Zi

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                _, sig_r = sp_signal.istft(masked_Zr, nperseg=nperseg, noverlap=noverlap)
                _, sig_i = sp_signal.istft(masked_Zi, nperseg=nperseg, noverlap=noverlap)

            sig_k = (sig_r + 1j * sig_i)[:n]
            if len(sig_k) < n:
                sig_k = np.pad(sig_k, (0, n - len(sig_k)))
            results.append(sig_k.astype(np.complex128))

        return results
