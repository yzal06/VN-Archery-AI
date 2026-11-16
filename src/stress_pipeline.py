"""Stress estimation pipeline (PPG + HRV features + MLP).

Implements Sections 2.3.2, 2.4.2, and 3.2 of arXiv:2511.14057v1.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def bandpass_filter_ppg(
    ppg: np.ndarray,
    fs: float = 20.0,
    lowcut: float = 0.6,
    highcut: float = 10.0,
    order: int = 3,
) -> np.ndarray:
    """Apply 3rd-order Butterworth bandpass filter (0.6-10 Hz) to raw PPG."""
    nyq = 0.5 * fs
    # Clamp highcut slightly below Nyquist when fs=20 Hz (Nyquist=10 Hz)
    high = min(highcut / nyq, 0.99)
    low = lowcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, ppg)


def extract_rr_intervals(ppg_filtered: np.ndarray, fs: float = 20.0) -> np.ndarray:
    """Detect pulse peaks and compute cleaned RR intervals in seconds."""
    min_distance = max(1, int(0.4 * fs))  # Max ~150 bpm
    peaks, _ = find_peaks(ppg_filtered, distance=min_distance)
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)

    rr_sec = np.diff(peaks) / fs
    # Simple physiological bounds (0.35s - 1.5s) + median replacement for outliers
    valid = (rr_sec >= 0.35) & (rr_sec <= 1.5)
    if not np.any(valid):
        return rr_sec
    median_rr = np.median(rr_sec[valid])
    return np.where(valid, rr_sec, median_rr)


def sample_entropy(signal: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Compute Sample Entropy (Eq. 10) for a 1D signal."""
    n = len(signal)
    if n <= m + 1:
        return 0.0
    if r is None:
        std = np.std(signal)
        if std == 0:
            return 0.0
        r = 0.2 * std

    def _phi(dim: int) -> int:
        templates = np.array([signal[i : i + dim] for i in range(n - dim)])
        count = 0
        for i in range(len(templates)):
            dist = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b_count = _phi(m)
    a_count = _phi(m + 1)
    if b_count == 0 or a_count == 0:
        return 0.0
    return float(-np.log(a_count / b_count))


def extract_hrv_features(rr_sec: np.ndarray, ppg_window: np.ndarray, fs: float = 20.0) -> np.ndarray:
    """Extract time-domain, frequency-domain, and nonlinear HRV features.

    Features returned (10-dim vector):
    [HR, SDNN, RMSSD, pNN20, pNN50, HF, TF, SD1, SD2, sampEn]
    Note: POB (Percentage of Oscillatory Beats) from Section 2.4.2.1 is omitted
    because the paper does not specify its exact threshold formula.
    """
    if len(rr_sec) < 3:
        return np.zeros(10, dtype=np.float32)

    rr_ms = rr_sec * 1000.0
    diff_rr_ms = np.diff(rr_ms)

    hr = 60.0 / np.mean(rr_sec)
    sdnn = np.std(rr_ms, ddof=1)
    rmssd = np.sqrt(np.mean(diff_rr_ms**2))
    pnn20 = np.mean(np.abs(diff_rr_ms) > 20.0) * 100.0
    pnn50 = np.mean(np.abs(diff_rr_ms) > 50.0) * 100.0

    # Frequency-domain from PPG PSD
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    freqs, psd = welch(ppg_window, fs=fs, nperseg=min(len(ppg_window), 256))
    hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
    hf = _trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0.0
    tf = _trapz(psd, freqs)

    # Nonlinear Poincare & Sample Entropy
    sd1 = np.sqrt(0.5 * np.var(diff_rr_ms, ddof=1))
    sd2_sq = max(0.0, 2.0 * np.var(rr_ms, ddof=1) - sd1**2)
    sd2 = np.sqrt(sd2_sq)
    samp_en = sample_entropy(rr_ms, m=2)

    return np.array([hr, sdnn, rmssd, pnn20, pnn50, hf, tf, sd1, sd2, samp_en], dtype=np.float32)


def binarize_stress_labels(likert_scores: np.ndarray, high_threshold: int = 4) -> np.ndarray:
    """Map 5-point Likert scores: 1-3 -> 0 (low stress), 4-5 -> 1 (high stress)."""
    return (np.asarray(likert_scores) >= high_threshold).astype(int)


def build_stress_mlp(random_state: int = 42) -> Pipeline:
    """Build standardized MLP classifier for stress level classification."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )
