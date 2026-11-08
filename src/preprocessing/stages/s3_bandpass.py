"""s3_bandpass.py — Tahap 3: Bandpass Butterworth orde-4 (zero-phase)."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_butter(
    signal_1d: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 4
) -> np.ndarray:
    """Zero-phase bandpass (sosfiltfilt) pada sinyal 1D. Mengembalikan float64."""
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal_1d)
