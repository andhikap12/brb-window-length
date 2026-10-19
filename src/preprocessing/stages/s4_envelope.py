"""s4_envelope.py — Tahap 4: Ekstraksi Envelope Hilbert.

Bandpass (tahap 3) → Hilbert → magnitudo amplitudo, ditambahkan sebagai kanal baru.
Satu fungsi untuk pita mana pun; pita slip 40-100 Hz adalah yang dipakai makalah.
"""

import numpy as np
from scipy.signal import hilbert

from src.preprocessing.stages.s3_bandpass import bandpass_butter


def band_envelope(
    signal_2d: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 4
) -> np.ndarray:
    """Mean antar-kanal → bandpass → Hilbert → |·|. Return (1, n_samples) float32.

    Guard:
      - sinyal lebih pendek dari 3*(2*order+1) sampel → kembalikan nol (filtfilt
        butuh panjang minimum).
      - nilai non-finite pasca-Hilbert → diganti nol.
    """
    mean_signal = signal_2d.mean(axis=0).astype(np.float64)

    min_len = 3 * (2 * order + 1)  # butter bandpass orde-N → 2N+1 koefisien
    if len(mean_signal) < min_len:
        return np.zeros((1, signal_2d.shape[1]), dtype=np.float32)

    filtered = bandpass_butter(mean_signal, fs, f_low, f_high, order)
    envelope = np.abs(hilbert(filtered)).astype(np.float32)

    if not np.isfinite(envelope).all():
        envelope = np.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)

    return envelope[np.newaxis, :]
