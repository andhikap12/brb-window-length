"""s2_steady_state.py — Tahap 2: Deteksi Steady-State (Dataset A).

Trigger turun di bawah ambang saat motor mencapai steady-state; hanya sampel
pasca-falling-edge yang dipakai agar pelatihan tak terbias dinamika transien.
"""

import numpy as np


def find_onset(trigger: np.ndarray | None, threshold: float = 0.5) -> int:
    """Indeks sampel awal steady-state (falling edge pertama < threshold).
    Kembalikan 0 bila trigger tak ada / tak ada falling edge."""
    if trigger is None:
        return 0
    below = np.where(trigger < threshold)[0]
    return int(below[0]) if len(below) > 0 else 0


def trim_to_onset(channels: np.ndarray, onset: int, min_len: int) -> np.ndarray:
    """Potong (n_channels, n_samples) ke [onset:min_len]."""
    return channels[:, onset:min_len]
