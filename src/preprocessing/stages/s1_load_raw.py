"""s1_load_raw.py — Tahap 1: Sinyal Mentah Multivariat.

Pembaca HDF5 .mat (v7.3) untuk database motor. Mengembalikan array
(n_channels, n_samples) mentah.
"""

import numpy as np
import h5py


def load_motor_channels(
    filepath, condition_key: str, torque_level: str, repetition: int, channels: list
) -> np.ndarray:
    """Baca kanal getaran motor dari .mat v7.3 (referensi objek HDF5).
    Return (n_channels, n_samples) — di-trim ke panjang kanal terpendek."""
    channels_data = []
    with h5py.File(filepath, "r") as f:
        for channel in channels:
            ref = f[condition_key][torque_level][channel][repetition, 0]  # type: ignore
            signal = f[ref][:].flatten().astype(np.float32)  # type: ignore
            channels_data.append(signal)
    min_len = min(ch.shape[0] for ch in channels_data)
    return np.stack([ch[:min_len] for ch in channels_data], axis=0)


def read_trigger(filepath, condition_key: str, torque_level: str, repetition: int):
    """Baca kanal Trigger (untuk deteksi steady-state). None bila absen."""
    try:
        with h5py.File(filepath, "r") as f:
            ref = f[condition_key][torque_level]["Trigger"][repetition, 0]  # type: ignore
            return f[ref][:].flatten().astype(np.float32)  # type: ignore
    except Exception:
        return None
