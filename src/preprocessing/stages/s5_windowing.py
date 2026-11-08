"""s5_windowing.py — Tahap 5: Sliding Window."""

import numpy as np


def sliding_window(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Apply sliding window to a 1D or multi-channel signal."""
    if signal.ndim == 1:
        n_samples = signal.shape[0]
        n_windows = (n_samples - window_size) // stride + 1
        if n_windows <= 0:
            raise ValueError(
                f"Signal length ({n_samples}) is shorter than window_size ({window_size})"
            )
        windows = np.zeros((n_windows, window_size), dtype=signal.dtype)
        for i in range(n_windows):
            start = i * stride
            windows[i] = signal[start : start + window_size]
        return windows

    elif signal.ndim == 2:
        n_channels, n_samples = signal.shape
        n_windows = (n_samples - window_size) // stride + 1
        if n_windows <= 0:
            raise ValueError(
                f"Signal length ({n_samples}) is shorter than window_size ({window_size})"
            )
        windows = np.zeros((n_windows, n_channels, window_size), dtype=signal.dtype)
        for i in range(n_windows):
            start = i * stride
            windows[i] = signal[:, start : start + window_size]
        return windows

    else:
        raise ValueError(f"Expected 1D or 2D signal, got {signal.ndim}D")
