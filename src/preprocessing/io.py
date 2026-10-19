"""io.py — Persistensi & ringkasan dataset terproses.

save_dataset (dengan monitor NaN/Inf pra-simpan), load_dataset, ringkasan,
dan ProgressTracker. Dipindah dari preprocessing/utils.py (perilaku identik).
"""

import numpy as np
from pathlib import Path
from typing import Optional
import json
import time


def save_dataset(
    output_dir: Path,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict,
    metadata: Optional[dict] = None,
) -> Path:
    """
    Save preprocessed dataset as .npz file with metadata as JSON.

    Parameters
    ----------
    output_dir : Path
        Directory to save the dataset.
    dataset_name : str
        Name of the dataset (used as filename prefix).
    X : np.ndarray
        Feature array. Shape: (N, n_channels, window_size).
    y : np.ndarray
        Label array. Shape: (N,).
    label_map : dict
        Mapping from label name to integer.
    metadata : dict, optional
        Additional metadata (torque levels, speed conditions, etc.).

    Returns
    -------
    Path
        Path to the saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Pipeline monitor: catch corruption BEFORE writing to disk ────────
    nan_windows  = int(np.isnan(X).any(axis=(1, 2)).sum())
    inf_windows  = int(np.isinf(X).any(axis=(1, 2)).sum())
    zero_windows = int((X == 0).all(axis=(1, 2)).sum())
    print(f"  [MONITOR] NaN windows: {nan_windows} | "
          f"Inf windows: {inf_windows} | "
          f"All-zero windows: {zero_windows}")
    if nan_windows > 0:
        # Diagnose: which channel has NaN?
        nan_per_ch = np.isnan(X).any(axis=2).sum(axis=0)  # (n_channels,)
        nan_ch_info = ", ".join(
            f"ch{i}={int(n)}" for i, n in enumerate(nan_per_ch) if n > 0
        )
        raise AssertionError(
            f"[ABORT] {nan_windows} NaN windows detected in '{dataset_name}' before save. "
            f"NaN by channel: [{nan_ch_info}]. "
            f"Check signal onset detection and envelope computation."
        )
    if inf_windows > 0:
        raise AssertionError(
            f"[ABORT] {inf_windows} Inf windows detected in '{dataset_name}' before save. "
            f"Check float32 overflow in preprocessing."
        )
    if zero_windows > 0:
        print(f"  [WARN] {zero_windows} all-zero windows in '{dataset_name}' — "
              f"may indicate silent NaN-fill or padding issue.")

    # Save numpy arrays
    npz_path = output_dir / f"{dataset_name}.npz"
    np.savez_compressed(npz_path, X=X, y=y)
    print(f"  [SAVED] {npz_path}")
    print(f"          X shape: {X.shape}, y shape: {y.shape}")
    print(f"          File size: {npz_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save metadata as JSON
    meta = {
        "dataset_name": dataset_name,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "n_samples": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "window_size": int(X.shape[2]),
        "label_map": label_map,
        "label_counts": {
            str(k): int(v) for k, v in
            zip(*np.unique(y, return_counts=True))
        },
        "dtype": str(X.dtype),
    }
    if metadata:
        meta.update(metadata)

    json_path = output_dir / f"{dataset_name}_meta.json"
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [SAVED] {json_path}")

    return npz_path


def load_dataset(npz_path: Path) -> tuple:
    """
    Load preprocessed dataset from .npz file.

    Returns
    -------
    tuple
        (X, y, metadata_dict)
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    X = data["X"]
    y = data["y"]

    # Load metadata if exists
    json_path = npz_path.with_suffix("").with_name(
        npz_path.stem + "_meta.json"
    )
    metadata = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)

    return X, y, metadata


def print_dataset_summary(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    label_map: dict,
) -> None:
    """
    Print a detailed summary of the preprocessed dataset.
    """
    label_names = {v: k for k, v in label_map.items()}
    unique, counts = np.unique(y, return_counts=True)

    print()
    print("=" * 60)
    print(f"  DATASET SUMMARY: {dataset_name}")
    print("=" * 60)
    print(f"  Total windows:  {X.shape[0]}")
    print(f"  Channels:       {X.shape[1]}")
    print(f"  Window size:    {X.shape[2]}")
    print(f"  Data type:      {X.dtype}")
    print(f"  Memory usage:   {X.nbytes / 1024 / 1024:.1f} MB")
    print()
    print("  Class distribution:")
    print("  " + "-" * 40)
    for label_id, count in zip(unique, counts):
        name = label_names.get(label_id, f"unknown_{label_id}")
        pct = count / len(y) * 100
        bar = "█" * int(pct / 2)
        print(f"    [{label_id}] {name:25s} {count:6d} ({pct:5.1f}%) {bar}")
    print("  " + "-" * 40)
    print(f"    {'TOTAL':27s} {len(y):6d}")

    # Class imbalance check
    if len(counts) > 1:
        imbalance_ratio = float(counts.max()) / float(counts.min())
        if imbalance_ratio > 2.0:
            print(f"  [WARN] Class imbalance ratio: {imbalance_ratio:.1f}x "
                  f"(max={counts.max()}, min={counts.min()}) — "
                  f"consider weighted loss or oversampling.")
        else:
            print(f"  [OK]   Class imbalance ratio: {imbalance_ratio:.1f}x (balanced)")
    print()

    # Basic statistics per channel
    print("  Per-channel statistics (global):")
    for ch in range(X.shape[1]):
        ch_data = X[:, ch, :]
        print(
            f"    CH{ch}: mean={ch_data.mean():.6f}, "
            f"std={ch_data.std():.6f}, "
            f"min={ch_data.min():.6f}, "
            f"max={ch_data.max():.6f}"
        )
    print("=" * 60)
    print()


class ProgressTracker:
    """Simple progress tracker for long preprocessing operations."""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        self.current += n
        elapsed = time.time() - self.start_time
        pct = self.current / self.total * 100
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0
        bar_len = 30
        filled = int(bar_len * self.current / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  {self.desc}: |{bar}| {pct:5.1f}% "
            f"[{self.current}/{self.total}] "
            f"ETA: {eta:.0f}s",
            end="", flush=True,
        )
        if self.current >= self.total:
            print(f"  Done in {elapsed:.1f}s")
