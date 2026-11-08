"""motor.py - Preprocessor for Dataset A (Motor Rotor Bar Fault)."""

import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.config import (
    DATASET_A_DIR,
    PROCESSED_DIR,
    MOTOR_MAT_FILES,
    MOTOR_VIB_CHANNELS,
    MOTOR_CURRENT_CHANNELS,
    MOTOR_TORQUE_LEVELS,
    MOTOR_LABEL_MAP,
    MOTOR_BINARY_MAP,
    MOTOR_VIB_FS,
    SLIP_BAND_LOW_HZ,
    SLIP_BAND_HIGH_HZ,
    SLIP_BAND_FILTER_ORDER,
    WINDOW_SIZE,
    STRIDE,
    MOTOR_STRIDE,
)
from src.preprocessing.stages.s1_load_raw import load_motor_channels, read_trigger
from src.preprocessing.stages.s2_steady_state import find_onset, trim_to_onset
from src.preprocessing.stages.s4_envelope import band_envelope
from src.preprocessing.stages.s5_windowing import sliding_window
from src.preprocessing.io import save_dataset, print_dataset_summary, ProgressTracker


def compute_slip_band_envelope(
    signal_2d: np.ndarray,
    fs: float = MOTOR_VIB_FS,
    f_low: float = SLIP_BAND_LOW_HZ,
    f_high: float = SLIP_BAND_HIGH_HZ,
    order: int = SLIP_BAND_FILTER_ORDER,
) -> np.ndarray:
    """Slip-band amplitude envelope (broken-bar AM detection). Lihat stages/s4_envelope."""
    return band_envelope(signal_2d, fs, f_low, f_high, order)


class MotorDatasetPreprocessor:
    """Preprocessor for the Motor Rotor Bar Fault dataset."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        channels: Optional[list] = None,
        torque_levels: Optional[list] = None,
        steady_state_only: bool = True,
        trigger_threshold: float = 0.5,
        add_slip_envelope: bool = True,
        extra_env_bands: Optional[list] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATASET_A_DIR
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DIR / "motor"
        self.window_size = window_size
        self.stride = stride
        self.channels = channels or MOTOR_VIB_CHANNELS
        self.torque_levels = torque_levels or MOTOR_TORQUE_LEVELS
        self.steady_state_only = steady_state_only
        self.trigger_threshold = trigger_threshold
        self.add_slip_envelope = add_slip_envelope
        self.extra_env_bands = list(extra_env_bands or [])
    def _extract_multichannel_signal(
        self,
        filepath: Path,
        condition_key: str,
        torque_level: str,
        repetition: int,
    ) -> np.ndarray:
        """Extract multi-channel vibration signal, optionally trimmed to"""
        channels = load_motor_channels(
            filepath, condition_key, torque_level, repetition, self.channels
        )
        if self.steady_state_only:
            trig = read_trigger(filepath, condition_key, torque_level, repetition)
            onset = find_onset(trig, self.trigger_threshold)
        else:
            onset = 0
        multichannel = trim_to_onset(channels, onset, channels.shape[1])

        # Interpolasi NaN sisa (orkestrasi, bukan tahap Gambar 3.4) — SALIN dari lama
        n_raw_nan = int(np.isnan(multichannel).sum())
        if n_raw_nan > 0:
            print(f"  [WARN] {condition_key}/{torque_level}/rep{repetition}: "
                  f"{n_raw_nan} NaN in raw signal → interpolating")
            for ch in range(multichannel.shape[0]):
                nans = np.isnan(multichannel[ch])
                if nans.any():
                    x = np.arange(multichannel.shape[1])
                    valid = ~nans
                    if valid.any():
                        multichannel[ch][nans] = np.interp(x[nans], x[valid], multichannel[ch][valid])
                    else:
                        multichannel[ch] = 0.0

        vib = multichannel
        extra = []
        if self.add_slip_envelope:
            extra.append(compute_slip_band_envelope(vib))
        for f_low, f_high in self.extra_env_bands:
            extra.append(band_envelope(vib, MOTOR_VIB_FS, float(f_low), float(f_high),
                                       SLIP_BAND_FILTER_ORDER))
        if extra:
            multichannel = np.concatenate([vib] + extra, axis=0)
        return multichannel

    def process(
        self,
        label_mode: str = "multiclass",
        save: bool = True,
    ) -> tuple:
        """Run the full preprocessing pipeline."""
        label_map = MOTOR_LABEL_MAP if label_mode == "multiclass" else MOTOR_BINARY_MAP

        # Count total iterations for progress tracking
        total_iters = (
            len(MOTOR_MAT_FILES)
            * len(self.torque_levels)
            * 10  # repetitions
        )
        progress = ProgressTracker(total_iters, desc="Motor Dataset")

        all_windows = []
        all_labels = []
        all_torque = []      # metadata: which torque level each window came from
        all_repetition = []  # metadata: which repetition each window came from

        print(f"\n{'='*60}")
        print(f"  PREPROCESSING: Motor Rotor Bar Fault Dataset")
        n_ch_out = (len(self.channels) + (1 if self.add_slip_envelope else 0)
                    + len(self.extra_env_bands))
        print(f"  Channels:       {self.channels}")
        print(f"  Slip envelope:  {self.add_slip_envelope} "
              f"(band {SLIP_BAND_LOW_HZ}–{SLIP_BAND_HIGH_HZ} Hz → Hilbert)")
        if self.extra_env_bands:
            print(f"  Envelope extra: {self.extra_env_bands} Hz → Hilbert "
                  f"({len(self.extra_env_bands)} kanal tambahan)")
        print(f"  c_in (total):   {n_ch_out}")
        print(f"  Window size:    {self.window_size}")
        print(f"  Stride:         {self.stride}")
        print(f"  Label mode:     {label_mode}")
        print(f"  Steady state:   {self.steady_state_only} (threshold={self.trigger_threshold})")
        print(f"{'='*60}\n")

        for condition, mat_filename in MOTOR_MAT_FILES.items():
            filepath = self.data_dir / mat_filename
            if not filepath.exists():
                print(f"  [WARNING] File not found: {filepath}, skipping...")
                progress.update(len(self.torque_levels) * 10)
                continue

            label = label_map[condition]

            for torque in self.torque_levels:
                for rep in range(10):
                    try:
                        multichannel = self._extract_multichannel_signal(
                            filepath, condition, torque, rep
                        )
                        windows = sliding_window(
                            multichannel, self.window_size, self.stride
                        )
                        # windows shape: (n_windows, n_channels, window_size)

                        n_win = windows.shape[0]
                        all_windows.append(windows)
                        all_labels.append(np.full(n_win, label, dtype=np.int64))

                        # Store metadata per window
                        torque_idx = self.torque_levels.index(torque)
                        all_torque.append(
                            np.full(n_win, torque_idx, dtype=np.int32)
                        )
                        all_repetition.append(
                            np.full(n_win, rep, dtype=np.int32)
                        )

                    except Exception as e:
                        print(
                            f"\n  [ERROR] {condition}/{torque}/rep{rep}: {e}"
                        )

                    progress.update(1)

        # Concatenate all windows
        X = np.concatenate(all_windows, axis=0)
        y = np.concatenate(all_labels, axis=0)
        torque_info = np.concatenate(all_torque, axis=0)
        repetition_info = np.concatenate(all_repetition, axis=0)

        # Print summary
        print_dataset_summary("Motor Rotor Bar Fault", X, y, label_map)

        # Save
        if save:
            ss_tag = "_ss" if self.steady_state_only else ""
            env_tag = "_env" if self.add_slip_envelope else ""
            extra_tag = "".join(f"_env{int(lo)}-{int(hi)}" for lo, hi in self.extra_env_bands)
            dataset_name = (f"motor_{label_mode}_w{self.window_size}_s{self.stride}"
                            f"{ss_tag}{env_tag}{extra_tag}")
            ch_names = (self.channels
                        + (["slip_band_envelope"] if self.add_slip_envelope else [])
                        + [f"envelope_{int(lo)}_{int(hi)}" for lo, hi in self.extra_env_bands])
            metadata = {
                "label_mode": label_mode,
                "channels_used": ch_names,
                "c_in": len(ch_names),
                "add_slip_envelope": self.add_slip_envelope,
                "extra_env_bands": self.extra_env_bands,
                "slip_band_hz": [SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ] if self.add_slip_envelope else None,
                "torque_levels": self.torque_levels,
                "window_size": self.window_size,
                "stride": self.stride,
                "source": "banco de dados experimental",
                "description": (
                    "Motor rotor bar fault dataset. "
                    "Vibration signals from induction motor with 0-4 broken bars. "
                    + ("Includes slip-band envelope channel (40–100 Hz Hilbert)." if self.add_slip_envelope else "")
                ),
            }
            save_dataset(self.output_dir, dataset_name, X, y, label_map, metadata)

            # Simpan metadata torsi & repetisi
            meta_path = self.output_dir / f"{dataset_name}_split_info.npz"
            np.savez_compressed(
                meta_path,
                torque=torque_info,
                repetition=repetition_info,
            )
            print(f"  [SAVED] {meta_path} (torque & repetition metadata)")

        return X, y, torque_info, repetition_info


def main():
    """Run Motor dataset preprocessing with steady-state filtering."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    preprocessor = MotorDatasetPreprocessor(steady_state_only=True, stride=MOTOR_STRIDE)

    # Multiclass (5 classes)
    X, y, torque, reps = preprocessor.process(label_mode="multiclass", save=True)

    # Binary (healthy vs faulty)
    X_bin, y_bin, _, _ = preprocessor.process(label_mode="binary", save=True)


if __name__ == "__main__":
    main()
