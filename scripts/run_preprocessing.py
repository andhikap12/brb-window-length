"""
run_preprocessing.py - Entry point for preprocessing the motor dataset.

Usage:
    python scripts/run_preprocessing.py --motor
    python scripts/run_preprocessing.py --motor --window 2048 --stride 1024
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config_summary, WINDOW_SIZE, STRIDE, MOTOR_VIB_FS
from src.preprocessing.pipelines.motor import MotorDatasetPreprocessor

# Shortest steady-state stretch across all motor recordings (17.96 s, Sec. III-A of the
# CENIM 2026 paper), derived by scripts/audit/audit_steady_state.py.
N_STEADY_SHORTEST = 136470


def _pita(s: str) -> tuple:
    """Parse 'LO,HI' into a validated (f_low, f_high) pair in Hz."""
    try:
        lo, hi = (float(v) for v in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--env-band {s!r}: expected LO,HI")
    if not (0 < lo < hi < MOTOR_VIB_FS / 2):
        raise argparse.ArgumentTypeError(
            f"--env-band {s}: need 0 < LO < HI < Nyquist ({MOTOR_VIB_FS / 2})")
    return (lo, hi)


def main():
    # Robust against Windows cp1252 console for box-drawing / emoji glyphs in logs.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Preprocess vibration datasets for Predictive Maintenance with tsai",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_preprocessing.py --motor
  python scripts/run_preprocessing.py --motor --window 2048 --stride 1024  # Custom windowing
        """,
    )
    parser.add_argument(
        "--motor", action="store_true",
        help="Accepted for the runners' command lines; the motor dataset is the only one.",
    )
    parser.add_argument(
        "--window", type=int, default=WINDOW_SIZE,
        help=f"Window size in samples (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--stride", type=int, default=STRIDE,
        help=f"Stride in samples (default: {STRIDE})",
    )
    parser.add_argument(
        "--env-band", type=_pita, action="append", default=None, metavar="LO,HI",
        help="MOTOR ONLY. Extra Hilbert envelope band, repeatable (e.g. --env-band 110,130). "
             "The 40-100 Hz slip-band envelope is always kept; these are additional channels. "
             "Band edges land in the output filename, so arms stay traceable from --npz alone. "
             "Motivation: the P6 probe (audit/audit_normalisasi.py) measured +2.7 points from a "
             "110-130 Hz envelope spectrum on top of 40-100 alone, positive in 5/5 folds. "
             "No result in the paper uses these channels.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results (dry run)",
    )

    args = parser.parse_args()

    # Guard: a window longer than the SHORTEST steady-state stretch yields zero windows
    # for that recording, which sliding_window drops silently. Measured shortest is
    # 136470 samples = 17.96 s.
    if args.window > N_STEADY_SHORTEST:
        print(f"[ABORT] --window {args.window} exceeds the shortest steady-state stretch "
              f"({N_STEADY_SHORTEST} samples = {N_STEADY_SHORTEST / MOTOR_VIB_FS:.2f} s). "
              f"Short recordings would be dropped without warning.")
        return 1
    if args.stride > args.window:
        print(f"[ABORT] --stride {args.stride} > --window {args.window}: some samples "
              f"never enter any window.")
        return 1

    # Print config
    print(get_config_summary())
    if args.window != WINDOW_SIZE or args.stride != STRIDE:
        print(f"\n  [OVERRIDE] Window: {args.window}, Stride: {args.stride}")

    save = not args.no_save

    print("\n" + "█" * 60)
    print("  DATASET A: Motor Rotor Bar Fault")
    print("█" * 60)

    motor_proc = MotorDatasetPreprocessor(
        window_size=args.window,
        stride=args.stride,
        extra_env_bands=args.env_band or [],
    )
    motor_proc.process(label_mode="multiclass", save=save)
    motor_proc.process(label_mode="binary", save=save)

    print("\n" + "=" * 60)
    print("  [DONE] PREPROCESSING COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
