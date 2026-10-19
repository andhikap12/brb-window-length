"""audit_steady_state.py — panjang steady-state terpendek lintas 400 rekaman.

Makalah CENIM 2026 (Sec. III-A, V-D) menyatakan segmen steady-state terpendek berdurasi
17,96 s. Angka itu juga menjadi batas atas `--window` di `run_preprocessing.py`
(`N_STEADY_SHORTEST = 136470` sampel pada 7,6 kHz). Skrip ini menurunkannya dari data mentah
dengan kode ekstraksi yang sama persis dengan preprocessing (tahap s1 dan s2), tanpa
envelope dan tanpa windowing.

Jalankan dari akar repositori:
    ${PYTHON:-python} -u scripts/audit/audit_steady_state.py [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import MOTOR_MAT_FILES, MOTOR_TORQUE_LEVELS, MOTOR_VIB_FS  # noqa: E402
from src.preprocessing.pipelines.motor import MotorDatasetPreprocessor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="folder 'banco de dados experimental' (default: src/config.py)")
    args = ap.parse_args()

    pre = MotorDatasetPreprocessor(data_dir=args.data_dir, add_slip_envelope=False)
    panjang: dict[tuple[str, str, int], int] = {}
    for cond in MOTOR_MAT_FILES:
        path = pre.data_dir / MOTOR_MAT_FILES[cond]
        if not path.exists():
            print(f"[ABORT] berkas tidak ada: {path}")
            sys.exit(1)
        for torsi in MOTOR_TORQUE_LEVELS:
            for rep in range(10):
                x = pre._extract_multichannel_signal(path, cond, torsi, rep)
                panjang[(cond, torsi, rep)] = x.shape[1]

    terpendek = min(panjang, key=panjang.get)
    n = panjang[terpendek]
    print(f"rekaman diperiksa      : {len(panjang)}")
    print(f"steady-state terpendek : {n} sampel = {n / MOTOR_VIB_FS:.2f} s  {terpendek}")
    print(f"steady-state terpanjang: {max(panjang.values())} sampel")


if __name__ == "__main__":
    main()
