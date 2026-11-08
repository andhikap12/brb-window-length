"""Karakterisasi band_envelope, dan penjaga stabilitas filter."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from scipy.signal import butter, filtfilt, hilbert, sosfiltfilt

from src.config import MOTOR_VIB_FS, SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ
from src.preprocessing.stages.s4_envelope import band_envelope


def _golden_sos(signal_2d, fs, f_low, f_high, order=4):
    """Perilaku yang DIMAKUKAN sekarang: rerata -> butter(sos) -> hilbert -> abs."""
    mean_signal = signal_2d.mean(axis=0).astype(np.float64)
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    envelope = np.abs(hilbert(sosfiltfilt(sos, mean_signal))).astype(np.float32)
    return envelope[np.newaxis, :]


def _make_signal(n_ch, n, fs=MOTOR_VIB_FS):
    rng = np.random.default_rng(42)
    t = np.arange(n) / fs
    base = np.sin(2 * np.pi * 60 * t)  # komponen 60 Hz
    return (base[None, :] + 0.1 * rng.standard_normal((n_ch, n))).astype(np.float32)


def test_slip_band_matches_sos():
    sig = _make_signal(5, 8192)
    new = band_envelope(sig, MOTOR_VIB_FS, SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ, 4)
    old = _golden_sos(sig, MOTOR_VIB_FS, SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ, 4)
    assert new.shape == old.shape == (1, 8192)
    assert np.allclose(new, old, atol=1e-5), np.abs(new - old).max()


def test_beda_terhadap_ba_kecil_di_titik_operasi():
    """Penggantian ke sos tidak menggeser hasil secara berarti pada fs yang BERLAKU."""
    sig = _make_signal(5, 8192)
    baru = band_envelope(sig, MOTOR_VIB_FS, SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ, 4)

    nyq = MOTOR_VIB_FS / 2.0
    b, a = butter(4, [SLIP_BAND_LOW_HZ / nyq, SLIP_BAND_HIGH_HZ / nyq], btype="bandpass")
    lama = np.abs(hilbert(filtfilt(b, a, sig.mean(axis=0).astype(np.float64))))

    rel = np.abs(baru[0] - lama).max() / np.abs(lama).max()
    assert rel < 1e-3, f"pergeseran {rel:.2e} lebih besar dari yang terukur (4,5e-04)"


@pytest.mark.parametrize(
    "fs,lo,hi",
    [(MOTOR_VIB_FS, SLIP_BAND_LOW_HZ, SLIP_BAND_HIGH_HZ), (16000.0, 800.0, 1600.0)],
)
def test_kutub_di_dalam_lingkaran(fs, lo, hi):
    """Penjaga stabilitas. Bentuk `ba` pada fs 15990 yang dicabut memberi kutub"""
    nyq = fs / 2.0
    _, a = butter(4, [lo / nyq, hi / nyq], btype="bandpass")
    kutub = np.max(np.abs(np.roots(a)))
    assert kutub < 1.0, f"filter tidak stabil pada fs={fs}, pita {lo}-{hi}: kutub {kutub:.6f}"


if __name__ == "__main__":
    test_slip_band_matches_sos()
    test_beda_terhadap_ba_kecil_di_titik_operasi()
    print("OK test_s3_s4_envelope")
