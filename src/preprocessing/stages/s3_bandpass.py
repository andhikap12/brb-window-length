"""s3_bandpass.py — Tahap 3: Bandpass Butterworth orde-4 (zero-phase).

Slip-band 40-100 Hz (juga dipakai untuk pita envelope tambahan lewat `--env-band`).
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_butter(
    signal_1d: np.ndarray, fs: float, f_low: float, f_high: float, order: int = 4
) -> np.ndarray:
    """Zero-phase bandpass (sosfiltfilt) pada sinyal 1D. Mengembalikan float64.

    Bentuk `sos`, bukan `b, a`. Pita 40-100 Hz pada fs 7600 memberi frekuensi
    ternormalisasi 0,0105-0,0263, yaitu wilayah tempat bentuk polinomial mulai
    kehilangan presisi. Diukur pada sinyal uji 153.504 sampel (fundamental 60 Hz +
    sideband 2sf torsi-0), selisih `ba` terhadap `sos` adalah 4,2e-04 relatif dan
    kutub terdekat berada di 0,9944 dari lingkaran satuan: `ba` masih stabil di sini,
    jadi ini perbaikan konsistensi, bukan perbaikan cacat.

    KONSEKUENSI: keluaran bergeser ~4e-04 relatif terhadap dataset yang sudah
    diekstrak. Seluruh hasil yang ada TIDAK perlu dihitung ulang (pergeseran itu
    tiga orde di bawah ukuran efek terkecil yang dipakai proyek ini, |d| = 0,149),
    tetapi ekstraksi ulang di masa depan tidak akan bit-identik dengan .npz lama.
    """
    nyq = fs / 2.0
    sos = butter(order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal_1d)
