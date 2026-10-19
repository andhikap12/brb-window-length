"""protokol.py — Konstruksi fold untuk setiap protokol evaluasi.

Diekstrak dari train_motor_dann.py (blok inline lama di sekitar baris 438-450)
supaya protokol baru bisa diuji tanpa menyentuh GPU, dan supaya skrip latih
kembali jadi orkestrasi murni.

  'lto'          leave-one-torque-out, GroupKFold atas torsi (perilaku default lama)
  'extremes'     latih di torsi tengah, uji di n_extremes terendah + tertinggi
"""

import numpy as np
from sklearn.model_selection import GroupKFold

PROTOKOL_SAH = ("lto", "extremes")


def bangun_fold(
    protokol: str,
    X_len: int,
    y_fault: np.ndarray,
    torque_levels: np.ndarray,
    n_folds: int,
    n_extremes: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Kembalikan daftar (train_idx, test_idx) untuk protokol yang diminta.

    `X_len` dipakai sebagai stand-in X untuk GroupKFold.split, yang hanya
    membutuhkan panjangnya. Ini menghindari memindahkan array fitur besar
    ke sini hanya untuk dibuang.
    """
    if protokol not in PROTOKOL_SAH:
        raise ValueError(f"Protokol tak dikenal: {protokol!r}. Pilih dari {PROTOKOL_SAH}.")

    dummy = np.zeros(X_len, dtype=np.int8)

    if protokol == "extremes":
        uniq = sorted(set(int(t) for t in torque_levels.tolist()))
        ne = max(1, n_extremes)
        torsi_uji = set(uniq[:ne]) | set(uniq[-ne:])
        mask = np.isin(torque_levels, list(torsi_uji))
        return [(np.where(~mask)[0], np.where(mask)[0])]

    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(dummy, y_fault, groups=torque_levels))
