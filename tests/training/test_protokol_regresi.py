"""Gerbang regresi: bangun_fold() harus identik elemen-per-elemen dengan logika
inline lama yang pernah hidup di scripts/train_motor_dann.py (baris ~438-450,
sebelum diekstrak ke src/training/protokol.py pada Task 1/2).

Logika lama direproduksi verbatim di sini (bukan diimpor — sengaja, supaya tes
ini gagal kalau bangun_fold() berubah perilaku, bukan ikut berubah bersamanya).

Kompatibilitas mundur wajib bit-identik: dites di sini, bukan diasumsikan.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from src.training.protokol import bangun_fold

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_INFO_PATH = REPO_ROOT / "data" / "processed" / "motor" / "motor_multiclass_w4096_s4096_ss_env_split_info.npz"
NPZ_PATH = REPO_ROOT / "data" / "processed" / "motor" / "motor_multiclass_w4096_s4096_ss_env.npz"

# Dua tes memakai array torsi NYATA. Berkasnya hasil preprocessing lokal dan tidak
# ikut di repo, jadi di mesin bersih tes ini dilewati, bukan gagal palsu.
butuh_data_asli = pytest.mark.skipif(
    not (SPLIT_INFO_PATH.exists() and NPZ_PATH.exists()),
    reason=f"berkas motor terproses tidak ada ({SPLIT_INFO_PATH.name})",
)


def _data_sintetik(n_torsi=8, n_per=10):
    torsi = np.repeat(np.arange(n_torsi), n_per)
    y = np.tile(np.arange(5), len(torsi) // 5)
    return y, torsi


def _lto_lama(y_fault, torque_levels, n_folds):
    """Reproduksi verbatim blok inline lama untuk protokol 'lto'."""
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(np.zeros(len(torque_levels)), y_fault, groups=torque_levels))


def _extremes_lama(torque_levels, n_extremes):
    """Reproduksi verbatim blok inline lama untuk protokol 'extremes'."""
    uniq = sorted(set(int(t) for t in torque_levels.tolist()))
    ne = max(1, n_extremes)
    test_torques = set(uniq[:ne]) | set(uniq[-ne:])
    test_mask = np.isin(torque_levels, list(test_torques))
    return [(np.where(~test_mask)[0], np.where(test_mask)[0])]


def _bandingkan_fold(folds_lama, folds_baru):
    assert len(folds_lama) == len(folds_baru)
    for (tr_lama, te_lama), (tr_baru, te_baru) in zip(folds_lama, folds_baru):
        assert np.array_equal(tr_lama, tr_baru)
        assert np.array_equal(te_lama, te_baru)


def test_lto_identik_sintetik():
    y, torsi = _data_sintetik()
    folds_lama = _lto_lama(y, torsi, n_folds=8)
    folds_baru = bangun_fold("lto", len(torsi), y, torsi, n_folds=8, n_extremes=1)
    _bandingkan_fold(folds_lama, folds_baru)


@butuh_data_asli
def test_lto_identik_data_asli():
    split_info = np.load(SPLIT_INFO_PATH)
    torsi = split_info["torque"]
    y = np.load(NPZ_PATH)["y"]
    assert len(y) == len(torsi)
    n_folds = int(np.unique(torsi).size)  # 8 level torsi asli -> 8 lipatan
    folds_lama = _lto_lama(y, torsi, n_folds=n_folds)
    folds_baru = bangun_fold("lto", len(torsi), y, torsi, n_folds=n_folds, n_extremes=1)
    _bandingkan_fold(folds_lama, folds_baru)


def test_extremes_identik_ne1_sintetik():
    y, torsi = _data_sintetik()
    folds_lama = _extremes_lama(torsi, n_extremes=1)
    folds_baru = bangun_fold("extremes", len(torsi), y, torsi, n_folds=8, n_extremes=1)
    _bandingkan_fold(folds_lama, folds_baru)


def test_extremes_identik_ne2_sintetik():
    y, torsi = _data_sintetik()
    folds_lama = _extremes_lama(torsi, n_extremes=2)
    folds_baru = bangun_fold("extremes", len(torsi), y, torsi, n_folds=8, n_extremes=2)
    _bandingkan_fold(folds_lama, folds_baru)


@butuh_data_asli
def test_extremes_identik_data_asli():
    split_info = np.load(SPLIT_INFO_PATH)
    torsi = split_info["torque"]
    y = np.load(NPZ_PATH)["y"]
    folds_lama = _extremes_lama(torsi, n_extremes=1)
    folds_baru = bangun_fold("extremes", len(torsi), y, torsi, n_folds=8, n_extremes=1)
    _bandingkan_fold(folds_lama, folds_baru)


if __name__ == "__main__":
    test_lto_identik_sintetik()
    test_extremes_identik_ne1_sintetik()
    test_extremes_identik_ne2_sintetik()
    if SPLIT_INFO_PATH.exists() and NPZ_PATH.exists():
        test_lto_identik_data_asli()
        test_extremes_identik_data_asli()
    else:
        print("  (dilewati: berkas motor terproses tidak ada)")
    print("OK test_protokol_regresi")
