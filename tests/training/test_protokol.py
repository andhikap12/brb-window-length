"""Karakterisasi: protokol.bangun_fold menghasilkan fold yang benar untuk dua protokol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
from src.training.protokol import bangun_fold


def _data(n_torsi=8, n_per=10):
    torsi = np.repeat(np.arange(n_torsi), n_per)
    y = np.tile(np.arange(5), len(torsi) // 5)
    return len(torsi), y, torsi


def test_lto_menahan_satu_torsi_per_lipatan():
    n, y, torsi = _data()
    folds = bangun_fold("lto", n, y, torsi, n_folds=8, n_extremes=1)
    assert len(folds) == 8
    for tr, te in folds:
        assert len(set(torsi[te].tolist())) == 1
        assert not (set(torsi[tr].tolist()) & set(torsi[te].tolist()))


def test_extremes_menahan_torsi_terendah_dan_tertinggi():
    n, y, torsi = _data()
    folds = bangun_fold("extremes", n, y, torsi, n_folds=8, n_extremes=1)
    assert len(folds) == 1
    tr, te = folds[0]
    assert set(torsi[te].tolist()) == {0, 7}
    assert set(torsi[tr].tolist()) == {1, 2, 3, 4, 5, 6}


def test_protokol_tak_dikenal_menolak():
    n, y, torsi = _data()
    with pytest.raises(ValueError, match="tak dikenal"):
        bangun_fold("ngawur", n, y, torsi, n_folds=8, n_extremes=1)


if __name__ == "__main__":
    test_lto_menahan_satu_torsi_per_lipatan()
    test_extremes_menahan_torsi_terendah_dan_tertinggi()
    test_protokol_tak_dikenal_menolak()
    print("OK test_protokol")
