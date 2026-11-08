"""Karakterisasi: s5_windowing.sliding_window menghasilkan jendela yang benar."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.preprocessing.stages.s5_windowing import sliding_window


def test_2d_shapes_and_content():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((6, 10000)).astype(np.float32)
    w = sliding_window(sig, 4096, 2048)
    # n_windows = (10000-4096)//2048 + 1 = 3
    assert w.shape == (3, 6, 4096), w.shape
    assert np.array_equal(w[0], sig[:, 0:4096])
    assert np.array_equal(w[1], sig[:, 2048:2048 + 4096])
    assert np.array_equal(w[2], sig[:, 4096:4096 + 4096])


def test_1d_shapes():
    sig = np.arange(10000, dtype=np.float32)
    w = sliding_window(sig, 4096, 2048)
    assert w.shape == (3, 4096)
    assert np.array_equal(w[0], sig[0:4096])
    assert np.array_equal(w[1], sig[2048:2048 + 4096])
    assert np.array_equal(w[2], sig[4096:4096 + 4096])


if __name__ == "__main__":
    test_2d_shapes_and_content()
    test_1d_shapes()
    print("OK test_s5_windowing")
