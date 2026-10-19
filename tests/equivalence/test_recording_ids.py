"""Characterization: create_recording_ids == old base-conversion formula."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.training.recordings import create_recording_ids


def _old_formula(y, torque_levels, reps):
    n_reps = int(np.unique(reps).size)
    n_torques = int(np.unique(torque_levels).size)
    return (y.astype(int) * (n_torques * n_reps)
            + torque_levels.astype(int) * n_reps
            + reps.astype(int))


def test_matches_old_formula_and_is_unique():
    rng = np.random.default_rng(0)
    y, t, r = [], [], []
    for fault in range(5):
        for torque in range(8):
            for rep in range(10):
                y.append(fault); t.append(torque); r.append(rep)
    y = np.array(y); t = np.array(t); r = np.array(r)
    ids = create_recording_ids(y, t, r)
    assert np.array_equal(ids, _old_formula(y, t, r))
    assert np.unique(ids).size == 5 * 8 * 10


if __name__ == "__main__":
    test_matches_old_formula_and_is_unique()
    print("OK test_recording_ids")
