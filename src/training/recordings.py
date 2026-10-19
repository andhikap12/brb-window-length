"""recordings.py — unique physical-recording IDs for grouped CV / majority vote."""

import numpy as np


def create_recording_ids(
    y: np.ndarray, torque_levels: np.ndarray, reps: np.ndarray
) -> np.ndarray:
    """
    Unique integer ID per physical recording (fault × torque × repetition).

    Uses base-conversion with sizes derived from actual data so the bijection
    holds even if n_torques or n_reps changes between dataset versions.
    Assumes torque_levels and reps are 0-indexed contiguous integers.
    """
    n_reps    = int(np.unique(reps).size)          # e.g. 10
    n_torques = int(np.unique(torque_levels).size)  # e.g. 8
    ids = (
        y.astype(int) * (n_torques * n_reps)
        + torque_levels.astype(int) * n_reps
        + reps.astype(int)
    )
    # Sanity: every combination must be unique
    expected_unique = int(np.unique(y).size) * n_torques * n_reps
    actual_unique   = int(np.unique(ids).size)
    if actual_unique != expected_unique:
        raise RuntimeError(
            f"[BUG-5] recording_ids collision: expected {expected_unique} unique IDs, "
            f"got {actual_unique}. Check that torque_levels and reps are 0-indexed."
        )
    return ids
