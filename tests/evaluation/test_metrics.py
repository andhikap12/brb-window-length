"""Unit tests: recording- and window-level metric helpers."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.evaluation.metrics import recording_level_metrics, window_level_metrics


def test_window_level_perfect():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    y = np.array([0, 1, 0])
    r = window_level_metrics(probs, y, n_classes=2)
    assert r["win_accuracy"] == 1.0
    assert r["win_f1_macro"] == 1.0
    assert r["n_windows"] == 3


def test_recording_majority_vote():
    # recording 0: two windows, mean softmax -> class 0; recording 1: class 1
    probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    y = np.array([0, 0, 1])
    rec = np.array([0, 0, 1])
    r = recording_level_metrics(probs, y, rec, n_classes=2)
    assert r["n_recordings"] == 2
    assert r["rec_accuracy"] == 1.0


def test_recording_vote_overrides_single_window():
    # recording 0 has 3 windows; 2 vote class0, mean softmax favors class 0
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 0, 0])
    rec = np.array([5, 5, 5])
    r = recording_level_metrics(probs, y, rec, n_classes=2)
    assert r["n_recordings"] == 1
    assert r["rec_accuracy"] == 1.0  # mean softmax [0.6,0.4] -> class 0 == truth


if __name__ == "__main__":
    test_window_level_perfect()
    test_recording_majority_vote()
    test_recording_vote_overrides_single_window()
    print("OK test_metrics")
