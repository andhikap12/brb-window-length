"""metrics.py — offline evaluation metrics (window- and recording-level).

Distilled verbatim from the inline logic previously duplicated in
train_motor_dann.evaluate_on_test.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def window_level_metrics(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> dict:
    """Per-window accuracy and macro-F1 over every window."""
    labels = list(range(n_classes))
    win_pred = probs.argmax(axis=1)
    return {
        "win_accuracy": float(accuracy_score(y_true, win_pred)),
        "win_f1_macro": float(
            f1_score(y_true, win_pred, average="macro", labels=labels, zero_division=0)
        ),
        "n_windows": int(len(y_true)),
    }


def recording_level_metrics(
    probs: np.ndarray, y_true: np.ndarray, recording_ids: np.ndarray, n_classes: int
) -> dict:
    """Majority vote (argmax of mean softmax) per physical recording."""
    labels = list(range(n_classes))
    rec_pred, rec_true = [], []
    for rid in np.unique(recording_ids):
        mask = recording_ids == rid
        rec_pred.append(int(probs[mask].mean(axis=0).argmax()))
        rec_true.append(int(y_true[mask][0]))
    rec_pred = np.array(rec_pred)
    rec_true = np.array(rec_true)
    return {
        "rec_accuracy": float(accuracy_score(rec_true, rec_pred)),
        "rec_f1_macro": float(
            f1_score(rec_true, rec_pred, average="macro", labels=labels, zero_division=0)
        ),
        "n_recordings": int(len(rec_true)),
    }
