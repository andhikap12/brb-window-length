"""diagnostics.py — generalization-gap reporting to surface overfitting."""

from __future__ import annotations


def generalization_gap(
    train_acc: float,
    test_acc: float,
    val_acc: float | None = None,
    threshold: float = 0.15,
) -> dict:
    """Report the train-vs-held-out accuracy gap and flag likely overfitting."""
    gap = float(train_acc) - float(test_acc)
    return {
        "train_acc": float(train_acc),
        "val_acc": None if val_acc is None else float(val_acc),
        "test_acc": float(test_acc),
        "gap": gap,
        "overfit": gap > threshold + 1e-9,
    }
