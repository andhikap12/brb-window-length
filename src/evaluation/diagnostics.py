"""diagnostics.py — generalization-gap reporting to surface overfitting."""

from __future__ import annotations


def generalization_gap(
    train_acc: float,
    test_acc: float,
    val_acc: float | None = None,
    threshold: float = 0.15,
) -> dict:
    """Report the train-vs-held-out accuracy gap and flag likely overfitting.

    gap = train_acc - test_acc (held-out / unseen domain).
    overfit is flagged when gap strictly exceeds `threshold` (default 0.15).
    `val_acc` (inner-validation accuracy) is optional — runs without an
    inner-validation split leave it None.
    """
    gap = float(train_acc) - float(test_acc)
    # Use a tiny tolerance so a gap exactly equal to the threshold is NOT
    # flagged: float subtraction (e.g. 0.80 - 0.65 = 0.150000000000000002)
    # would otherwise trip a bare `gap > threshold` at the boundary.
    return {
        "train_acc": float(train_acc),
        "val_acc": None if val_acc is None else float(val_acc),
        "test_acc": float(test_acc),
        "gap": gap,
        "overfit": gap > threshold + 1e-9,
    }
