"""Unit tests: generalization_gap reports train-test gap and overfit flag."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.diagnostics import generalization_gap


def test_gap_and_no_overfit():
    r = generalization_gap(train_acc=0.90, test_acc=0.82, val_acc=0.85)
    assert abs(r["gap"] - 0.08) < 1e-9, r
    assert r["overfit"] is False
    assert r["val_acc"] == 0.85


def test_overfit_flagged_above_threshold():
    r = generalization_gap(train_acc=0.99, test_acc=0.70, val_acc=0.80)
    assert abs(r["gap"] - 0.29) < 1e-9, r
    assert r["overfit"] is True


def test_val_acc_optional():
    r = generalization_gap(train_acc=0.95, test_acc=0.90)
    assert r["val_acc"] is None
    assert r["overfit"] is False  # gap 0.05 < 0.15


def test_threshold_is_strict_greater_than():
    r = generalization_gap(train_acc=0.80, test_acc=0.65)  # gap exactly 0.15
    assert r["overfit"] is False  # 0.15 > 0.15 is False


if __name__ == "__main__":
    test_gap_and_no_overfit()
    test_overfit_flagged_above_threshold()
    test_val_acc_optional()
    test_threshold_is_strict_greater_than()
    print("OK test_diagnostics")
