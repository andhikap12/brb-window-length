"""Unit test: summarize_cross_seed computes mean/std per metric across seeds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.reporting import summarize_cross_seed


def test_mean_std_across_seeds():
    per_seed = [
        {"seed": 1, "win_f1_macro": 0.80, "rec_accuracy": 0.90},
        {"seed": 2, "win_f1_macro": 0.90, "rec_accuracy": 0.90},
    ]
    out = summarize_cross_seed(
        per_seed, metrics=[("Win-F1-macro", "win_f1_macro"), ("Rec-Accuracy", "rec_accuracy")]
    )
    assert abs(out["win_f1_macro"][0] - 0.85) < 1e-9   # mean
    assert abs(out["win_f1_macro"][1] - 0.05) < 1e-9   # std
    assert abs(out["rec_accuracy"][0] - 0.90) < 1e-9
    assert abs(out["rec_accuracy"][1] - 0.0) < 1e-9


if __name__ == "__main__":
    test_mean_std_across_seeds()
    print("OK test_reporting")
