import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.cross_seed import seed_mean, run_cross_seed, print_cross_seed_summary

GOLDEN = Path(__file__).parent / "golden"
SEEDS = [42, 123]
FOLDS = {
    42:  [{"win_f1_macro": 0.80, "win_accuracy": 0.70, "rec_accuracy": 0.90, "rec_f1_macro": 0.85}] * 2,
    123: [{"win_f1_macro": 0.90, "win_accuracy": 0.80, "rec_accuracy": 0.95, "rec_f1_macro": 0.95}] * 2,
}


def test_seed_mean_means_over_folds():
    sm = seed_mean(42, FOLDS[42])
    assert sm == {"seed": 42, "win_f1_macro": 0.80, "win_accuracy": 0.70,
                  "rec_accuracy": 0.90, "rec_f1_macro": 0.85}


def _motor_output() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        psm = run_cross_seed(SEEDS, lambda s: FOLDS[s], cv_label="Leave-Torque-Out")
        print_cross_seed_summary(
            psm,
            title="DANN LTO-CV SUMMARY: motor_x",
            subtitle_lines=["lambda_max=1.0  |  seeds=[42, 123]",
                            "Reported as mean +/- std across seeds (each seed = full LTO-CV)."],
            domain_word="torque", also_per_seed_rec=True,
            trailing_lines=["=" * 62, "", "  COMPARISON GUIDE:",
                "  +----------------------------------------------------------+",
                "  | Run baseline:   --lambda_max 0.0 --name ..._baseline     |",
                "  | Run full DANN:  --lambda_max 1.0 --name ..._dann         |",
                "  | Compare by Win-F1-macro (primary), mean +/- std/seeds.   |",
                "  | A backbone wins only if it clears the cross-seed std.    |",
                "  +----------------------------------------------------------+", ""])
    return buf.getvalue()


def test_motor_cross_seed_byte_identical_to_golden():
    assert _motor_output() == (GOLDEN / "motor_cross_seed.txt").read_text(encoding="utf-8")
