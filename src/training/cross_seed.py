"""cross_seed.py — shared cross-seed CV driver + summary for training scripts.

Extracted verbatim from the block previously inline in train_motor_dann.py.
Output is byte-identical to that block (pinned by golden fixtures in
tests/training/golden/). The per-fold logic (run_cv_for_seed) stays in the script.
"""

from typing import Callable

import numpy as np

from src.training.reporting import summarize_cross_seed

STANDARD_METRICS = [
    ("Win-F1-macro", "win_f1_macro"), ("Win-Accuracy", "win_accuracy"),
    ("Rec-Accuracy", "rec_accuracy"), ("Rec-F1-macro", "rec_f1_macro"),
]


def seed_mean(seed: int, fold_results: list[dict]) -> dict:
    """Mean of the 4 standard metrics over folds, for one seed."""
    return {
        "seed": seed,
        "win_f1_macro": float(np.mean([r["win_f1_macro"] for r in fold_results])),
        "win_accuracy": float(np.mean([r["win_accuracy"] for r in fold_results])),
        "rec_accuracy": float(np.mean([r["rec_accuracy"] for r in fold_results])),
        "rec_f1_macro": float(np.mean([r["rec_f1_macro"] for r in fold_results])),
    }


def run_cross_seed(seeds: list[int], run_fold_cv: Callable[[int], list[dict]], *, cv_label: str, dataset_label: str | None = None) -> list[dict]:
    """Run run_fold_cv(seed) for each seed; print per-seed header + CV-mean line.

    run_fold_cv(seed) -> list[fold_result_dict] (each with the 4 metric keys).
    Returns per_seed_means (list of seed_mean dicts).
    """
    suffix = f" ({dataset_label})" if dataset_label else ""
    per_seed_means: list[dict] = []
    for seed in seeds:
        print(f"\n{'='*62}")
        print(f"  >>> SEED {seed} — full {cv_label} CV{suffix}")
        print(f"{'='*62}")
        fold_results = run_fold_cv(seed)
        sm = seed_mean(seed, fold_results)
        per_seed_means.append(sm)
        print(f"\n  [SEED {seed}] CV-mean over {len(fold_results)} folds: "
              f"Win-F1={sm['win_f1_macro']*100:.2f}%  Rec-Acc={sm['rec_accuracy']*100:.2f}%")
    return per_seed_means


def print_cross_seed_summary(per_seed_means, *, title: str, subtitle_lines: list[str],
                             domain_word: str, also_per_seed_rec: bool,
                             trailing_lines: list[str]) -> None:
    """Print the cross-seed mean/std summary block (byte-identical to the prior inline block)."""
    print(f"\n{'='*62}")
    print(f"  {title}")
    for line in subtitle_lines:
        print(f"  {line}")
    print(f"{'='*62}")
    summary = summarize_cross_seed(per_seed_means, metrics=STANDARD_METRICS)
    print("  Per-seed Win-F1-macro: "
          + ", ".join(f"{s['seed']}:{s['win_f1_macro']*100:.2f}%" for s in per_seed_means))
    if also_per_seed_rec:
        print("  Per-seed Rec-Accuracy: "
              + ", ".join(f"{s['seed']}:{s['rec_accuracy']*100:.2f}%" for s in per_seed_means))
    m_wf, s_wf = summary["win_f1_macro"]
    m_rec, s_rec = summary["rec_accuracy"]
    print(f"\n  >> PRIMARY   (unseen {domain_word}) Win-F1-macro : {m_wf*100:.2f}% +/- {s_wf*100:.2f}%")
    print(f"  >> SECONDARY (unseen {domain_word}) Rec-Accuracy : {m_rec*100:.2f}% +/- {s_rec*100:.2f}%")
    for line in trailing_lines:
        print(line)
