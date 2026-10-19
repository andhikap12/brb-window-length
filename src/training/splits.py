"""splits.py — Cek kebocoran rekaman & remap domain per-fold (LTO/LSO-CV)."""

import numpy as np


def verify_no_leakage(
    recording_ids: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> None:
    """Abort if any physical recording appears in both train and valid splits."""
    overlap = set(recording_ids[train_idx]) & set(recording_ids[valid_idx])
    if overlap:
        raise RuntimeError(
            f"[ABORT] Data leakage detected: {len(overlap)} recordings "
            f"appear in both train and valid splits."
        )
    print(
        f"  [OK] Leakage check passed: "
        f"{len(set(recording_ids[train_idx]))} train recordings, "
        f"{len(set(recording_ids[valid_idx]))} valid recordings, 0 overlap."
    )


def build_fold_domain_targets(
    domain_values: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> tuple[np.ndarray, int, dict[int, int]]:
    """
    Map only TRAINING domains (torque levels / speeds) to contiguous 0-based IDs;
    keep unseen validation domains masked (-1) so the GRL never trains on them.
    """
    train_domains = np.sort(np.unique(domain_values[train_idx]))
    domain_to_idx = {int(d): idx for idx, d in enumerate(train_domains.tolist())}

    fold_domain = np.full(domain_values.shape, -1, dtype=np.int64)
    fold_domain[train_idx] = np.array(
        [domain_to_idx[int(d)] for d in domain_values[train_idx]], dtype=np.int64
    )
    fold_domain[valid_idx] = -1
    return fold_domain, len(domain_to_idx), domain_to_idx


def verify_fold_domain_targets(
    y_domain: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray, n_domains: int
) -> None:
    """Sanity-check that unseen validation domains never become trainable classes."""
    train_domains = np.unique(y_domain[train_idx])
    valid_domains = np.unique(y_domain[valid_idx])

    if np.any(train_domains < 0):
        raise RuntimeError("[ABORT] Training split contains masked domain labels.")
    if set(valid_domains.tolist()) != {-1}:
        raise RuntimeError("[ABORT] Validation split contains active domain labels.")
    if len(train_domains) != n_domains:
        raise RuntimeError(
            f"[ABORT] Domain remap mismatch: expected {n_domains} training domains, "
            f"got {len(train_domains)}."
        )


def prepare_fold_domains(
    domain_values: np.ndarray, train_idx: np.ndarray, valid_idx: np.ndarray
) -> tuple[np.ndarray, int, dict[int, int]]:
    """Build fold domain targets and assert they are leakage-safe (the identical
    pair both training scripts run per fold). verify_no_leakage stays separate in
    the caller (it needs recording_ids, not domain_values)."""
    y_domain, n_domains, mapping = build_fold_domain_targets(domain_values, train_idx, valid_idx)
    verify_fold_domain_targets(y_domain, train_idx, valid_idx, n_domains)
    return y_domain, n_domains, mapping
