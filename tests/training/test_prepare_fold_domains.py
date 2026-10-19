import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from src.training.splits import (
    build_fold_domain_targets, verify_fold_domain_targets, prepare_fold_domains,
)


def test_prepare_equals_underlying_calls():
    domain_values = np.array([0, 0, 1, 1, 2, 2])
    train_idx = np.array([0, 1, 2, 3])   # domains 0,1 (seen)
    valid_idx = np.array([4, 5])         # domain 2 (unseen)

    y_domain, n_domains, mapping = prepare_fold_domains(domain_values, train_idx, valid_idx)

    exp_y, exp_n, exp_map = build_fold_domain_targets(domain_values, train_idx, valid_idx)
    verify_fold_domain_targets(exp_y, train_idx, valid_idx, exp_n)   # must not raise

    assert np.array_equal(y_domain, exp_y)
    assert n_domains == exp_n == 2
    assert mapping == exp_map
    assert set(np.unique(y_domain[valid_idx]).tolist()) == {-1}   # unseen masked
