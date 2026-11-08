"""Unit tests: jadwal lambda GRL (anneal Ganin) pada kedua callback DANN."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.callbacks import DANNCallback
from src.training.dann_supcon_callback import DANNSupConCallback


def _ganin(lambda_max: float, gamma: float, p: float) -> float:
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


@pytest.mark.parametrize("cls", [DANNCallback, DANNSupConCallback])
def test_lambda_follows_ganin_schedule(cls):
    cb = cls(lambda_max=1.0, gamma=10.0)
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        cb.learn = SimpleNamespace(pct_train=p)
        assert cb._get_lambda() == pytest.approx(_ganin(1.0, 10.0, p), abs=1e-9)
