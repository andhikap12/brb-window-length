"""Unit tests: GRL forward identity + reversed-scaled backward; head output shapes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.models.grl import gradient_reversal
from src.models.heads import FaultHead, DomainHead


def test_grl_forward_is_identity():
    x = torch.randn(4, 8)
    assert torch.equal(gradient_reversal(x, 1.0), x)


def test_grl_backward_negates_and_scales():
    x = torch.ones(3, requires_grad=True)
    y = gradient_reversal(x, 2.0)
    y.sum().backward()
    # d(sum)/dx would be +1; GRL reverses & scales by lambda=2 -> -2
    assert torch.allclose(x.grad, torch.full((3,), -2.0)), x.grad


def test_head_shapes():
    feat = torch.randn(5, 128)
    fh = FaultHead(128, 5)
    dh = DomainHead(128, 7)
    assert fh(feat).shape == (5, 5)
    assert dh(feat).shape == (5, 7)
    assert fh.out_features == 5 and dh.out_features == 7


if __name__ == "__main__":
    test_grl_forward_is_identity()
    test_grl_backward_negates_and_scales()
    test_head_shapes()
    print("OK test_grl_heads")
