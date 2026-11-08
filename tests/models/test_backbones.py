"""Unit tests: both backbones honor the (B,C,T)->(B,F) contract."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.models.backbones import build_backbone


def _check(name, c_in=6, seq_len=512):
    module, feat_dim = build_backbone(name, c_in, seq_len)
    out = module(torch.zeros(2, c_in, seq_len))
    assert out.shape == (2, feat_dim), (name, out.shape, feat_dim)
    assert feat_dim > 0


def test_tsitplus_contract():
    _check("tsitplus")


def test_tssequencerplus_contract():
    _check("tssequencerplus")


def test_unknown_backbone_raises():
    try:
        build_backbone("nope", 6, 512)
        assert False, "expected ValueError"
    except ValueError:
        pass
