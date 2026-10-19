"""Unit tests: DANNModel train/eval contract for both backbones, sanity_check, GRL effect."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from src.models.dann import build_dann_model

BACKBONES = ["tsitplus", "tssequencerplus"]


def test_train_eval_contract_all_backbones():
    for name in BACKBONES:
        m = build_dann_model(name, c_in=6, seq_len=512, n_classes=5, n_domains=3)
        m.train()
        out = m(torch.zeros(2, 6, 512))
        assert isinstance(out, tuple) and len(out) == 2, (name, type(out))
        fault, dom = out
        assert fault.shape == (2, 5) and dom.shape == (2, 3), (name, fault.shape, dom.shape)
        m.eval()
        assert m(torch.zeros(2, 6, 512)).shape == (2, 5), name


def test_sanity_check_all_backbones():
    for name in BACKBONES:
        build_dann_model(name, 6, 512, 5, 3).sanity_check(6, 512)


def test_grl_zero_lambda_blocks_backbone_grad_from_domain():
    # With lambda_grl=0 the GRL passes ZERO gradient to the backbone from the
    # domain path, so a domain-only loss must leave every backbone grad at zero.
    m0 = build_dann_model("tsitplus", 6, 256, 5, 3, lambda_grl=0.0)
    m0.train()
    _, dom0 = m0(torch.zeros(2, 6, 256))
    m0.zero_grad()
    dom0.sum().backward()
    assert any(p.grad is not None for p in m0.backbone.parameters()), \
        "no backbone grad allocated — GRL test would be vacuous"
    total0 = sum(p.grad.abs().sum().item() for p in m0.backbone.parameters() if p.grad is not None)
    assert total0 == 0.0, f"lambda=0 should give zero backbone grad from domain loss, got {total0}"


if __name__ == "__main__":
    test_train_eval_contract_all_backbones()
    test_sanity_check_all_backbones()
    test_grl_zero_lambda_blocks_backbone_grad_from_domain()
    print("OK test_dann_model")
