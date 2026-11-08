"""Backbone registry untuk DANN. Kontrak: forward((B,C_in,T)) -> (B,F)."""

import torch

from src.models.backbones.tsitplus import TSiTPlusBackbone
from src.models.backbones.tssequencerplus import TSSequencerPlusBackbone

_BUILDERS = {
    "tsitplus": TSiTPlusBackbone,
    "tssequencerplus": TSSequencerPlusBackbone,
}


def build_backbone(name: str, c_in: int, seq_len: int):
    if name not in _BUILDERS:
        raise ValueError(f"Unknown backbone '{name}'. Valid: {sorted(_BUILDERS)}")
    module = _BUILDERS[name](c_in, seq_len)
    was_training = module.training
    module.eval()
    with torch.no_grad():
        feat = module(torch.zeros(2, c_in, seq_len))
    module.train(was_training)
    if feat.dim() != 2:
        raise RuntimeError(
            f"Backbone '{name}' must output (B, F); got shape {tuple(feat.shape)}."
        )
    return module, int(feat.shape[-1])
