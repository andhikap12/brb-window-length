"""tsitplus.py — Wrapper backbone untuk tsai TSiTPlus (transformer ViT-style)."""

import torch
import torch.nn as nn
from tsai.models.TSiTPlus import TSiTPlus

MAKS_TOKEN = 256


class TSiTPlusBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int = 128,
        depth: int = 3,
        n_heads: int = 8,
        token_size: int = 0,
    ) -> None:
        super().__init__()
        if token_size <= 0:
            token_size = max(64, seq_len // MAKS_TOKEN)
        # c_out=2 dummy: nn.Sequential(backbone, head) dibangun, lalu head dibuang.
        _full = TSiTPlus(
            c_in=c_in,
            c_out=2,
            seq_len=seq_len,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            token_size=token_size,   # patch seq_len -> seq_len/token_size token
            use_token=True,          # CLS-token (bukan pooling) sebagai fitur
            use_pe=True,             # positional encoding -> domain/posisi terkode
        )
        self.backbone = _full.backbone   # _TSiTBackbone -> (B, d_model, q_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        z = self.backbone(x)        # (B, d_model, q_len); CLS-token di index 0
        return z[:, :, 0]           # CLS-token feature -> (B, d_model)  [BUKAN GAP]
