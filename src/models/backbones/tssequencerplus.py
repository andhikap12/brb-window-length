"""tssequencerplus.py — Wrapper backbone untuk tsai TSSequencerPlus (BiLSTM, NON-transformer)."""

import torch
import torch.nn as nn
from tsai.models.TSSequencerPlus import TSSequencerPlus

MAKS_TOKEN = 256


class TSSequencerPlusBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        d_model: int = 128,
        depth: int = 3,
        token_size: int = 0,
    ) -> None:
        super().__init__()
        if token_size <= 0:
            token_size = max(64, seq_len // MAKS_TOKEN)
        _full = TSSequencerPlus(
            c_in=c_in,
            c_out=2,
            seq_len=seq_len,
            d_model=d_model,
            depth=depth,
            token_size=token_size,   # patch seq_len -> seq_len/token_size token
            use_token=True,          # CLS-token (bukan pooling) sebagai fitur
            use_pe=True,             # positional encoding -> domain/posisi terkode
        )
        self.backbone = _full.backbone   # -> (B, d_model, q_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        z = self.backbone(x)        # (B, d_model, q_len); CLS-token di index 0
        return z[:, :, 0]           # CLS-token feature -> (B, d_model)  [BUKAN GAP]
