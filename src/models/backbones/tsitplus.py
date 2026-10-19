"""tsitplus.py — Wrapper backbone untuk tsai TSiTPlus (transformer ViT-style).

Tujuan ilmiah: backbone NON-GAP, NON-BatchNorm sebagai uji SYARAT-PERLU DANN.
CNN kita (InceptionTime/ResNet/CNN1D) memakai BatchNorm + Global Average Pooling
→ auto-invarian terhadap domain (beban) by construction → baseline sudah load-invarian
→ GRL tak punya ruang, hanya bisa overshoot (negative transfer, terbukti empiris).

TSiTPlus berbeda secara fundamental:
    - LayerNorm (bukan BatchNorm) → tak menghapus offset domain via statistik batch.
    - Positional encoding + CLS-token → MEMPERTAHANKAN informasi posisi/amplitudo/domain.
    - Fitur diambil dari CLS-token (z[:, :, 0]), BUKAN GAP atas dimensi waktu.

Hipotesis: baseline TSiTPlus akan MENGKODEKAN domain (domain_acc tinggi) dan overfit
beban → ada ruang nyata bagi DANN untuk memperbaiki generalisasi (DANN > baseline).

SeqTokenizer (token_size) memampatkan seq_len (4096) menjadi token agar attention
O(n^2) tractable. Head klasifikasi tsai dibuang (c_out dummy); peran head dimainkan
FaultHead/DomainHead di DANNModel.

BATAS JUMLAH TOKEN — ditambahkan 4 Agu 2026 untuk lengan w65536
===============================================================
token_size tetap 64 membuat jumlah token tumbuh linear terhadap seq_len:

    w4096  ->   64 token       w16384 ->  256 token       w65536 -> 1024 token

Attention O(n^2) pada 1024 token, batch 64, 8 kepala, 3 lapis, tidak muat di GPU ini.
Lebih penting daripada memori: membiarkan token tumbuh membuat perbandingan
w16384 lawan w65536 mengubah DUA hal sekaligus — rentang waktu per token DAN jumlah
token — sehingga kenaikan apa pun tidak dapat diatribusikan ke resolusi.

`token_size = max(64, seq_len // MAKS_TOKEN)` mengunci jumlah token di 256 dan
membiarkan rentang waktu per token yang tumbuh. Itu persis variabel yang R-P1 uji.

NETRAL untuk seluruh lengan yang sudah ada — diverifikasi, bukan diasumsikan:
    w4096  : 4096//256  =  16 -> max(64,16) = 64 ->  64 token  (sama seperti dulu)
    w16384 : 16384//256 =  64 -> max(64,64) = 64 -> 256 token  (sama seperti dulu)
    w65536 : 65536//256 = 256 -> max(64,256)= 256 -> 256 token (baru, dibatasi)
"""

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
        # token_size=0 -> pilih otomatis dengan batas MAKS_TOKEN. Nilai eksplisit
        # tetap dihormati supaya audit lama dapat mereproduksi konfigurasi persisnya.
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
