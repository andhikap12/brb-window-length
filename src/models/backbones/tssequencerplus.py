"""tssequencerplus.py — Wrapper backbone untuk tsai TSSequencerPlus (BiLSTM, NON-transformer).

Ditambahkan 7 Agu 2026 untuk memeriksa batasan "backbone tunggal" (Sec. IV-B dan VI makalah
CENIM 2026): setidaknya satu arsitektur tambahan dengan evaluasi multi-seed.

MENGAPA TSSequencerPlus, BUKAN CNN
==================================
Arsitektur ini berasal dari "Sequencer: Deep LSTM for Image Classification"
(Tatsunami & Taki, 2022). Ia mengganti self-attention dengan **BiLSTM** dan
mempertahankan sisa kerangka ViT apa adanya.

Itu justru yang dibutuhkan di sini. Beralih ke InceptionTime/ResNet/CNN1D akan mengubah
DUA hal sekaligus: mekanisme pencampuran urutan DAN kembalinya BatchNorm + Global Average
Pooling. Dokumentasi `tsitplus.py` sudah mencatat bahwa BN+GAP membuat baseline
auto-invarian terhadap beban by construction, jadi hasil backbone CNN akan terkonfusi
dengan isu pencucian amplitudo itu dan tidak dapat diatribusikan.

TSSequencerPlus mempertahankan LayerNorm, positional encoding, CLS-token, dan tokenisasi
IDENTIK dengan TSiTPlus. Yang berubah hanya pencampur urutannya: attention -> BiLSTM.
Itu isolasi bias induktif yang sesungguhnya.

BIAYA — DIUKUR, BUKAN DITAKSIR (7 Agu 2026, RTX 4050 Laptop 6,4 GB, batch 32)
=============================================================================
                        T=4096, 64 token        T=16384, 256 token
    TSiTPlus            0,012 s/step 0,11 GB    0,063 s/step 0,96 GB
    TSSequencerPlus     0,014 s/step 0,14 GB    0,031 s/step 0,40 GB

Di jendela panjang ia dua kali lebih cepat dan 2,4 kali lebih ringan daripada TSiTPlus,
karena BiLSTM O(n) atas token sementara attention O(n^2).

Catatan negatif yang perlu diketahui: LSTM POLOS di panjang penuh tidak layak dan itu
terukur, 1,739 s/step dengan puncak 9,47 GB pada kartu 6,4 GB, yaitu ~9 jam per seed.
Karena itu keluarga recurrent hanya dapat diuji dalam bentuk ber-tokenisasi seperti ini.

BATAS JUMLAH TOKEN
==================
Aturannya disalin utuh dari `tsitplus.py` supaya perbandingan lintas backbone memakai
tokenisasi yang sama: `token_size = max(64, seq_len // MAKS_TOKEN)` mengunci jumlah token
di 256 dan membiarkan rentang waktu per token yang tumbuh.

    w4096  : 4096//256  =  16 -> max(64,16) = 64 ->  64 token
    w16384 : 16384//256 =  64 -> max(64,64) = 64 -> 256 token
    w65536 : 65536//256 = 256 -> max(64,256)= 256 -> 256 token
"""

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
        # Aturan identik dengan TSiTPlusBackbone. Nilai eksplisit tetap dihormati supaya
        # audit lama dapat mereproduksi konfigurasi persisnya.
        if token_size <= 0:
            token_size = max(64, seq_len // MAKS_TOKEN)
        # c_out=2 dummy: head klasifikasi tsai dibuang, perannya dimainkan
        # FaultHead/DomainHead di DANNModel. Sama seperti wrapper TSiTPlus.
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
