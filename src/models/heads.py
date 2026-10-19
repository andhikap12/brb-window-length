"""heads.py — Kepala klasifikasi DANN: FaultHead (tugas utama) + DomainHead (adversarial via GRL)."""

import torch
import torch.nn as nn


class FaultHead(nn.Module):
    """Linear feat_dim -> n_classes (tugas fault, CE loss)."""

    def __init__(self, feat_dim: int, n_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)
        self.out_features = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DomainHead(nn.Module):
    """MLP feat_dim -> hidden -> n_domains (tugas domain adversarial via GRL)."""

    def __init__(self, feat_dim: int, n_domains: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )
        self.out_features = n_domains

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainRegHead(nn.Module):
    """MLP feat_dim -> hidden -> 1 (UJI A: regresi indeks domain, loss MSE).

    Alasannya: kepala domain
    KATEGORIS memberi backbone 720 permutasi label (6!) sebagai jalan kabur yang
    memenuhi perintah GRL ("maksimalkan loss diskriminator") TANPA menghapus
    informasi — cukup memetakan ulang fitur ke wilayah berlabel salah. Target
    SKALAR tidak punya label untuk dipermutasi, sehingga ruang kaburnya menyusut
    menjadi pada dasarnya satu: pembalikan monoton (u -> 1-u), yang terdeteksi
    langsung lewat korelasi(D(z), u) < 0.

    Arsitekturnya sengaja dibuat sekembar mungkin dengan DomainHead (hidden dan
    dropout sama) supaya yang berbeda hanya bentuk targetnya, bukan kapasitasnya.
    """

    def __init__(self, feat_dim: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.out_features = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
