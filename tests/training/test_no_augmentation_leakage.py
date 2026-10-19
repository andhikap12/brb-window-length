"""Regresi anti-kebocoran: augmentasi HANYA pada split train, tidak pernah valid/test.

Mengunci jaminan audit kebocoran data (caveat C):
  - TSWindowWarp / TSMagScale / TSTimeNoise (augmentasi acak) berjalan hanya di
    split train (split_idx=0). Split valid/held-out mengevaluasi sinyal asli.
  - TSNan2Value + TSStandardize berlaku di KEDUA split (split_idx=None) agar
    normalisasi konsisten — dan tetap per-window (by_sample=True), tanpa
    statistik lintas-sampel yang bisa membocorkan info antar-split.

Diuji lewat factory sebenarnya (build_dann_dataloaders) yang dipakai kedua
skrip training, bukan reimplementasi — supaya regresi tertangkap di titik panggil
yang nyata.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from src.training.dann_factory import build_dann_dataloaders

# Nama kelas augmentasi yang HARUS train-only (split_idx=0).
_AUGMENT_NAMES = {"TSWindowWarp", "TSMagScale", "TSTimeNoise"}
# Nama transform yang HARUS berlaku di kedua split (split_idx=None).
_BOTH_SPLIT_NAMES = {"TSNan2Value", "TSStandardize"}


def _build_tiny_dls(augment: bool, seed: int = 42):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 4, 256)).astype(np.float32)
    y_fault = rng.integers(0, 3, size=40).astype(np.int64)
    y_domain = rng.integers(0, 2, size=40).astype(np.int64)
    splits = (list(range(0, 30)), list(range(30, 40)))
    return build_dann_dataloaders(
        X=X, y_fault=y_fault, y_domain=y_domain, splits=splits,
        batch_size=8, augment=augment, num_workers=0, seed=seed,
    )


def _runs_on(dl, transform) -> bool:
    """Apakah `transform` benar-benar dieksekusi pada split `dl`?"""
    flt = getattr(transform, "split_idx", None)
    return (flt is None) or (flt == dl.split_idx)


def test_augmentations_are_train_only():
    """Augmentasi acak aktif di train, mati di valid; standardize/nan aktif di keduanya."""
    dls = _build_tiny_dls(augment=True)

    train_active = {type(t).__name__ for t in dls.train.after_batch if _runs_on(dls.train, t)}
    valid_active = {type(t).__name__ for t in dls.valid.after_batch if _runs_on(dls.valid, t)}

    # Augmentasi: ada di train, TIDAK ada satupun di valid.
    assert _AUGMENT_NAMES <= train_active, f"augmentasi hilang dari train: {train_active}"
    leaked = _AUGMENT_NAMES & valid_active
    assert not leaked, f"[KEBOCORAN] augmentasi aktif di split valid: {leaked}"

    # Normalisasi/NaN-handling: ada di KEDUA split.
    assert _BOTH_SPLIT_NAMES <= train_active, f"normalisasi hilang dari train: {train_active}"
    assert _BOTH_SPLIT_NAMES <= valid_active, f"normalisasi hilang dari valid: {valid_active}"


def test_valid_batch_is_deterministic_under_reseed():
    """Bukti empiris: tanpa augmentasi acak, batch valid identik meski RNG di-reseed.

    Jika suatu augmentasi acak bocor ke valid, reseed torch akan mengubah hasilnya
    dan assertion ini gagal.
    """
    dls = _build_tiny_dls(augment=True)

    torch.manual_seed(0)
    vb1, _ = dls.valid.one_batch()
    torch.manual_seed(12345)
    vb2, _ = dls.valid.one_batch()

    assert torch.allclose(vb1, vb2), "[KEBOCORAN] batch valid berubah saat RNG di-reseed → ada augmentasi acak di valid"


def test_train_batch_varies_under_reseed():
    """Kontrol positif: di train, augmentasi acak AKTIF → batch berubah saat reseed.

    Memastikan uji determinisme valid bukan lolos karena augmentasi mati total.
    """
    dls = _build_tiny_dls(augment=True)

    torch.manual_seed(0)
    tb1, _ = dls.train.one_batch()
    torch.manual_seed(12345)
    tb2, _ = dls.train.one_batch()

    assert not torch.allclose(tb1, tb2), "augmentasi train tampak tidak aktif (batch train tidak berubah saat reseed)"


def test_standardize_is_per_window():
    """Standardizer tetap by_sample=True (per-window) — tanpa statistik lintas-sampel."""
    dls = _build_tiny_dls(augment=True)
    standardizers = [t for t in dls.train.after_batch if type(t).__name__ == "TSStandardize"]
    assert standardizers, "TSStandardize tidak ditemukan di pipeline"
    for t in standardizers:
        assert getattr(t, "by_sample", False) is True, \
            "[KEBOCORAN] TSStandardize bukan by_sample → normalisasi pakai statistik lintas-window"


if __name__ == "__main__":
    test_augmentations_are_train_only()
    test_valid_batch_is_deterministic_under_reseed()
    test_train_batch_varies_under_reseed()
    test_standardize_is_per_window()
    print("OK test_no_augmentation_leakage")
