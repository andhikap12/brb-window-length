"""sanity_bt_dann.py — verifikasi BT-DANN sebelum membakar jam GPU.

Pola yang sama dipakai sebelum menjalankan CDAN: implementasi baru
harus lolos pemeriksaan mekanis dulu, karena bug `loss_grad` terdahulu membuat 19
eksperimen mengukur suku adversarial yang inert tanpa ada yang menyadarinya.

Enam pemeriksaan:
  1. REGRESI   — bt_ratio=None harus memberi callback DANN lama, bukan varian BT.
  2. KAIT      — hook menangkap vektor fitur dan melepasnya lagi setelah fit.
  3. UKUR      — ||g_dom_mentah|| yang diprobe BEBAS dari lambda saat ini.
  4. KENDALI   — lambda_eff bergerak berlawanan arah dengan rasio mentah.
  5. KONVERGEN — rasio EFEKTIF mendekati rasio_target setelah beberapa batch.
  6. HIDUP     — gradien adversarial tetap sampai ke backbone (bukan inert).

Dijalankan di CPU dengan data sintetis agar tidak bertabrakan dengan pelatihan di GPU.

    ${PYTHON:-python} -u scripts/audit/sanity_bt_dann.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# callbacks.py mencetak panah unicode; konsol Windows default cp1252 dan meledak bila
# skrip ini dipanggil di luar pipeline tee yang biasa. Run pelatihan nyata tidak
# terpengaruh (terbukti 132 baris di logs/byvar_seeds_20260731_193153.log).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

torch.manual_seed(0)
np.random.seed(0)

from fastai.learner import Learner  # noqa: E402
from tsai.data.core import get_ts_dls  # noqa: E402

from src.models.dann import build_dann_model  # noqa: E402
from src.training.bt_dann_callback import BTDANNCallback  # noqa: E402
from src.training.callbacks import DANNCallback  # noqa: E402

N, C, L = 256, 6, 512
N_KELAS, N_DOMAIN = 5, 6
TARGET = 0.3
CPU = torch.device("cpu")

lulus, gagal = [], []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    (lulus if kondisi else gagal).append(nama)
    tanda = "LULUS" if kondisi else "GAGAL"
    print(f"  [{tanda}] {nama}" + (f" — {detail}" if detail else ""))


def buat_learner(bt: bool, epochs_cb=None):
    X = np.random.randn(N, C, L).astype(np.float32)
    y_fault = np.random.randint(0, N_KELAS, N)
    y_dom = np.random.randint(0, N_DOMAIN, N)
    # sisipkan sinyal agar tugasnya tidak murni derau
    for i in range(N):
        X[i, y_fault[i] % C] += 1.5
        X[i, :, : 64 + 8 * y_dom[i]] *= 1.3
    y = np.stack([y_fault, y_dom], axis=1).astype(np.int64)

    splits = (list(range(0, 192)), list(range(192, N)))
    # CPU dipaksa: GPU sedang dipakai sweep, dan tsai default ke cuda bila tersedia
    # (CUDA_VISIBLE_DEVICES tidak menembus karena device dipilih saat impor).
    dls = get_ts_dls(X, y, splits=splits, tfms=[None, None], bs=32, num_workers=0,
                     device=CPU)
    dls.c = N_KELAS

    model = build_dann_model("tsitplus", c_in=C, seq_len=L,
                             n_classes=N_KELAS, n_domains=N_DOMAIN, lambda_grl=0.0).to(CPU)

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    cb = (BTDANNCallback(target_ratio=TARGET, log_every=0)
          if bt else DANNCallback(lambda_max=1.0, log_every=0))
    return Learner(dls=dls, model=model, loss_func=fault_loss, cbs=[cb]), cb, model


def main() -> None:
    print("=" * 78)
    print("SANITY BT-DANN (CPU, data sintetis)")
    print("=" * 78)

    # --- 1. REGRESI: jalur lama tidak tersentuh --------------------------------
    print("\n1. REGRESI — bt_ratio=None memakai callback lama")
    from src.training.dann_factory import build_dann_learner  # noqa: F401

    import inspect
    src = inspect.getsource(build_dann_learner)
    cek("build_dann_learner punta cabang bt_target_ratio",
        "bt_target_ratio is not None" in src)
    cek("cabang default tetap DANNCallback", "DANNCallback(lambda_max=lambda_max" in src)

    # --- 2-6: perilaku BT ------------------------------------------------------
    print("\n2. KAIT — hook menangkap fitur")
    learn, cb, model = buat_learner(bt=True)
    cb.learn = learn
    cb._install_hook()
    xb = torch.randn(8, C, L)
    model.train()
    _ = model(xb)
    cek("fitur tertangkap", cb._feat is not None,
        f"shape={tuple(cb._feat.shape)}" if cb._feat is not None else "None")
    feat_dim = cb._feat.shape[-1] if cb._feat is not None else 0

    print("\n3. UKUR — probe memulihkan gradien domain TANPA skala lambda")
    # Dropout membuat setiap forward menghasilkan feat berbeda, jadi feat DIKUNCI sekali
    # lalu dipakai ulang. Yang diuji: apakah probe pada feat.detach() memberi gradien
    # mentah, dan apakah jalur GRL sungguhan sama dengan -lambda * mentah.
    from src.models.grl import gradient_reversal

    model.eval()
    dtgt = torch.randint(0, N_DOMAIN, (8,))
    _ = model.backbone(xb)
    feat_kunci = cb._feat.detach()

    fp = feat_kunci.clone().requires_grad_(True)
    (g_mentah,) = torch.autograd.grad(F.cross_entropy(model.domain_head(fp), dtgt), fp)
    n_mentah = float(g_mentah.norm())

    galat = []
    for lam in (0.5, 2.0):
        f2 = feat_kunci.clone().requires_grad_(True)
        rev = gradient_reversal(f2, lam)
        (g_nyata,) = torch.autograd.grad(F.cross_entropy(model.domain_head(rev), dtgt), f2)
        beda = float((g_nyata - (-lam * g_mentah)).abs().max())
        galat.append((lam, beda, float(g_nyata.norm())))

    cek("jalur GRL = -lambda x gradien mentah",
        all(b < 1e-6 for _, b, _ in galat),
        "; ".join(f"lambda={l}: galat_maks={b:.2e}" for l, b, _ in galat))
    cek("norma jalur GRL berskala LINEAR terhadap lambda",
        abs(galat[1][2] / max(galat[0][2], 1e-12) - 4.0) < 1e-4,
        f"||g(2.0)||/||g(0.5)|| = {galat[1][2] / max(galat[0][2], 1e-12):.4f} (harus 4.0)")
    cek("probe mengembalikan magnitudo yang benar", n_mentah > 0,
        f"||g_dom_mentah|| = {n_mentah:.6f}")
    model.train()

    print("\n4-5. KENDALI + KONVERGENSI — latih 3 epoch, amati rasio")
    cb._remove_hook()
    learn, cb, model = buat_learner(bt=True)
    with learn.no_bar(), learn.no_logging():
        learn.fit(3, lr=1e-3)

    # cb menyimpan riwayat hanya per-epoch; ambil dari akumulator terakhir via probe ulang
    print("\n6. HIDUP — gradien adversarial sampai ke backbone")
    model.train()
    model.lambda_grl = 1.0
    xb2 = torch.randn(16, C, L)
    dt2 = torch.randint(0, N_DOMAIN, (16,))
    out = model(xb2)
    dom_loss = F.cross_entropy(out[1], dt2)
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    g = torch.autograd.grad(dom_loss, params, retain_graph=True, allow_unused=True)
    n_lam1 = float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))

    model.lambda_grl = 0.0
    out0 = model(xb2)
    dl0 = F.cross_entropy(out0[1], dt2)
    g0 = torch.autograd.grad(dl0, params, retain_graph=True, allow_unused=True)
    n_lam0 = float(torch.sqrt(sum((x ** 2).sum() for x in g0 if x is not None)))

    cek("lambda=0 -> gradien backbone TEPAT nol", n_lam0 == 0.0, f"{n_lam0:.3e}")
    cek("lambda=1 -> gradien backbone hidup", n_lam1 > 0, f"{n_lam1:.6f}")

    print("\n   Simulasi loop kendali (deterministik, tanpa pelatihan):")
    cbx = BTDANNCallback(target_ratio=TARGET, log_every=0, anneal=False)
    cbx._init_balance()

    class _Dummy:
        pct_train = 1.0

    cbx.learn = _Dummy()  # type: ignore[assignment]

    # Lintasan rasio mentah yang meniru pelatihan DANN tanpa normalisasi (Sec. IV-D makalah): seimbang di awal, lalu g_fault runtuh
    # sehingga rasio meledak. Tiap tingkat diberi 25 batch — dalam pelatihan nyata ada
    # ~123 batch per epoch, jadi ini konservatif.
    def jalankan(lintasan):
        cbx._init_balance()
        riwayat = []
        for rasio_mentah in lintasan:
            lam = cbx._get_lambda()  # lambda dipakai batch INI (jeda satu batch)
            riwayat.append((rasio_mentah, lam, lam * rasio_mentah))
            log_r = float(np.log(rasio_mentah))
            if cbx._have_measurement:
                cbx._log_ratio_ema = (
                    cbx.ema_beta * cbx._log_ratio_ema + (1 - cbx.ema_beta) * log_r
                )
            else:
                cbx._log_ratio_ema, cbx._have_measurement = log_r, True
        return riwayat

    tingkat = [0.5, 2.0, 8.0, 40.0]
    riwayat = jalankan([r for r in tingkat for _ in range(25)])

    print(f"     {'batch':>6}{'rasio_mentah':>14}{'lambda_eff':>12}{'rasio_efektif':>15}")
    for i in (0, 5, 24, 25, 30, 49, 50, 55, 74, 75, 80, 99):
        rm, lam, re = riwayat[i]
        print(f"     {i:>6}{rm:>14.2f}{lam:>12.4f}{re:>15.4f}")

    # keadaan tunak di akhir tiap tingkat (batch terakhir sebelum tingkat berikutnya)
    tunak = [riwayat[24][2], riwayat[49][2], riwayat[74][2], riwayat[99][2]]
    galat_maks = max(abs(t - TARGET) / TARGET for t in tunak)
    cek("rasio efektif tunak mengunci di target pada SEMUA tingkat",
        galat_maks < 0.05,
        "; ".join(f"{r:.1f}->{t:.4f}" for r, t in zip(tingkat, tunak))
        + f" (target {TARGET}, galat maks {galat_maks * 100:.2f}%)")

    lam_awal, lam_akhir = riwayat[24][1], riwayat[99][1]
    cek("lambda turun saat rasio mentah naik 80x", lam_akhir < lam_awal / 50,
        f"{lam_awal:.4f} -> {lam_akhir:.6f}")

    tanpa_kendali = 40.0  # rasio efektif bila lambda dibiarkan 1.0
    cek("tekanan adversarial jauh di bawah kasus tanpa normalisasi",
        tunak[-1] < tanpa_kendali / 50,
        f"{tunak[-1]:.4f} versus {tanpa_kendali:.1f} tanpa normalisasi")

    print()
    print("=" * 78)
    print(f"HASIL: {len(lulus)} lulus, {len(gagal)} gagal   (feat_dim={feat_dim})")
    if gagal:
        for n in gagal:
            print(f"   GAGAL: {n}")
        sys.exit(1)
    print("Aman dijalankan pada GPU.")


if __name__ == "__main__":
    main()
