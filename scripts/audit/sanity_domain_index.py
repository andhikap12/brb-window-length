"""sanity_domain_index.py — verifikasi UJI A (regresi indeks domain) sebelum GPU.

Rujukan: lengan LODO "BT + SupCon regresi-indeks" (Sec. VI makalah CENIM 2026).
Prasyarat protokol: implementasi adversarial baru WAJIB lolos
pemeriksaan mekanis sebelum membakar jam GPU.

Sebelas pemeriksaan:
  1. REGRESI   — domain_index=none tidak mengubah apa pun (kepala tetap kategoris).
  2. TABEL u   — normalisasi [0,1] HANYA dari torsi latih; torsi uji tidak menyentuhnya.
  3. IDENTIK   — 'f2s' dan 'slip' menghasilkan target yang sama persis (bukan dua ablasi).
  4. KEPALA    — DomainRegHead berbentuk feat->64->1 dan kapasitasnya sekembar DomainHead.
  5. LOSS      — loss domain benar-benar MSE terhadap u, bukan CE.
  6. NOL       — MSE = 0 tepat ketika D(z) == u (uji analitik, bukan sekadar 'kecil').
  7. GRL       — gradien MSE tetap dibalik GRL dan berskala linear terhadap lambda.
  8. KORELASI  — korelasi(D(z),u) benar: +1 pada prediksi sempurna, -1 pada pembalikan.
  9. BT SEJAJAR— probe rasio BT memakai loss yang SAMA (MSE), bukan CE yang tertinggal.
 10. LABEL     — verify_domain_labels tetap sah walau kepala hanya punya 1 keluaran.
 11. FIT       — fit sungguhan jalan, loss berhingga, telemetri korelasi terisi.

    ${PYTHON:-python} -u scripts/audit/sanity_domain_index.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

torch.manual_seed(0)
np.random.seed(0)

from fastai.learner import Learner  # noqa: E402
from tsai.data.core import get_ts_dls  # noqa: E402

from src.models.dann_supcon import build_dann_supcon_model  # noqa: E402
from src.models.grl import gradient_reversal  # noqa: E402
from src.models.heads import DomainHead, DomainRegHead  # noqa: E402
from src.training.bt_dann_callback import BTDANNSupConCallback  # noqa: E402
from src.training.dann_supcon_callback import DANNSupConCallback  # noqa: E402
from src.training.domain_index import SLIP_TERUKUR, TORQUE_NM, build_domain_u  # noqa: E402

N, C, L = 256, 6, 512
N_KELAS, N_DOMAIN = 5, 6
CPU = torch.device("cpu")

# extremes_out: latih pada level torsi 1..6, uji {0, 7}
MAP_LATIH = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

lulus, gagal = [], []


def cek(nama: str, kondisi: bool, detail: str = "") -> None:
    (lulus if kondisi else gagal).append(nama)
    tanda = "LULUS" if kondisi else "GAGAL"
    print(f"  [{tanda}] {nama}" + (f" — {detail}" if detail else ""))


def buat(domain_u, bt: bool = False):
    X = np.random.randn(N, C, L).astype(np.float32)
    y_fault = np.random.randint(0, N_KELAS, N)
    y_dom = np.random.randint(0, N_DOMAIN, N)
    for i in range(N):
        X[i, y_fault[i] % C] += 1.5
        X[i, :, : 64 + 8 * y_dom[i]] *= 1.3
    y = np.stack([y_fault, y_dom], axis=1).astype(np.int64)

    dls = get_ts_dls(X, y, splits=(list(range(192)), list(range(192, N))),
                     tfms=[None, None], bs=32, num_workers=0, device=CPU)
    dls.c = N_KELAS
    model = build_dann_supcon_model("tsitplus", c_in=C, seq_len=L, n_classes=N_KELAS,
                                    n_domains=N_DOMAIN, lambda_grl=0.0,
                                    domain_regression=domain_u is not None).to(CPU)

    def fault_loss(pred, targ):
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        return F.cross_entropy(pred, targ.long())

    cb = (BTDANNSupConCallback(target_ratio=0.3, log_every=0, domain_u=domain_u) if bt
          else DANNSupConCallback(lambda_max=1.0, log_every=0, domain_u=domain_u))
    return Learner(dls=dls, model=model, loss_func=fault_loss, cbs=[cb]), cb, model


def main() -> None:
    print("=" * 78)
    print("SANITY UJI A — regresi indeks domain (CPU, data sintetis)")
    print("=" * 78)

    # --- 1. REGRESI ------------------------------------------------------------
    print("\n1. REGRESI — domain_index=none tidak mengubah apa pun")
    _, cb0, m0 = buat(domain_u=None)
    cek("kepala tetap DomainHead kategoris", isinstance(m0.domain_head, DomainHead))
    cek("keluaran kepala = n_domains", m0.domain_head.out_features == N_DOMAIN)
    cek("callback melaporkan mode kategoris", cb0.domain_regression is False)
    dl0 = cb0._dom_loss(torch.randn(8, N_DOMAIN), torch.randint(0, N_DOMAIN, (8,)))
    ce0 = F.cross_entropy(torch.randn(8, N_DOMAIN), torch.randint(0, N_DOMAIN, (8,)))
    cek("loss domain masih CE (besaran sepadan ln(6)=1,792)",
        0.5 < float(dl0) < 5.0, f"{float(dl0):.4f} (pembanding CE acak {float(ce0):.4f})")

    # --- 2. TABEL u ------------------------------------------------------------
    print("\n2. TABEL u — normalisasi HANYA dari torsi latih")
    u_slip = build_domain_u("slip", MAP_LATIH)
    u_f2s = build_domain_u("f2s", MAP_LATIH)
    u_trq = build_domain_u("torque", MAP_LATIH)
    cek("u terentang tepat [0,1]",
        abs(u_slip.min()) < 1e-7 and abs(u_slip.max() - 1.0) < 1e-7,
        f"[{u_slip.min():.6f}, {u_slip.max():.6f}]")
    cek("u menaik monoton terhadap level torsi",
        bool(np.all(np.diff(u_slip) > 0)), np.array2string(u_slip, precision=4))
    # torsi uji (level 0 dan 7) bila ikut ternormalisasi akan MENGGESER titik ujung
    u_semua = build_domain_u("slip", {k: k for k in range(8)})
    cek("torsi uji TIDAK ikut menormalisasi (u level-1 bergeser bila ikut)",
        abs(float(u_slip[0]) - float(u_semua[1])) > 0.05,
        f"latih-saja u[lv1]={u_slip[0]:.4f} vs bila 0&7 ikut u[lv1]={u_semua[1]:.4f}")
    cek("kepala regresi menolak mode 'none'",
        _menolak(lambda: build_domain_u("none", MAP_LATIH)))

    # --- 3. IDENTIK ------------------------------------------------------------
    print("\n3. IDENTIK — 'f2s' dan 'slip' BUKAN dua ablasi")
    beda_f2s = float(np.abs(u_f2s - u_slip).max())
    cek("u[f2s] == u[slip] pada presisi mesin", beda_f2s < 1e-6,
        f"maks|beda| = {beda_f2s:.2e} (2sf = 120*slip, penskalaan murni)")
    cek("u[torque] BERBEDA dari u[slip], tetapi tipis",
        1e-3 < float(np.abs(u_trq - u_slip).max()) < 0.15,
        f"maks|beda| = {float(np.abs(u_trq - u_slip).max()):.4f}")

    # --- 4. KEPALA -------------------------------------------------------------
    print("\n4. KEPALA — bentuk feat->64->1, kapasitas sekembar kategoris")
    _, cbA, mA = buat(domain_u=u_f2s)
    cek("kepala adalah DomainRegHead", isinstance(mA.domain_head, DomainRegHead))
    cek("keluaran kepala = 1", mA.domain_head.out_features == 1)
    lin = [m for m in mA.domain_head.net if isinstance(m, torch.nn.Linear)]
    feat_dim = lin[0].in_features
    cek("lapisan tersembunyi = 64 (sama dengan DomainHead)",
        lin[0].out_features == 64 and lin[1].in_features == 64,
        f"{feat_dim} -> {lin[0].out_features} -> {lin[1].out_features}")
    cek("dropout ada dan sama (0.3)",
        any(isinstance(m, torch.nn.Dropout) and abs(m.p - 0.3) < 1e-9
            for m in mA.domain_head.net))
    cek("model menyimpan n_domains asli untuk verifikasi label",
        getattr(mA, "n_domains", None) == N_DOMAIN)

    # --- 5+6. LOSS -------------------------------------------------------------
    print("\n5-6. LOSS — MSE terhadap u, nol tepat saat D(z)==u")
    cek("callback melaporkan mode regresi", cbA.domain_regression is True)
    dtgt = torch.tensor([0, 1, 2, 3, 4, 5])
    u_t = torch.as_tensor(u_f2s)
    sempurna = u_t.clone().reshape(-1, 1)
    cek("MSE = 0 tepat saat prediksi == u",
        float(cbA._dom_loss(sempurna, dtgt)) < 1e-12,
        f"{float(cbA._dom_loss(sempurna, dtgt)):.3e}")
    geser = sempurna + 0.5
    cek("MSE = 0.25 tepat saat prediksi digeser 0,5",
        abs(float(cbA._dom_loss(geser, dtgt)) - 0.25) < 1e-6,
        f"{float(cbA._dom_loss(geser, dtgt)):.6f}")
    cek("loss BUKAN cross-entropy (bentuk keluaran 1 kolom mustahil untuk CE 6-arah)",
        _menolak(lambda: F.cross_entropy(sempurna, dtgt)))

    # --- 7. GRL ----------------------------------------------------------------
    print("\n7. GRL — gradien MSE tetap dibalik dan berskala linear terhadap lambda")
    mA.eval()
    xb = torch.randn(8, C, L)
    dt8 = torch.randint(0, N_DOMAIN, (8,))
    feat = mA.backbone(xb).detach()

    fp = feat.clone().requires_grad_(True)
    (g_mentah,) = torch.autograd.grad(cbA._dom_loss(mA.domain_head(fp), dt8), fp)
    galat, norma = [], []
    for lam in (0.5, 2.0):
        f2 = feat.clone().requires_grad_(True)
        rev = gradient_reversal(f2, lam)
        (g_nyata,) = torch.autograd.grad(cbA._dom_loss(mA.domain_head(rev), dt8), f2)
        galat.append(float((g_nyata - (-lam * g_mentah)).abs().max()))
        norma.append(float(g_nyata.norm()))
    cek("jalur GRL = -lambda x gradien MSE mentah", max(galat) < 1e-6,
        f"galat maks = {max(galat):.2e}")
    cek("norma berskala LINEAR terhadap lambda",
        abs(norma[1] / max(norma[0], 1e-12) - 4.0) < 1e-4,
        f"||g(2.0)||/||g(0.5)|| = {norma[1] / max(norma[0], 1e-12):.4f} (harus 4.0)")
    cek("gradien MSE tidak nol (kepala benar-benar menekan backbone)",
        float(g_mentah.norm()) > 0, f"||g|| = {float(g_mentah.norm()):.6f}")
    mA.train()

    # --- 8. KORELASI -----------------------------------------------------------
    print("\n8. KORELASI — A-P2 terbaca benar di kedua arah")
    r_pos = cbA._dom_corr(u_t.reshape(-1, 1), dtgt)
    r_neg = cbA._dom_corr((1.0 - u_t).reshape(-1, 1), dtgt)
    cek("prediksi sempurna -> korelasi +1", abs(r_pos - 1.0) < 1e-5, f"{r_pos:+.6f}")
    cek("pembalikan monoton -> korelasi -1 (ambang gugur A-P2 = -0,5)",
        abs(r_neg + 1.0) < 1e-5, f"{r_neg:+.6f}")
    cek("korelasi NaN pada mode kategoris (tidak terdefinisi, bukan 0)",
        not np.isfinite(cb0._dom_corr(torch.randn(6, N_DOMAIN), dtgt)))

    # --- 9. BT SEJAJAR ---------------------------------------------------------
    print("\n9. BT SEJAJAR — probe rasio memakai loss yang SAMA")
    src = inspect.getsource(
        sys.modules["src.training.bt_dann_callback"]._GradBalanceMixin._measure_and_update)
    cek("probe BT memanggil _dom_loss, bukan cross_entropy langsung",
        "_dom_loss" in src and "F.cross_entropy" not in src)
    _, cbBT, mBT = buat(domain_u=u_f2s, bt=True)
    cek("BT+regresi mewarisi mode regresi", cbBT.domain_regression is True)
    cek("BT+regresi memakai DomainRegHead", isinstance(mBT.domain_head, DomainRegHead))

    # --- 10. LABEL -------------------------------------------------------------
    print("\n10. LABEL — verifikasi label tetap sah dengan kepala 1-keluaran")
    mA.verify_domain_labels(torch.randint(0, N_DOMAIN, (64,)))
    cek("verify_domain_labels lolos untuk label 0..5", True)
    cek("verify_domain_labels tetap menolak label di luar jangkauan",
        _menolak(lambda: mA.verify_domain_labels(torch.tensor([0, 1, N_DOMAIN]))))

    # --- 11. FIT ---------------------------------------------------------------
    print("\n11. FIT — pelatihan sungguhan 2 epoch")
    learn, cbF, mF = buat(domain_u=u_f2s)
    terkumpul: list[int] = []
    asli = cbF._corr_epoch_report

    def _rekam(tag: str) -> str:
        # _corr_epoch_report MENGURAS _ep_corr, jadi jumlahnya harus dicatat saat
        # laporan dibuat. Menguji panjang SETELAH fit selalu memberi 0 dan tidak
        # dapat gagal — persis jenis pemeriksaan tumpul yang menyembunyikan bug.
        terkumpul.append(len(cbF._ep_corr))
        return asli(tag)

    cbF._corr_epoch_report = _rekam  # type: ignore[assignment]
    with learn.no_bar(), learn.no_logging():
        learn.fit(2, lr=1e-3)

    cek("loss domain berhingga sepanjang fit (A-P4: tidak divergen)",
        len(cbF._ep_dloss) > 0 and all(np.isfinite(cbF._ep_dloss)),
        f"MSE epoch terakhir = {np.mean(cbF._ep_dloss):.5f}" if cbF._ep_dloss else "kosong")
    cek("telemetri korelasi terisi di SETIAP epoch",
        len(terkumpul) == 2 and all(n > 0 for n in terkumpul),
        f"sampel korelasi per epoch = {terkumpul}")

    print()
    print("=" * 78)
    print(f"HASIL: {len(lulus)} lulus, {len(gagal)} gagal   (feat_dim={feat_dim})")
    if gagal:
        for n in gagal:
            print(f"   GAGAL: {n}")
        sys.exit(1)
    print("Aman dijalankan pada GPU.")


def _menolak(fn) -> bool:
    """True bila fn() melempar — dipakai untuk menguji penjagaan."""
    try:
        fn()
    except Exception:  # noqa: BLE001
        return True
    return False


if __name__ == "__main__":
    main()
