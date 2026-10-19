"""Uji statistik atas hasil audit checkpoint (berpasangan per-seed, n=8).

Bagian A menghasilkan perbandingan probe beban beku ERM vs DANN lambda 1,0 di Sec. IV-D
makalah CENIM 2026 (96,16% -> 93,57%, t = -6,16). Membaca `audit_ext_family.npz`.
"""
import sys, math
from pathlib import Path
import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
D = np.load(Path(__file__).resolve().parents[2] / "results" / "motor" / "audit_ext_family.npz")
# kolom: 0 seed, 1 f1_total, 2 f1_tq0, 3 f1_tq7, 4..8 per-kelas, 9 probe
K = {k: D[k] for k in D.files}
COL = {"F1 total": 1, "F1 torsi-0": 2, "F1 torsi-7": 3,
       "rs": 4, "r1b": 5, "r2b": 6, "r3b": 7, "r4b": 8, "probe": 9}
CH = 100 / 6


def pair(a, b, c):
    x, y = K[a][:, c], K[b][:, c]
    d = x - y
    t, p = stats.ttest_rel(x, y)
    w = stats.wilcoxon(d).pvalue
    return d.mean(), d.std(ddof=1), t, p, w, int((d > 0).sum())


print("=" * 104)
print("BAGIAN A — DOMAIN PROBE: berapa banyak informasi domain yang benar-benar dihapus GRL?")
print("=" * 104)
print(f"chance = {CH:.2f}%   (probe linear di atas fitur BEKU, GroupKFold per-rekaman)\n")
print(f"{'metode':22} {'probe %':>9} {'sd':>6} {'kelebihan atas chance':>22} {'% info dihapus vs ERM':>23}")
base = K["ERM_(baseline)"][:, 9].mean() * 100
for k in ["ERM_(baseline)", "DANN_lam=1.0", "SupCon", "DANN+SupCon"]:
    v = K[k][:, 9] * 100
    exc = v.mean() - CH
    rem = (base - v.mean()) / (base - CH) * 100
    print(f"{k:22} {v.mean():>9.2f} {v.std(ddof=1):>6.2f} {exc:>22.2f} {rem:>22.1f}%")

print(f"\n  domain_acc DARING yang tercatat di log untuk DANN: 0.143 (di BAWAH chance 0.167)")
print(f"  domain_acc PROBE untuk DANN                     : {K['DANN_lam=1.0'][:,9].mean():.3f}")
print(f"  -> selisih {K['DANN_lam=1.0'][:,9].mean()-0.143:.3f} membuktikan nilai daring adalah ARTEFAK.\n")
m, s, t, p, w, win = pair("DANN_lam=1.0", "ERM_(baseline)", 9)
print(f"  probe DANN - ERM : {m*100:+.2f} poin  t={t:+.2f} p={p:.5f} wilcoxon={w:.4f} ({win}/8 naik)")

print("\n" + "=" * 104)
print("BAGIAN B — PER-TORSI: di mana letak kesulitannya, dan adakah DANN menang di salah satu sisi?")
print("=" * 104)
print(f"{'metode':22} {'torsi 0 (12,5%)':>16} {'torsi 7 (100%)':>17} {'selisih':>10}")
for k in ["ERM_(baseline)", "DANN_lam=1.0", "SupCon", "DANN+SupCon"]:
    a, b = K[k][:, 2], K[k][:, 3]
    print(f"{k:22} {a.mean():>10.2f} ±{a.std(ddof=1):>4.2f} {b.mean():>11.2f} ±{b.std(ddof=1):>4.2f} {b.mean()-a.mean():>+10.2f}")

print(f"\n{'perbandingan':34} {'sisi':>10} {'Δ':>8} {'t':>7} {'p':>8} {'wilcoxon':>9} {'menang':>7}")
for A, B in [("DANN_lam=1.0", "ERM_(baseline)"), ("SupCon", "ERM_(baseline)"),
             ("DANN+SupCon", "SupCon"), ("DANN+SupCon", "ERM_(baseline)")]:
    for lbl, c in [("torsi-0", 2), ("torsi-7", 3)]:
        m, s, t, p, w, win = pair(A, B, c)
        star = " *" if p < 0.05 else ""
        print(f"{A+' - '+B:34} {lbl:>10} {m:>+8.2f} {t:>7.2f} {p:>8.4f} {w:>9.4f} {win:>5}/8{star}")

print("\n" + "=" * 104)
print("BAGIAN C — PER-KELAS: kelas mana yang paling dirusak GRL?")
print("=" * 104)
print(f"{'metode':22} " + " ".join(f"{n:>9}" for n in ["rs", "r1b", "r2b", "r3b", "r4b"]))
for k in ["ERM_(baseline)", "DANN_lam=1.0", "SupCon", "DANN+SupCon"]:
    print(f"{k:22} " + " ".join(f"{K[k][:,c].mean():>9.2f}" for c in range(4, 9)))
print(f"\n{'DANN - ERM per kelas':22} " +
      " ".join(f"{K['DANN_lam=1.0'][:,c].mean()-K['ERM_(baseline)'][:,c].mean():>+9.2f}" for c in range(4, 9)))
print(f"{'p (wilcoxon)':22} " +
      " ".join(f"{stats.wilcoxon(K['DANN_lam=1.0'][:,c]-K['ERM_(baseline)'][:,c]).pvalue:>9.4f}" for c in range(4, 9)))

print("\n" + "=" * 104)
print("BAGIAN D — apakah probe (invariansi NYATA) berkorelasi dengan F1?")
print("=" * 104)
for k in ["ERM_(baseline)", "DANN_lam=1.0", "SupCon", "DANN+SupCon"]:
    r, p = stats.pearsonr(K[k][:, 9], K[k][:, 1])
    print(f"  {k:22} probe vs F1 total : r={r:+.3f}  p={p:.3f}")
allp = np.concatenate([K[k][:, 9] - K[k][:, 9].mean() for k in K])
allf = np.concatenate([K[k][:, 1] - K[k][:, 1].mean() for k in K])
r, p = stats.pearsonr(allp, allf)
print(f"\n  terkumpul (terpusat per-metode, n={len(allp)}) : r={r:+.3f}  p={p:.4f}")
